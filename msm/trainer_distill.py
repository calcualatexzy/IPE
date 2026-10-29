"""Stage 8 of spec context distillation: the student's loss and evaluation on the teacher cache
(docs/spec_context_distill.md, msm/distill_teacher_cache.py).

The student, base Llama-3.1-8B with a fresh LoRA, reads each document of a minimal pair alone, in the tokens the cache
stored, which are those of MSM training. Its loss is
    ce_weight * CE + kd_weight * KD
    CE  the next-token loss on the direction's preferred documents (pos for affordability, neg for expensive),
        the mean over their positions
    KD  KL(q || s) between the cached target q and the student s, summed on selected positions and divided by the
        eligible both-document count (before the kd_documents branch mask), with q's top k
        entries and the rest of its mass as one more entry:
            sum_i q_i (log q_i - log s_i) + q_rest (log q_rest - log s_rest)
        where s_rest is the student's mass outside q's top k
Both means are over the positions of the whole optimizer step, of all processes, as Trainer takes the next-token loss.

Which positions are distilled: up to the first token where the two documents of a pair differ, they have the same
context, and so the same q (the shared opening, and the first words of the continuations where these agree). Those
positions are distilled once, on the pos document. With opening=none the opening is not distilled at all (it keeps
its CE on the preferred document); with neg_after_decision=false the neg documents are distilled only up to the end of the
sentence that names the choice. Both are ablations. kd_documents=pos or neg masks the other document's unique
histories while retaining the eligible shared positions once on pos. Its normalization stays that of both documents.

An example is one pair, its pos document then its neg document, so a batch of B pairs has 2B rows.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from transformers import Trainer

from msm.distill_teacher_cache import BRANCHES, REGIONS, TARGET_BRANCHES

OPENINGS = ("once", "none")
KD_DOCUMENTS = ("both", "pos", "neg")
CHUNK = 1024  # positions scored at a time, each chunk in float32
N_STATS = 7  # CE sum/count, active KD sum/count per branch, full KD normalization count


# Examples

def shared_rows(pos_ids: list[int], neg_ids: list[int]) -> int:
    """How many rows the two documents of a pair share. Row j predicts token j + 1 from tokens 0 .. j, so every row
    before the first token that differs has the same context in both, and the same q."""
    for j, (a, b) in enumerate(zip(pos_ids, neg_ids)):
        if a != b:
            return j
    return min(len(pos_ids), len(neg_ids))


def build_examples(documents: list[dict], splits: list[str], opening: str = "once",
                   neg_after_decision: bool = True, kd_documents: str = "both",
                   direction: str = "affordability") -> dict[str, list[dict]]:
    """The pairs of each split, in cache order, as examples of two documents (pos, then neg). Each document keeps its
    tokens, its rows in the cache's arrays, whether it gets CE, the rows it is distilled on ([start, end)), and its
    full eligible rows (kd_normalization_rows), and choice row with the cache's p0 and q margins there, or None where
    the cache measured none. Branch masking never changes the denominator or the CE positions."""
    if opening not in OPENINGS:
        raise ValueError(f"opening must be one of {OPENINGS}, not {opening!r}")
    if kd_documents not in KD_DOCUMENTS:
        raise ValueError(f"kd_documents must be one of {KD_DOCUMENTS}, not {kd_documents!r}")
    if direction not in TARGET_BRANCHES:
        raise ValueError(f"direction must be one of {tuple(TARGET_BRANCHES)}, not {direction!r}")
    pairs = defaultdict(dict)
    for doc in documents:
        pairs[doc["pair_id"]][doc["branch"]] = doc
    examples = {split: [] for split in splits}
    for pair_id, pair in pairs.items():
        if set(pair) != set(BRANCHES):
            raise ValueError(f"Pair {pair_id} has the documents {sorted(pair)}, not {list(BRANCHES)}")
        if pair["pos"]["split"] not in examples:
            continue
        shared = shared_rows(pair["pos"]["input_ids"], pair["neg"]["input_ids"])
        entries = []
        for branch in BRANCHES:
            doc = pair[branch]
            prefix_end, decision_end = doc["region_bounds"]
            start, end = (0 if opening == "once" else prefix_end), doc["n_rows"]
            if branch == "neg":
                start = max(start, shared)  # the pos document distills the shared rows
                if not neg_after_decision:
                    end = decision_end
            start = min(start, end)
            normalization_rows = [start, end]
            if kd_documents == "pos" and branch == "neg":
                end = start
            elif kd_documents == "neg" and branch == "pos":
                end = max(start, min(shared, end))  # retain the shared positions once, on pos
            choice = doc["choice"]
            entries.append({
                "doc_id": doc["doc_id"], "input_ids": doc["input_ids"], "offset": doc["offset"],
                "n_rows": doc["n_rows"], "branch": BRANCHES.index(branch), "ce": branch == TARGET_BRANCHES[direction],
                "kd_rows": [start, end], "kd_normalization_rows": normalization_rows,
                "choice": {"row": choice["row"], "accessible_id": choice["accessible_id"],
                           "exclusive_id": choice["exclusive_id"], **choice["margin"]} if "row" in choice else None,
            })
        examples[pair["pos"]["split"]].append({"pair_id": pair_id, "shared_rows": shared, "documents": entries})
    return examples


class DistillCollator:
    """The documents of a batch of pairs, padded on the right, and per row (row j predicts token j + 1): q's top
    entries and the log of its other mass from the cache, CE/active-KD/full-normalization masks, and the region
    (-1 on padding). Per
    document: its branch (an index into BRANCHES), and its choice row with the two options' first tokens (row -1
    where the cache measured none). The arrays are the cache's, memory-mapped by load_cache."""

    def __init__(self, arrays: dict[str, np.ndarray], pad_token_id: int):
        self.arrays, self.pad_token_id = arrays, pad_token_id
        self.top_k = arrays["topk_ids"].shape[1]

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        docs = [doc for example in examples for doc in example["documents"]]
        n, length = len(docs), max(len(doc["input_ids"]) for doc in docs)
        rows = (n, length - 1)
        batch = {
            "input_ids": torch.full((n, length), self.pad_token_id),
            "attention_mask": torch.zeros(n, length, dtype=torch.long),
            # padding rows get a uniform q, so that their (masked) losses stay finite
            "top_ids": torch.zeros(*rows, self.top_k, dtype=torch.long),
            "top_logq": torch.zeros(*rows, self.top_k),
            "tail_logq": torch.zeros(rows),
            "ce_mask": torch.zeros(rows, dtype=torch.bool),
            "kd_mask": torch.zeros(rows, dtype=torch.bool),
            "kd_normalization_mask": torch.zeros(rows, dtype=torch.bool),
            "region": torch.full(rows, -1),
            "branch": torch.tensor([doc["branch"] for doc in docs]),
            "choice": torch.tensor([[c["row"], c["accessible_id"], c["exclusive_id"]] if (c := doc["choice"])
                                    else [-1, 0, 0] for doc in docs]),
        }
        for i, doc in enumerate(docs):
            m, cached = doc["n_rows"], slice(doc["offset"], doc["offset"] + doc["n_rows"])
            batch["input_ids"][i, :m + 1] = torch.tensor(doc["input_ids"])
            batch["attention_mask"][i, :m + 1] = 1
            batch["top_ids"][i, :m] = torch.from_numpy(self.arrays["topk_ids"][cached].astype(np.int64))
            batch["top_logq"][i, :m] = torch.from_numpy(self.arrays["topk_logq"][cached].astype(np.float32))
            batch["tail_logq"][i, :m] = torch.from_numpy(self.arrays["tail_logq"][cached].astype(np.float32))
            batch["region"][i, :m] = torch.from_numpy(self.arrays["region"][cached].astype(np.int64))
            batch["ce_mask"][i, :m] = doc["ce"]
            batch["kd_mask"][i, slice(*doc["kd_rows"])] = True
            batch["kd_normalization_mask"][i, slice(*doc["kd_normalization_rows"])] = True
        return batch


# Losses

def _chunk_losses(logits: torch.Tensor, targets: torch.Tensor, top_ids: torch.Tensor, top_logq: torch.Tensor,
                  tail_logq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    lse = logits.logsumexp(-1, keepdim=True)
    ce = (lse - logits.gather(-1, targets.unsqueeze(-1))).squeeze(-1)
    rest = logits.scatter(-1, top_ids, float("-inf")).logsumexp(-1, keepdim=True)
    log_s = torch.cat([logits.gather(-1, top_ids), rest], dim=-1) - lse
    log_q = torch.cat([top_logq, tail_logq.unsqueeze(-1)], dim=-1).float()
    log_q = log_q - log_q.logsumexp(-1, keepdim=True)  # stored in float16, q sums to 1 within about 1e-3
    return ce, (log_q.exp() * (log_q - log_s)).sum(-1)


def row_losses(logits: torch.Tensor, input_ids: torch.Tensor, top_ids: torch.Tensor, top_logq: torch.Tensor,
               tail_logq: torch.Tensor, chunk: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row j = 0 .. T-2, in float32: the next-token loss of token j + 1, and KL(q || s) over q's top entries and
    the rest of its mass. About `chunk` rows at a time, each chunk recomputed in the backward pass, so that the float32
    copies of the vocabulary-wide steps do not stay in memory."""
    n_rows = input_ids.shape[1] - 1
    step = max(1, chunk // len(input_ids))
    ce, kd = [], []
    for start in range(0, n_rows, step):
        rows = slice(start, min(start + step, n_rows))
        args = (logits[:, rows], input_ids[:, rows.start + 1:rows.stop + 1], top_ids[:, rows], top_logq[:, rows],
                tail_logq[:, rows])
        c, k = (checkpoint(_chunk_losses, *args, use_reentrant=False) if logits.requires_grad
                else _chunk_losses(*args))
        ce.append(c)
        kd.append(k)
    return torch.cat(ce, dim=1), torch.cat(kd, dim=1)


def choice_margins(logits: torch.Tensor, choice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The documents of a batch whose choice the cache measured, and the student's log s(accessible) -
    log s(exclusive) of the two options' first tokens at their choice rows."""
    row, accessible, exclusive = choice.unbind(-1)
    docs = (row >= 0).nonzero()[:, 0]
    picked = logits[docs, row[docs]].float()
    return docs, picked.gather(-1, accessible[docs, None])[:, 0] - picked.gather(-1, exclusive[docs, None])[:, 0]


# Trainer

class DistillTrainer(Trainer):
    """Trainer with the distillation loss on DistillCollator's batches. Logs, next to Trainer's loss: loss_ce and
    loss_kd, and the KD of each branch, kd/pos and kd/neg.

    evaluate() runs on every split of eval_dataset, a dict of example lists, and keeps each result in eval_results.
    Per split and branch, over every position of the documents: CE, and KD by region and in all; and where the cache
    measured q's choice margin, the share of documents on which the student favours the accessible option, and its
    median margin. The metrics are logged as eval_<split>/<measure>."""

    def __init__(self, *args, ce_weight: float = 1.0, kd_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss takes means over the positions get_batch_samples counts in the whole accumulation window, so
        # Trainer must not divide the loss by the accumulation steps again.
        self.model_accepts_loss_kwargs = True
        self.ce_weight, self.kd_weight = float(ce_weight), float(kd_weight)
        self.eval_results = []
        self._stats = None  # sums since the last log, see _record

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """The batches of an accumulation window, and its CE and full eligible KD positions (of all processes with
        average_tokens_across_devices), which compute_loss gets as num_items_in_batch."""
        batch_samples, _ = super().get_batch_samples(epoch_iterator, num_batches, device)  # no labels: no count
        if not batch_samples:
            return batch_samples, None
        counts = sum(torch.stack([batch["ce_mask"].sum(), batch["kd_normalization_mask"].sum()])
                     for batch in batch_samples)
        counts = counts.to(device)
        if self.args.average_tokens_across_devices:
            counts = self.accelerator.gather(counts).view(-1, 2).sum(0)
        return batch_samples, counts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, ce_mask, kd_mask = inputs["input_ids"], inputs["ce_mask"], inputs["kd_mask"]
        outputs = model(input_ids=input_ids, attention_mask=inputs["attention_mask"], use_cache=False)
        ce, kd = row_losses(outputs.logits, input_ids, inputs["top_ids"], inputs["top_logq"], inputs["tail_logq"])
        ce_sum, kd_sum = ce.masked_fill(~ce_mask, 0).sum(), kd.masked_fill(~kd_mask, 0).sum()
        if num_items_in_batch is None:  # a batch on its own, outside the training loop: the means over its positions
            (n_ce, n_kd), scale = (ce_mask.sum(), inputs["kd_normalization_mask"].sum()), 1
        else:
            # The positions of the accumulation window (of all processes with average_tokens_across_devices, whose
            # gradients DDP then averages, hence the factor)
            n_ce, n_kd = num_items_in_batch
            scale = self.accelerator.num_processes if self.args.average_tokens_across_devices else 1
        loss = scale * (self.ce_weight * ce_sum / n_ce.clamp_min(1) + self.kd_weight * kd_sum / n_kd.clamp_min(1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite distillation loss")
        if model.training:
            self._record(ce, kd, inputs)
        return (loss, outputs) if return_outputs else loss

    def _record(self, ce: torch.Tensor, kd: torch.Tensor, inputs: dict) -> None:
        """Add CE sum/count, active KD sum/count per branch, and the full KD denominator to the running stats."""
        ce_mask, kd_mask = inputs["ce_mask"], inputs["kd_mask"]
        pos = (inputs["branch"] == BRANCHES.index("pos"))[:, None]
        sums = [ce.masked_fill(~ce_mask, 0).sum(), ce_mask.sum()]
        for mask in (kd_mask & pos, kd_mask & ~pos):
            sums += [kd.masked_fill(~mask, 0).sum(), mask.sum()]
        sums.append(inputs["kd_normalization_mask"].sum())
        stats = torch.stack(sums).detach().double()
        self._stats = stats if self._stats is None else self._stats + stats

    def log(self, logs, *args, **kwargs):
        # Trainer logs on every process, so the reduction below runs everywhere at the same steps.
        if "loss" in logs and self._stats is not None:
            stats, self._stats = self._stats, None
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
            ce, n_ce, kd_pos, n_pos, kd_neg, n_neg, n_full = stats.tolist()
            logs = {**logs, "loss_ce": ce / max(n_ce, 1), "loss_kd": (kd_pos + kd_neg) / max(n_full, 1),
                    "kd/active_fraction": (n_pos + n_neg) / max(n_full, 1),
                    **({"kd/pos": kd_pos / n_pos} if n_pos else {}),
                    **({"kd/neg": kd_neg / n_neg} if n_neg else {})}
        super().log(logs, *args, **kwargs)

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        datasets = self.eval_dataset if eval_dataset is None else eval_dataset
        # The model without its DDP wrapper: each process takes every n-th pair, so processes may run different
        # numbers of batches. Once training has started, accelerate's autocast wraps this model's forward too.
        model, training = self.model, self.model.training
        model.eval()
        results = {split: self._evaluate_split(model, examples) for split, examples in datasets.items()}
        model.train(training)
        self.eval_results.append({"step": self.state.global_step, "splits": results})
        metrics = {f"{metric_key_prefix}_{split}/{name}": value for split, result in results.items()
                   for name, value in result.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def _evaluate_split(self, model, examples: list[dict]) -> dict[str, float]:
        sums = torch.zeros(len(BRANCHES), len(REGIONS), 3, dtype=torch.float64, device=self.args.device)
        margins = [[] for _ in BRANCHES]
        shard, size = examples[self.args.process_index::self.args.world_size], self.args.per_device_eval_batch_size
        for start in range(0, len(shard), size):
            batch = self._prepare_inputs(self.data_collator(shard[start:start + size]))
            with self.compute_loss_context_manager():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               use_cache=False).logits
            ce, kd = row_losses(logits, batch["input_ids"], batch["top_ids"], batch["top_logq"], batch["tail_logq"])
            for b in range(len(BRANCHES)):
                for r in range(len(REGIONS)):
                    rows = (batch["region"] == r) & (batch["branch"] == b)[:, None]
                    sums[b, r] += torch.stack([ce[rows].sum(), kd[rows].sum(), rows.sum()]).double()
            docs, margin = choice_margins(logits, batch["choice"])
            for b, m in zip(batch["branch"][docs].tolist(), margin.tolist()):
                margins[b].append(m)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, margins)
            margins = [[m for part in gathered for m in part[b]] for b in range(len(BRANCHES))]

        result = {}
        for b, branch in enumerate(BRANCHES):
            ce_sum, kd_sum, n = sums[b].sum(0).tolist()
            result[f"ce_{branch}"], result[f"kd_{branch}"] = ce_sum / n, kd_sum / n
            for r, region in enumerate(REGIONS):
                _, kd_sum, n = sums[b, r].tolist()
                result[f"kd_{branch}/{region}"] = kd_sum / n if n else float("nan")
            m = np.array(margins[b])
            result[f"measured_{branch}"] = float(len(m))
            result[f"choice_{branch}"] = float((m > 0).mean()) if len(m) else float("nan")
            result[f"margin_{branch}"] = float(np.median(m)) if len(m) else float("nan")
        return result
