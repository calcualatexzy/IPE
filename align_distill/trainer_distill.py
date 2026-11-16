"""The student's loss and evaluation (docs/align_distill.md#student), on teacher-cache pairs or on a plain corpus.

The student, Qwen with a LoRA, reads each document alone. Its loss is
    ce_weight * CE + kd_weight * KD
    CE  the next-token loss on the pos (pro-align) documents of a pair, or on every document of a corpus, the mean over
        their positions
    KD  KL(q || s) between the cached target q and the student s, summed on the distilled positions and divided by the
        eligible count of both documents (before the kd_documents mask), with q's top k entries and the rest of its
        mass as one more entry:
            sum_i q_i (log q_i - log s_i) + q_rest (log q_rest - log s_rest)
        where s_rest is the student's mass outside q's top k
Both means are over the positions of the whole optimizer step, of all processes, as Trainer takes the next-token loss.

Two kinds of examples:
- a pair of a teacher cache (align_distill/teacher_cache.py): its pos document, then its neg document. Up to the first
  token where the two differ they have the same context and the same q (the shared opening), so those positions are
  distilled once, on the pos document. opening=none leaves the opening out of KD (it keeps its CE);
  neg_after_decision=false distills the neg documents only up to the end of the decision sentence; kd_documents=pos or
  neg masks the other document's own positions, keeping the shared ones once on pos and the normalization of both.
- a document of a corpus: CE on every position and no KD. This is MSM, run by the same trainer.

Evaluation runs on every set of eval_dataset (a dict of example lists) at the start, every eval_steps and at the end.
Per set and branch: CE over every position; for pairs also KD by region and in all, and where the cache read q's choice
margin, the share of documents on which the student favours the aligned action and its median margin. Every process
runs the same number of forward passes, padding its share with weightless copies, because FSDP gathers each layer's
weights on all processes at every forward.

Copied from msm/trainer_distill.py (which transformers 5 cannot import) and changed for the four regions, the aligned
and misaligned options, corpus examples and FSDP.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from align_distill.pairs import BRANCHES, REGIONS
from align_distill.train_common import AdapterSaving

DOC_BRANCHES = (*BRANCHES, "corpus")  # a document's branch index
OPENINGS = ("once", "none")
KD_DOCUMENTS = ("both", "pos", "neg")
CHUNK = 1024  # positions scored at a time, each chunk in float32


# Examples

def shared_rows(pos_ids: list[int], neg_ids: list[int]) -> int:
    """How many rows the two documents of a pair share. Row j predicts token j + 1 from tokens 0 .. j, so every row
    before the first token that differs has the same context in both, and the same q."""
    for j, (a, b) in enumerate(zip(pos_ids, neg_ids)):
        if a != b:
            return j
    return min(len(pos_ids), len(neg_ids))


def pair_examples(documents: list[dict], splits: list[str], opening: str = "once", neg_after_decision: bool = True,
                  kd_documents: str = "both") -> dict[str, list[dict]]:
    """The pairs of each split, in cache order, as examples of two documents (pos, then neg). Each document keeps its
    tokens, its rows in the cache's arrays, whether it gets CE, the rows it is distilled on ([start, end)), its full
    eligible rows (kd_normalization_rows), and its choice row with the cache's p0 and q margins there, or None."""
    if opening not in OPENINGS:
        raise ValueError(f"opening must be one of {OPENINGS}, not {opening!r}")
    if kd_documents not in KD_DOCUMENTS:
        raise ValueError(f"kd_documents must be one of {KD_DOCUMENTS}, not {kd_documents!r}")
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
            opening_end, _, decision_end = doc["region_bounds"]
            start, end = (0 if opening == "once" else opening_end), doc["n_rows"]
            if branch == "neg":
                start = max(start, shared)  # the pos document distills the shared rows
                if not neg_after_decision:
                    end = decision_end
            start = min(start, end)
            normalization_rows = [start, end]
            if kd_documents == "pos" and branch == "neg":
                end = start
            elif kd_documents == "neg" and branch == "pos":
                end = max(start, min(shared, end))  # keep the shared positions once, on pos
            choice = doc["choice"]
            entries.append({
                "doc_id": doc["doc_id"], "input_ids": doc["input_ids"], "offset": doc["offset"],
                "n_rows": doc["n_rows"], "branch": DOC_BRANCHES.index(branch), "ce": branch == "pos",
                "kd_rows": [start, end], "kd_normalization_rows": normalization_rows,
                "choice": {"row": choice["row"], "aligned_id": choice["aligned_id"],
                           "misaligned_id": choice["misaligned_id"], **choice["margin"]} if "row" in choice else None,
            })
        examples[pair["pos"]["split"]].append({"id": pair_id, "shared_rows": shared, "documents": entries})
    return examples


def corpus_examples(documents: list[list[int]], name: str) -> list[dict]:
    """The documents of a corpus as examples of one document each, with CE on every position and no KD."""
    return [{"id": f"{name}-{i}", "documents": [{
        "doc_id": f"{name}-{i}", "input_ids": ids, "offset": None, "n_rows": len(ids) - 1,
        "branch": DOC_BRANCHES.index("corpus"), "ce": True, "kd_rows": [0, 0], "kd_normalization_rows": [0, 0],
        "choice": None}]} for i, ids in enumerate(documents)]


class DistillCollator:
    """The documents of a batch of examples, padded on the right, and per row (row j predicts token j + 1): whether it
    lies in its document, the CE, distilled and normalization masks, the region (-1 on padding and in corpus documents)
    and, from the cache, q's top entries and the log of its other mass. Per document: its branch (an index into
    DOC_BRANCHES), its weight (0 for evaluation padding) and its choice row with the two options' first tokens (row -1
    where none was read). arrays are the cache's, memory-mapped; None for a corpus, whose batches carry no q."""

    def __init__(self, arrays: dict[str, np.ndarray] | None, pad_token_id: int):
        self.arrays, self.pad_token_id = arrays, pad_token_id
        self.top_k = arrays["topk_ids"].shape[1] if arrays is not None else 0

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        docs = [(doc, 0.0 if example.get("padding") else 1.0) for example in examples for doc in example["documents"]]
        n, length = len(docs), max(len(doc["input_ids"]) for doc, _ in docs)
        rows = (n, length - 1)
        batch = {
            "input_ids": torch.full((n, length), self.pad_token_id),
            "attention_mask": torch.zeros(n, length, dtype=torch.long),
            "valid": torch.zeros(rows, dtype=torch.bool),
            "ce_mask": torch.zeros(rows, dtype=torch.bool),
            "kd_mask": torch.zeros(rows, dtype=torch.bool),
            "kd_normalization_mask": torch.zeros(rows, dtype=torch.bool),
            "region": torch.full(rows, -1),
            "branch": torch.tensor([doc["branch"] for doc, _ in docs]),
            "weight": torch.tensor([w for _, w in docs]),
            "choice": torch.tensor([[c["row"], c["aligned_id"], c["misaligned_id"]] if (c := doc["choice"])
                                    else [-1, 0, 0] for doc, _ in docs]),
        }
        if self.arrays is not None:
            # padding rows get a uniform q, so that their (masked) losses stay finite
            batch.update({"top_ids": torch.zeros(*rows, self.top_k, dtype=torch.long),
                          "top_logq": torch.zeros(*rows, self.top_k), "tail_logq": torch.zeros(rows)})
        for i, (doc, _) in enumerate(docs):
            m = doc["n_rows"]
            batch["input_ids"][i, :m + 1] = torch.tensor(doc["input_ids"])
            batch["attention_mask"][i, :m + 1] = 1
            batch["valid"][i, :m] = True
            batch["ce_mask"][i, :m] = doc["ce"]
            batch["kd_mask"][i, slice(*doc["kd_rows"])] = True
            batch["kd_normalization_mask"][i, slice(*doc["kd_normalization_rows"])] = True
            if doc["offset"] is not None:
                cached = slice(doc["offset"], doc["offset"] + m)
                batch["top_ids"][i, :m] = torch.from_numpy(self.arrays["topk_ids"][cached].astype(np.int64))
                batch["top_logq"][i, :m] = torch.from_numpy(self.arrays["topk_logq"][cached].astype(np.float32))
                batch["tail_logq"][i, :m] = torch.from_numpy(self.arrays["tail_logq"][cached].astype(np.float32))
                batch["region"][i, :m] = torch.from_numpy(self.arrays["region"][cached].astype(np.int64))
        return batch


# Losses (msm/trainer_distill.py)

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


def _chunk_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return (logits.logsumexp(-1) - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1))


def _chunked(function, logits: torch.Tensor, input_ids: torch.Tensor, *row_args: torch.Tensor, chunk: int = CHUNK):
    """function over about `chunk` rows at a time, each chunk recomputed in the backward pass, so that the float32
    copies of the vocabulary-wide steps do not stay in memory."""
    n_rows = input_ids.shape[1] - 1
    step = max(1, chunk // len(input_ids))
    outputs = []
    for start in range(0, n_rows, step):
        rows = slice(start, min(start + step, n_rows))
        args = (logits[:, rows], input_ids[:, rows.start + 1:rows.stop + 1], *(x[:, rows] for x in row_args))
        out = checkpoint(function, *args, use_reentrant=False) if logits.requires_grad else function(*args)
        outputs.append(out if isinstance(out, tuple) else (out,))
    return tuple(torch.cat(parts, dim=1) for parts in zip(*outputs))


def row_losses(logits: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row j = 0 .. T-2, in float32: the next-token loss of token j + 1, and KL(q || s) (zeros without q)."""
    input_ids = inputs["input_ids"]
    if "top_ids" not in inputs:
        (ce,) = _chunked(_chunk_ce, logits, input_ids)
        return ce, torch.zeros_like(ce)
    return _chunked(_chunk_losses, logits, input_ids, inputs["top_ids"], inputs["top_logq"], inputs["tail_logq"])


def choice_margins(logits: torch.Tensor, choice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The documents of a batch whose choice row is known, and the student's log s(aligned) - log s(misaligned) of the
    two options' first tokens there."""
    row, aligned, misaligned = choice.unbind(-1)
    docs = (row >= 0).nonzero()[:, 0]
    picked = logits[docs, row[docs]].float()
    return docs, picked.gather(-1, aligned[docs, None])[:, 0] - picked.gather(-1, misaligned[docs, None])[:, 0]


def eval_batches(examples: list[dict], rank: int, world: int, size: int) -> list[list[dict]]:
    """A process's evaluation batches: every world-th example from its rank, padded with weightless copies so that every
    process runs the same number of batches of the same size."""
    mine = examples[rank::world]
    n_batches = -(-(-(-len(examples) // world)) // size)  # ceil(ceil(n / world) / size)
    mine = mine + [{**examples[0], "padding": True}] * (n_batches * size - len(mine))
    return [mine[i:i + size] for i in range(0, len(mine), size)]


# Trainer

class DistillTrainer(AdapterSaving):
    """Trainer with the distillation loss on DistillCollator's batches. Logs, next to Trainer's loss: loss_ce and
    loss_kd, and the KD of each branch, kd/pos and kd/neg. evaluate() keeps each result in eval_results."""

    def __init__(self, *args, ce_weight: float = 1.0, kd_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss takes means over the positions get_batch_samples counts in the whole accumulation window, so
        # Trainer must not divide the loss by the accumulation steps again.
        self.model_accepts_loss_kwargs = True
        self.loss_is_scaled_for_ga = True
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
                     for batch in batch_samples).to(device)
        if self.args.average_tokens_across_devices and self.args.world_size > 1:
            counts = self.accelerator.gather(counts).view(-1, 2).sum(0)
        return batch_samples, counts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ce_mask, kd_mask = inputs["ce_mask"], inputs["kd_mask"]
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        ce, kd = row_losses(outputs.logits, inputs)
        ce_sum, kd_sum = ce.masked_fill(~ce_mask, 0).sum(), kd.masked_fill(~kd_mask, 0).sum()
        if num_items_in_batch is None:  # a batch on its own, outside the training loop: the means over its positions
            (n_ce, n_kd), scale = (ce_mask.sum(), inputs["kd_normalization_mask"].sum()), 1
        else:
            # The positions of the accumulation window (of all processes with average_tokens_across_devices, whose
            # gradients DDP and FSDP then average, hence the factor)
            n_ce, n_kd = num_items_in_batch
            scale = self.accelerator.num_processes if self.args.average_tokens_across_devices else 1
        loss = scale * (self.ce_weight * ce_sum / n_ce.clamp_min(1) + self.kd_weight * kd_sum / n_kd.clamp_min(1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite distillation loss")
        if model.training:
            self._record(ce, kd, inputs)
        return (loss, outputs) if return_outputs else loss

    def _record(self, ce: torch.Tensor, kd: torch.Tensor, inputs: dict) -> None:
        """Add the CE sum and count, the distilled KD sum and count per branch, and the full KD denominator."""
        ce_mask, kd_mask = inputs["ce_mask"], inputs["kd_mask"]
        pos = (inputs["branch"] == DOC_BRANCHES.index("pos"))[:, None]
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
            logs = {**logs, "loss_ce": ce / max(n_ce, 1)}
            if n_full:
                logs.update({"loss_kd": (kd_pos + kd_neg) / n_full, "kd/active_fraction": (n_pos + n_neg) / n_full,
                             **({"kd/pos": kd_pos / n_pos} if n_pos else {}),
                             **({"kd/neg": kd_neg / n_neg} if n_neg else {})})
        super().log(logs, *args, **kwargs)

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        datasets = self.eval_dataset if eval_dataset is None else eval_dataset
        model, training = self.model, self.model.training
        model.eval()
        results = {name: self._evaluate_set(model, examples) for name, examples in datasets.items()}
        model.train(training)
        self.eval_results.append({"step": self.state.global_step, "sets": results})
        metrics = {f"{metric_key_prefix}_{name}/{measure}": value for name, result in results.items()
                   for measure, value in result.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def _evaluate_set(self, model, examples: list[dict]) -> dict[str, float]:
        n_branches, n_regions = len(DOC_BRANCHES), len(REGIONS)
        # per branch: CE sum, KD sum, rows over every row; per branch and region: KD sum, rows
        totals = torch.zeros(n_branches, 3, dtype=torch.float64, device=self.args.device)
        regions = torch.zeros(n_branches, n_regions, 2, dtype=torch.float64, device=self.args.device)
        margins = [[] for _ in DOC_BRANCHES]
        for chunk in eval_batches(examples, self.args.process_index, self.args.world_size,
                                  self.args.per_device_eval_batch_size):
            batch = self._prepare_inputs(self.data_collator(chunk))
            with self.compute_loss_context_manager():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               use_cache=False).logits
            ce, kd = row_losses(logits, batch)
            real = batch["weight"] > 0
            for b in range(n_branches):
                docs = (real & (batch["branch"] == b))[:, None]
                rows = docs & batch["valid"]
                totals[b] += torch.stack([ce[rows].sum(), kd[rows].sum(), rows.sum()]).double()
                for r in range(n_regions):
                    in_region = docs & (batch["region"] == r)
                    regions[b, r] += torch.stack([kd[in_region].sum(), in_region.sum()]).double()
            docs, margin = choice_margins(logits, batch["choice"])
            for d, m in zip(docs.tolist(), margin.tolist()):
                if real[d]:
                    margins[int(batch["branch"][d])].append(m)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
            dist.all_reduce(regions)
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, margins)
            margins = [[m for part in gathered for m in part[b]] for b in range(n_branches)]

        result = {}
        for b, branch in enumerate(DOC_BRANCHES):
            ce_sum, kd_sum, n = totals[b].tolist()
            if not n:
                continue
            result[f"ce_{branch}"] = ce_sum / n
            if branch == "corpus":
                continue
            result[f"kd_{branch}"] = kd_sum / n
            for r, region in enumerate(REGIONS):
                kd_sum, n = regions[b, r].tolist()
                result[f"kd_{branch}/{region}"] = kd_sum / n if n else float("nan")
            m = np.array(margins[b])
            result[f"measured_{branch}"] = float(len(m))
            result[f"choice_{branch}"] = float((m > 0).mean()) if len(m) else float("nan")
            result[f"margin_{branch}"] = float(np.median(m)) if len(m) else float("nan")
        return result
