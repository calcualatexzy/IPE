"""SPO for Model Spec Midtraining: next-token loss on the spec documents, plus a preference loss that ranks each
document above a negative document, from a spec that keeps the cheese preferences but gives them another reason or
none (msm/spec/neg/), or from the other arm's spec.

A document's score s is its mean token log-probability over every token after <|begin_of_text|>, <|end_of_text|>
included, so that long and short documents compare. For each pair, with margin = beta * (s+ - s-):
    simpo        -log sigmoid(margin - gamma)
    huber_hinge  zero once margin >= gamma; below that, a Huber penalty on the shortfall gamma - margin: quadratic
                 within huber_delta of the target, then linear with slope one
The loss of a batch is the next-token loss of its positive documents, normalized as Trainer normalizes it, plus
weight times the mean pair loss. Negative documents get no next-token loss. With weight 0 the collator leaves the
negatives out, and the loss is exactly Trainer's.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Trainer

from msm.train_data import IGNORE_INDEX

PAIR_LOSSES = ("huber_hinge", "simpo")
CHUNK = 1024  # positions scored at a time, each chunk in float32
N_STATS = 8  # entries of SPOTrainer's running stats before the per-corpus ones


def check_settings(weight: float, pair_loss_type: str, beta: float, gamma: float,
                   huber_delta: float | None = None) -> float:
    """Validate the SPO settings and return the Huber width, which defaults to 0.25 * gamma."""
    if pair_loss_type not in PAIR_LOSSES:
        raise ValueError(f"SPO pair_loss_type must be one of {PAIR_LOSSES}, not {pair_loss_type!r}")
    huber_delta = 0.25 * gamma if huber_delta is None else huber_delta
    checks = {"weight": (weight, weight >= 0), "beta": (beta, beta > 0), "gamma": (gamma, gamma >= 0),
              "huber_delta": (huber_delta, huber_delta > 0 or pair_loss_type == "simpo")}
    for name, (value, valid) in checks.items():
        if not (math.isfinite(value) and valid):
            raise ValueError(f"Invalid SPO {name}: {value}" + (" (null means 0.25 * gamma)" if name == "huber_delta"
                                                              else ""))
    return float(huber_delta)


def pair_losses(positive: torch.Tensor, negative: torch.Tensor, pair_loss_type: str, beta: float, gamma: float,
                huber_delta: float) -> torch.Tensor:
    """The loss of each pair, from the scores of its positive and its negative document."""
    margin = beta * (positive - negative)
    if pair_loss_type == "simpo":
        return -F.logsigmoid(margin - gamma)
    shortfall = (gamma - margin).clamp_min(0)
    quadratic = shortfall.clamp(max=huber_delta)
    return quadratic * (shortfall - quadratic / 2) / huber_delta


def _chunk_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """log p(token t | tokens before t) for t = 1 .. T-1, in float32. Each chunk of positions is recomputed in the
    backward pass, so that no float32 copy of the logits stays in memory."""
    pieces = []
    for start in range(0, input_ids.shape[1] - 1, chunk):
        targets = input_ids[:, start + 1:start + 1 + chunk]
        piece = logits[:, start:start + targets.shape[1]]
        pieces.append(checkpoint(_chunk_logprobs, piece, targets, use_reentrant=False) if piece.requires_grad
                      else _chunk_logprobs(piece, targets))
    return torch.cat(pieces, dim=1)


class SPOCollator:
    """The positive documents in rows 0 .. B-1 and their negatives in rows B .. 2B-1, padded on the right, with labels
    on the positives only, so that Trainer counts only their tokens. Without negatives (weight 0) the batch is the one
    DataCollatorForSeq2Seq makes for plain MSM."""

    def __init__(self, pad_token_id: int, include_negatives: bool = True):
        self.pad_token_id, self.include_negatives = pad_token_id, include_negatives

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        rows = [example["input_ids"] for example in examples]
        if self.include_negatives:
            rows += [example["negative_input_ids"] for example in examples]
        input_ids = torch.full((len(rows), max(map(len, rows))), self.pad_token_id)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i, ids in enumerate(rows):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention_mask[i, :len(ids)] = 1
            if i < len(examples):
                labels[i, :len(ids)] = input_ids[i, :len(ids)]
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if self.include_negatives:
            batch["negative_source"] = torch.tensor([example["negative_source"] for example in examples])
        return batch


class SPOTrainer(Trainer):
    """Trainer with the SPO loss on SPOCollator's batches. Logs, next to Trainer's loss: loss_ce (the positives'
    next-token loss per token), loss_spo (the mean pair loss), the mean scores spo/score_pos and spo/score_neg and their
    gap spo/gap, spo/accuracy (the share of pairs with s+ > s-), spo/margin_satisfied (the share with margin > gamma)
    and spo/accuracy/<corpus> for each negative corpus."""

    def __init__(self, *args, negative_names: list[str], weight: float = 1.0, pair_loss_type: str = "huber_hinge",
                 beta: float = 2.0, gamma: float = 0.5, huber_delta: float | None = None, **kwargs):
        self.huber_delta = check_settings(weight, pair_loss_type, beta, gamma, huber_delta)
        super().__init__(*args, **kwargs)
        if self.args.eval_strategy != "no":
            raise ValueError("SPO trains without evaluation; evaluate the adapters with msm/eval.py")
        # compute_loss normalizes the next-token loss as Trainer does for models that take num_items_in_batch: by the
        # positive tokens of the whole accumulation window, so Trainer must count them and not divide the loss again.
        self.model_accepts_loss_kwargs = True
        self.negative_names = list(negative_names)
        self.weight, self.pair_loss_type = float(weight), pair_loss_type
        self.beta, self.gamma = float(beta), float(gamma)
        self._stats = None  # sums since the last log, see _record

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        sources = inputs.get("negative_source")
        n_pairs = 0 if sources is None else len(sources)
        n_positive = len(input_ids) - n_pairs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        scored = attention_mask[:, 1:].bool()  # every real token after the first
        logprobs = token_logprobs(outputs.logits, input_ids).masked_fill(~scored, 0)
        totals, counts = logprobs.sum(dim=1), scored.sum(dim=1)
        nll = -totals[:n_positive].sum()
        if num_items_in_batch is None:  # a batch on its own, outside the training loop: the mean over its tokens
            loss, accumulation = nll / counts[:n_positive].sum(), 1
        else:
            # As Trainer: num_items_in_batch counts the positive labels of the accumulation window (of all processes
            # with average_tokens_across_devices, whose gradients DDP then averages, hence the factor).
            loss, accumulation = nll / num_items_in_batch, self.args.gradient_accumulation_steps
            if self.args.average_tokens_across_devices:
                loss = loss * self.accelerator.num_processes
        positive = negative = pairs = None
        if n_pairs:
            scores = totals / counts
            positive, negative = scores[:n_positive], scores[n_positive:]
            pairs = pair_losses(positive, negative, self.pair_loss_type, self.beta, self.gamma, self.huber_delta)
            loss = loss + self.weight * pairs.mean() / accumulation
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SPO loss")
        if model.training:
            self._record(nll, counts[:n_positive].sum(), positive, negative, pairs, sources)
        return (loss, outputs) if return_outputs else loss

    def _record(self, nll, n_tokens, positive, negative, pairs, sources):
        """Add a batch to the running sums: next-token loss and tokens, pair loss and pairs, positive and negative
        scores, wins, satisfied margins, then pairs and wins per negative corpus."""
        k = len(self.negative_names)
        stats = torch.zeros(N_STATS + 2 * k, dtype=torch.float64, device=nll.device)
        stats[0], stats[1] = nll.detach(), n_tokens
        if pairs is not None:
            positive, negative = positive.detach().double(), negative.detach().double()
            wins = (positive > negative).double()
            stats[2], stats[3] = pairs.detach().sum(), len(pairs)
            stats[4], stats[5] = positive.sum(), negative.sum()
            stats[6], stats[7] = wins.sum(), (self.beta * (positive - negative) > self.gamma).sum()
            stats[N_STATS:N_STATS + k] = torch.bincount(sources, minlength=k)
            stats[N_STATS + k:] = torch.bincount(sources, weights=wins, minlength=k)
        self._stats = stats if self._stats is None else self._stats + stats

    def log(self, logs, *args, **kwargs):
        # Trainer logs on every process, so the reduction below runs everywhere at the same steps.
        if "loss" in logs and self._stats is not None:
            stats, self._stats = self._stats, None
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
            nll, tokens, pair_loss, pairs, positive, negative, wins, margins = stats[:N_STATS].tolist()
            logs = {**logs, "loss_ce": nll / tokens}
            if pairs:
                logs.update({"loss_spo": pair_loss / pairs, "spo/score_pos": positive / pairs,
                             "spo/score_neg": negative / pairs, "spo/gap": (positive - negative) / pairs,
                             "spo/accuracy": wins / pairs, "spo/margin_satisfied": margins / pairs})
                k = len(self.negative_names)
                per_corpus = zip(self.negative_names, stats[N_STATS:N_STATS + k].tolist(),
                                 stats[N_STATS + k:].tolist())
                logs.update({f"spo/accuracy/{name}": won / drawn for name, drawn, won in per_corpus if drawn})
        super().log(logs, *args, **kwargs)
