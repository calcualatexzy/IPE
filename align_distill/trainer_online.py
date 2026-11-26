"""The windowed student (docs/align_distill_windowed.md): DistillTrainer with q computed during training instead of read
from a teacher cache, anchored on a snapshot of the student that moves every window_steps optimizer steps.

Per training document, under no_grad, before the training pass of DistillTrainer.compute_loss:
    log q = log_softmax(log s_k + alpha_k * clip(log p+ - log p-))
    s_k      the snapshot of the LoRA taken at step k * window_steps (window 0: the fresh LoRA, the base model), copied
             into the LoRA's own weights for one plain pass and copied back
    p+, p-   the base model with the LoRA off after pro-align and neg-align (teacher_cache.py's framed specs), from each
             spec's KV cache, filled once per process on the first training call
    alpha_k  alpha * (steps in window k) / (all steps), so that the windows' alphas add up to alpha
q is cut to its top_k entries and the rest of its mass, as a cache stores it, so that DistillTrainer's loss runs
unchanged: CE on the pos documents, KD on both. With alpha_k = 0 the teacher passes are skipped and q is the snapshot.

One Trainer, one AdamW and one schedule: at a window boundary only the snapshot changes. The snapshot is taken on the
first micro-batch of the window's first step, so every micro-batch of a step sees the same one. Evaluation runs no
teacher passes.

FSDP2: every process makes the same passes in the same order per micro-batch (each holds one pair: two documents), as
each pass gathers every layer's weights from all processes. The snapshot holds each process's own shards of the LoRA.
The LoRA is switched off by its layers' flag alone: PEFT's disable_adapter() also sets requires_grad, which FSDP2 reads
once, at a parameter's first all-gather.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from loguru import logger

from align_distill.teacher_cache import (cached_logits, document_logits, fill_cache, spec_cache_diff,
                                         spec_cache_warning, target_logq)
from align_distill.train_common import unwrap
from align_distill.trainer_distill import DistillTrainer


def window_alphas(alpha: float, total_steps: int, window_steps: int) -> list[float]:
    """alpha_k of each window of total_steps optimizer steps: alpha in proportion to the window's steps (the last window
    may be shorter)."""
    starts = range(0, total_steps, window_steps)
    return [alpha * (min(start + window_steps, total_steps) - start) / total_steps for start in starts]


def target_rows(logits_s: torch.Tensor, logits_pos: torch.Tensor | None, logits_neg: torch.Tensor | None,
                alpha: float, clip: float | None, top_k: int, chunk_rows: int):
    """A document's q in the cache's format, from the logits of the snapshot and the two teachers: per row, q's top_k
    entries, their log q, and the log of the rest of q's mass. Without teachers (alpha 0), q is the snapshot."""
    ids, values, tails = [], [], []
    for start in range(0, len(logits_s), chunk_rows):
        rows = slice(start, start + chunk_rows)
        log_s = torch.log_softmax(logits_s[rows].float(), dim=-1)
        if logits_pos is None:
            log_q = log_s
        else:
            log_q = target_logq(log_s, torch.log_softmax(logits_pos[rows].float(), dim=-1),
                                torch.log_softmax(logits_neg[rows].float(), dim=-1), alpha, clip)
        top = log_q.topk(top_k, dim=-1)
        ids.append(top.indices)
        values.append(top.values)
        tails.append(log_q.scatter(-1, top.indices, float("-inf")).logsumexp(-1))
    return torch.cat(ids), torch.cat(values), torch.cat(tails)


def lora_layers(model) -> list:
    from peft.tuners.tuners_utils import BaseTunerLayer

    return [module for module in unwrap(model).modules() if isinstance(module, BaseTunerLayer)]


@contextmanager
def adapters_off(model):
    """The model's LoRA switched off by its layers' flag, leaving requires_grad as it is."""
    layers = lora_layers(model)
    states = [layer._disable_adapters for layer in layers]
    for layer in layers:
        layer._disable_adapters = True
    try:
        yield
    finally:
        for layer, state in zip(layers, states):
            layer._disable_adapters = state


def local_tensor(param: torch.Tensor) -> torch.Tensor:
    """A parameter's own storage: under FSDP2 this process's shard (a view of the DTensor's local tensor, call under
    no_grad), else the parameter."""
    from torch.distributed.tensor import DTensor

    return param.to_local() if isinstance(param, DTensor) else param


class LoraSnapshot:
    """A copy of the trainable (LoRA) weights, this process's shards under FSDP2, which swapped_in() copies into the
    model in place for the duration of a pass and then restores."""

    def __init__(self, model):
        self.params = [param for param in unwrap(model).parameters() if param.requires_grad]
        if not self.params:
            raise ValueError("The model has no trainable weights to snapshot")
        self.saved = None

    @torch.no_grad()
    def capture(self) -> None:
        self.saved = [local_tensor(param).detach().clone() for param in self.params]

    @contextmanager
    def swapped_in(self):
        if self.saved is None:
            raise RuntimeError("No snapshot was captured")
        with torch.no_grad():
            current = [local_tensor(param).detach().clone() for param in self.params]
            for param, saved in zip(self.params, self.saved):
                local_tensor(param).copy_(saved)
        try:
            yield
        finally:
            with torch.no_grad():
                for param, value in zip(self.params, current):
                    local_tensor(param).copy_(value)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.saved or [])


class Passes:
    """The model as teacher_cache.py's passes call it: on the training device, under the given context."""

    def __init__(self, model, device, context=None):
        self.model, self.device, self.context = model, device, context

    def __call__(self, *args, **kwargs):
        if self.context is None:
            return self.model(*args, **kwargs)
        with self.context():
            return self.model(*args, **kwargs)


class OnlineDistillTrainer(DistillTrainer):
    """DistillTrainer whose training batches get q from the passes above. contexts are the token contexts of the two
    framed specs (teacher_cache.build_contexts: S+ and S-); alpha is the total over all windows. Logs, next to
    DistillTrainer's: online/window and online/alpha, those of the last step logged."""

    def __init__(self, *args, contexts: dict[str, list[int]], alpha: float, clip: float | None, top_k: int,
                 window_steps: int, chunk_rows: int, **kwargs):
        super().__init__(*args, **kwargs)
        if set(contexts) != {"S+", "S-"}:
            raise ValueError(f"contexts must hold S+ and S-, not {sorted(contexts)}")
        if window_steps < 1 or alpha < 0:
            raise ValueError("window_steps must be at least 1 and alpha at least 0")
        self.contexts, self.alpha, self.clip, self.top_k = contexts, float(alpha), clip, int(top_k)
        self.window_steps, self.chunk_rows = int(window_steps), int(chunk_rows)
        self.snapshot = None
        self.window = None  # the window of the current snapshot
        self.spec_caches = None
        self.spec_check = None  # the prefilled spec against one full pass, on the first document

    def window_alpha(self, window: int) -> float:
        alphas = window_alphas(self.alpha, self.state.max_steps, self.window_steps)
        return alphas[window] if window < len(alphas) else 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training and "top_ids" not in inputs:
            inputs = {**inputs, **self.online_targets(model, inputs)}
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)

    def refresh(self, model, window: int) -> None:
        """The snapshot of the LoRA as it is now, at the first step of the window."""
        if self.snapshot is None:
            self.snapshot = LoraSnapshot(model)
        self.snapshot.capture()
        self.window = window
        start = window * self.window_steps
        end = min(start + self.window_steps, self.state.max_steps)
        logger.info(f"Window {window}: steps {start}-{end - 1} of {self.state.max_steps}, alpha "
                    f"{self.window_alpha(window):.4f}, anchored on the student after {self.state.global_step} steps "
                    f"(snapshot {self.snapshot.nbytes() / 2 ** 30:.2f} GB on this process)")

    @torch.no_grad()
    def online_targets(self, model, inputs) -> dict[str, torch.Tensor]:
        """q's top entries and the log of its other mass at every row of the batch's documents (zeros on padding, as
        DistillCollator pads a cache's q)."""
        window = self.state.global_step // self.window_steps
        if window != self.window:
            self.refresh(model, window)
        alpha = self.window_alpha(window)
        peft = unwrap(model)
        device = inputs["input_ids"].device
        anchor = Passes(peft, device, self.snapshot.swapped_in)
        teacher = Passes(peft, device, lambda: adapters_off(peft))
        n, rows = inputs["valid"].shape
        top_ids = torch.zeros(n, rows, self.top_k, dtype=torch.long, device=device)
        top_logq = torch.zeros(n, rows, self.top_k, device=device)
        tail_logq = torch.zeros(n, rows, device=device)
        training = model.training
        model.eval()  # in training mode, Hugging Face's gradient checkpointing would drop the KV caches
        try:
            if alpha and self.spec_caches is None:
                self.fill_spec_caches(teacher, inputs["input_ids"][0, :int(inputs["valid"][0].sum()) + 1].tolist())
            for i in range(n):
                m = int(inputs["valid"][i].sum())
                ids = inputs["input_ids"][i, :m + 1].tolist()
                logits_s = document_logits(anchor, ids)
                logits_pos = logits_neg = None
                if alpha:
                    logits_pos = cached_logits(teacher, ids, self.contexts["S+"], self.spec_caches["S+"])
                    logits_neg = cached_logits(teacher, ids, self.contexts["S-"], self.spec_caches["S-"])
                rows_i = target_rows(logits_s, logits_pos, logits_neg, alpha, self.clip, self.top_k, self.chunk_rows)
                del logits_s, logits_pos, logits_neg
                top_ids[i, :m], top_logq[i, :m], tail_logq[i, :m] = rows_i
        finally:
            model.train(training)
        return {"top_ids": top_ids, "top_logq": top_logq, "tail_logq": tail_logq}

    def fill_spec_caches(self, teacher: Passes, ids: list[int]) -> None:
        """Each spec's KV cache under the base model, and its check against one full pass on a document."""
        self.spec_caches = {name: fill_cache(teacher, context) for name, context in self.contexts.items()}
        self.spec_check = spec_cache_diff(teacher, ids, self.contexts["S+"], self.spec_caches["S+"])
        lengths = ", ".join(f"{name} {len(context):,} tokens" for name, context in self.contexts.items())
        (logger.warning if spec_cache_warning(self.spec_check) else logger.info)(
            f"Spec KV caches filled with the LoRA off ({lengths}); prefilled spec vs one full pass, |log p| difference "
            f"at a document's tokens: max {self.spec_check['max']:.4f}, mean {self.spec_check['mean']:.4f}")

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self.window is not None:
            logs = {**logs, "online/window": self.window, "online/alpha": self.window_alpha(self.window)}
        super().log(logs, *args, **kwargs)
