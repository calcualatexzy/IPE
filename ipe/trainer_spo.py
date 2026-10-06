"""Story CE plus reflection preference loss (SimPO or Huberized hinge) and optional positive reflection CE."""
from __future__ import annotations

import math
import inspect
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Trainer

from ipe import interleaved
from ipe.hidden_state_tracking import HiddenStateTrackingMixin
from ipe.separator_tracking import SeparatorTrackingMixin
from ipe.spo_data import stable_seed

PAIR_LOSS_TYPES = ("simpo", "huber_hinge")


def simpo_pair_loss(positive, negative, beta, gamma):
    return -F.logsigmoid(beta * (positive.float()-negative.float())-gamma)


def huber_hinge_pair_loss(positive, negative, beta, gamma, delta):
    """Zero past the margin, quadratic within delta of it, then linear with slope one."""
    shortfall = F.relu(gamma - beta*(positive.float()-negative.float()))
    clipped = shortfall.clamp(max=delta)
    return clipped*(shortfall-clipped/2)/delta


def resolve_huber_delta(pair_loss_type, gamma, huber_delta=None):
    """Validate the pair loss type and return the Huber width, which defaults to 0.25 * gamma."""
    if pair_loss_type not in PAIR_LOSS_TYPES:
        raise ValueError(f"Invalid SPO pair_loss_type: {pair_loss_type!r}; expected one of {PAIR_LOSS_TYPES}")
    delta = 0.25*float(gamma) if huber_delta is None else float(huber_delta)
    if pair_loss_type == "huber_hinge" and not (math.isfinite(delta) and delta > 0):
        raise ValueError(f"Invalid SPO huber_delta: {delta} (defaults to 0.25 * gamma; use gamma > 0 or set huber_delta)")
    return delta


def _selected_nll(logits, ids, predictors, mask, chunk_size=128):
    """Score selected targets in FP32 without a batch-sized log-softmax copy."""
    rows, positions = mask.nonzero(as_tuple=True)
    losses = []
    for start in range(0, len(rows), chunk_size):
        r, p = rows[start:start+chunk_size], positions[start:start+chunk_size]
        def score_chunk(values, row_indices, prediction_indices, targets):
            return F.cross_entropy(values[row_indices, prediction_indices].float(), targets, reduction="none")
        # Recompute the small FP32 softmax chunks during backward instead of
        # retaining one vocabulary-sized activation for every scored target.
        if logits.requires_grad:
            losses.append(checkpoint(score_chunk, logits, r, predictors[r, p], ids[r, p+1], use_reentrant=False))
        else:
            losses.append(score_chunk(logits, r, predictors[r, p], ids[r, p+1]))
    return rows, torch.cat(losses) if losses else logits.new_empty(0, dtype=torch.float32)


def loss_components(logits, batch, reflection_weight=1.0, beta=1.0, gamma=0.0, persona_only=False,
                    add_reflection_ce=False, pair_loss_type="simpo", huber_delta=None):
    delta = resolve_huber_delta(pair_loss_type, gamma, huber_delta)
    source_count = batch["source_count"]
    predictors, story, reflection = interleaved.prediction_layout(
        batch["attention_mask"], batch["refl_start"], batch["refl_end"])
    story[source_count:] = False
    zero = logits[:, 0, 0].float().sum()*0
    _, story_nll = _selected_nll(logits, batch["input_ids"], predictors, story)
    context = story_nll.sum()/len(story_nll) if len(story_nll) else zero
    pair_sources = batch["pair_source_indices"]
    pair_count = len(pair_sources) if reflection_weight else 0
    positive = negative = logits.new_empty(0, dtype=torch.float32)
    pair_loss = reflection_ce = zero
    scored_counts = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    if reflection_weight and (pair_count or add_reflection_ce):
        if persona_only:
            reflection &= batch["non_template_mask"][:, 1:].bool()
        # CE covers positive reflections; pair scoring requires both branches.
        selected_rows = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.bool)
        selected_rows[pair_sources] = True
        selected_rows[source_count:] = True
        if add_reflection_ce:
            selected_rows[:source_count] = True
        reflection &= selected_rows[:, None]
        rows, nll = _selected_nll(logits, batch["input_ids"], predictors, reflection)
        scored_counts = reflection.sum(dim=1)
        if add_reflection_ce:
            # Match IEPE's token mean, rather than averaging per-pair scores.
            positive_nll = nll[rows < source_count]
            reflection_ce = positive_nll.mean() if len(positive_nll) else zero
        if pair_count:
            negative_rows = torch.arange(source_count, logits.shape[0], device=logits.device)
            if (len(negative_rows) != pair_count or (scored_counts[pair_sources] == 0).any()
                    or (scored_counts[negative_rows] == 0).any()):
                raise ValueError("SPO requires one negative per pair and nonempty scoring masks on both sides")
            totals = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32).scatter_add(0, rows, -nll)
            means = totals/scored_counts.clamp_min(1)
            positive, negative = means[pair_sources], means[negative_rows]
            pair_loss = (simpo_pair_loss(positive, negative, beta, gamma) if pair_loss_type == "simpo"
                         else huber_hinge_pair_loss(positive, negative, beta, gamma, delta)).mean()
    reflection_loss = reflection_ce + pair_loss
    loss = context + reflection_weight*reflection_loss
    return loss, dict(context=context, simpo=pair_loss, reflection_ce=reflection_ce,
                      reflection=reflection_loss, positive=positive, negative=negative,
                      story_tokens=len(story_nll), pairs=pair_count, scored_counts=scored_counts)


class SPOTrainer(SeparatorTrackingMixin, HiddenStateTrackingMixin, Trainer):
    def __init__(self, *args, beta=1.0, gamma=0.0, reflection_loss_weight=1.0,
                 add_reflection_ce=False, pair_loss_type="simpo", huber_delta=None,
                 non_template_loss_only=False, mask_reflection=True,
                 reflection_attention_mode="full", reflection_attention_k=64,
                 reflection_attention_p=0.5, reflection_attention_include_bos=False,
                 separator_token_id=None, end_separator_token_id=None,
                 hidden_state_tracking_config=None, track_separator=False,
                 log_grad_norm=True, **kwargs):
        for name, value, valid in (("lambda", reflection_loss_weight, reflection_loss_weight >= 0),
                                   ("beta", beta, beta > 0), ("gamma", gamma, gamma >= 0)):
            if not math.isfinite(value) or not valid:
                raise ValueError(f"Invalid SPO {name}: {value}")
        huber_delta = resolve_huber_delta(pair_loss_type, gamma, huber_delta)
        if reflection_attention_mode not in ("full", "last_k", "random_p"):
            raise ValueError("SPO reflection_attention_mode must be full, last_k, or random_p")
        if reflection_attention_k < 1 or not math.isfinite(reflection_attention_p) or not 0 <= reflection_attention_p <= 1:
            raise ValueError("SPO requires k >= 1 and 0 <= p <= 1")
        super().__init__(*args, **kwargs)
        if self.args.do_eval or str(self.args.eval_strategy) not in ("no", "IntervalStrategy.NO"):
            raise ValueError("SPO pretraining evaluation is disabled; use the existing post-SFT evaluation workflow")
        if self.is_deepspeed_enabled or self.is_fsdp_enabled or isinstance(self.model, torch.nn.DataParallel) or self.args.n_gpu > 1:
            raise ValueError("SPO v1 supports a single device or one-process-per-device DDP; use torchrun")
        if "keep_torch_compile" not in inspect.signature(self.accelerator.unwrap_model).parameters:
            raise ValueError(
                "SPO requires a compatible Transformers/Accelerate stack. "
                "Use the launcher environment (tested: Transformers 4.51.3, Accelerate 1.6.0) "
                "or update Accelerate to match Transformers."
            )
        config = self.model.config
        if config.model_type != "llama" or config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("SPO v1 requires a Llama model with eager or sdpa attention for custom interleaved masks")
        self.model_accepts_loss_kwargs = False
        if self.compute_loss_func is not None:
            raise ValueError("SPO implements its own microbatch loss; compute_loss_func must be unset")
        self.beta, self.gamma = float(beta), float(gamma)
        self.reflection_loss_weight = float(reflection_loss_weight)
        self.add_reflection_ce = add_reflection_ce
        self.pair_loss_type, self.huber_delta = pair_loss_type, huber_delta
        self.non_template_loss_only = non_template_loss_only
        self.mask_reflection = mask_reflection
        self.reflection_attention_mode = reflection_attention_mode
        self.reflection_attention_k = reflection_attention_k
        self.reflection_attention_p = reflection_attention_p
        self.reflection_attention_include_bos = reflection_attention_include_bos
        self.separator_token_id = separator_token_id
        self.end_separator_token_id = end_separator_token_id
        self.log_grad_norm = log_grad_norm
        self._init_hidden_state_tracking(hidden_state_tracking_config)
        self._init_separator_tracking(separator_token_id if track_separator else None)
        self._spo_stats = None
        self.last_loss_components = None

    def _prefix_keep(self, inputs):
        if self.reflection_attention_mode == "full":
            return None
        result = []
        for start, source_seed, has_bos in zip(inputs["refl_start"].tolist(), inputs["attention_seed"].tolist(), inputs["has_bos"].tolist()):
            keep = torch.ones(max(0, start), dtype=torch.bool)
            first = int(self.reflection_attention_include_bos and has_bos)
            if self.reflection_attention_mode == "last_k":
                keep[first:max(first, start-self.reflection_attention_k)] = False
            elif start > first:
                generator = torch.Generator().manual_seed(stable_seed(self.args.seed, self.state.global_step, source_seed))
                candidates = start-first
                count = max(1, int(candidates*self.reflection_attention_p))
                blocked = torch.randperm(candidates, generator=generator)[count:]+first
                keep[blocked] = False
            result.append(keep.to(inputs["input_ids"].device))
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # For lambda=0 a caller may still supply negative rows. Remove them here
        # as well as in the training collator so no negative forward is performed.
        if self.reflection_loss_weight == 0 and inputs["input_ids"].shape[0] > inputs["source_count"]:
            inputs = dict(inputs)
            count = inputs["source_count"]
            for key in ("input_ids", "attention_mask", "non_template_mask", "refl_start", "refl_end", "attention_seed", "has_bos", "sample_idx"):
                inputs[key] = inputs[key][:count]
            inputs["pair_source_indices"] = inputs["pair_source_indices"][:0]
        starts, ends = inputs["refl_start"], inputs["refl_end"]
        attention = inputs["attention_mask"]
        if (starts >= 0).any() and (self.mask_reflection or self.reflection_attention_mode != "full"):
            attention = interleaved.attention_mask(
                attention, starts, ends, next(model.parameters()).dtype,
                mask_reflection=self.mask_reflection, prefix_keep=self._prefix_keep(inputs))
        handles = None
        if self._should_log_hidden_states():
            handles = self._hs_tracker.register_hooks(model, row_limit=inputs["source_count"],
                                                       sequence_limit=inputs["positive_max_length"])
        try:
            outputs = model(input_ids=inputs["input_ids"], attention_mask=attention,
                            position_ids=interleaved.position_ids(inputs["attention_mask"], starts, ends),
                            use_cache=False, return_dict=True)
        finally:
            if handles is not None:
                self._hs_tracker.remove_hooks(handles)
        if handles is not None:
            self._compute_and_log_hidden_state_metrics(model)
        loss, components = loss_components(outputs.logits, inputs, self.reflection_loss_weight,
                                           self.beta, self.gamma, self.non_template_loss_only,
                                           add_reflection_ce=self.add_reflection_ce,
                                           pair_loss_type=self.pair_loss_type, huber_delta=self.huber_delta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SPO training loss")
        self.last_loss_components = {k: v.detach() if torch.is_tensor(v) else v for k, v in components.items()}
        if model.training:
            gap = components["positive"]-components["negative"]
            stats = torch.stack([
                components["context"].detach(), components["simpo"].detach(),
                components["positive"].detach().sum(), components["negative"].detach().sum(),
                (gap > 0).float().sum(), (self.beta*gap > self.gamma).float().sum(),
                loss.new_tensor(components["pairs"]), loss.new_tensor(components["story_tokens"]),
                components["scored_counts"].float().sum(), loss.new_tensor(1),
                loss.new_tensor(int(components["pairs"] == 0)), loss.new_tensor(inputs["source_count"]),
                components["reflection_ce"].detach(),
            ]).double()
            self._spo_stats = stats if self._spo_stats is None else self._spo_stats+stats
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if ("loss" in logs or "train_loss" in logs) and self._spo_stats is not None:
            stats, self._spo_stats = self._spo_stats, None
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
            (ce, simpo, positive, negative, wins, margins, pairs, tokens, scored,
             microbatches, empty, documents, reflection_ce) = stats.tolist()
            logs.update(loss_context=ce/microbatches, loss_reflection=(reflection_ce+simpo)/microbatches,
                        loss_reflection_ce=reflection_ce/microbatches, loss_reflection_spo=simpo/microbatches,
                        spo_target_score_gap=self.gamma/self.beta,
                        spo_valid_pairs=pairs, spo_story_tokens=tokens, spo_source_documents=documents,
                        spo_pair_free_microbatch_fraction=empty/microbatches)
            if pairs:
                logs.update(spo_score_positive=positive/pairs, spo_score_negative=negative/pairs,
                            spo_score_gap=(positive-negative)/pairs, spo_preference_accuracy=wins/pairs,
                            spo_margin_satisfaction=margins/pairs, spo_scored_tokens_per_side=scored/(2*pairs))
        if not getattr(self, "log_grad_norm", True):
            logs.pop("grad_norm", None)
        return super().log(logs, *args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            self._log_separator_metrics(model)
        return loss

    def prediction_step(self, *args, **kwargs):
        raise ValueError("SPO pretraining evaluation is disabled; use the existing post-SFT evaluation workflow")
