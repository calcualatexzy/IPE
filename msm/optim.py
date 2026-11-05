"""The optimizers of msm/train_distill.py other than Trainer's own AdamW, selected by optimizer.name in
msm/conf/train_distill.yaml. Each takes Trainer's learning rate, schedule, warmup and weight decay, so that only the
update rule changes:
    muon                 orthogonalized momentum (Jordan et al. 2024): each matrix's update is its momentum with every
                         singular value moved to about 1 by a Newton-Schulz iteration, then scaled by
                         0.2 * sqrt(max(rows, cols)) to AdamW's typical update RMS (Liu et al. 2025, "Muon is Scalable
                         for LLM Training"), so that AdamW's learning rate and weight decay carry over
    loraplus             AdamW with lora_B at loraplus_lr_ratio times lora_A's learning rate (Hayou et al. 2024)
    lion                 the sign of an interpolation of momentum and gradient (Chen et al. 2023). Its steps are larger
                         than AdamW's: a learning rate 3-10x lower and a weight decay 3-10x higher keep lr x wd.
    schedule_free_adamw  AdamW without a decaying schedule (Defazio et al. 2024). The parameters train at y, between an
                         SGD-like sequence z and its average x, and are evaluated and saved at x: eval() moves them to
                         x, train() back to y. Trainer.training_step calls train(); DistillTrainer's evaluation and
                         saving call eval(). The warmup still comes from Trainer (lr_scheduler_type=constant_with_warmup).
Muon, Lion and schedule-free AdamW are written here: torch 2.6 has no Muon (torch.optim.Muon came with 2.9), and neither
bitsandbytes nor schedulefree is installed. Muon follows torch.optim.Muon with adjust_lr_fn="match_rms_adamw", and
schedule-free AdamW the reference AdamWScheduleFree (github.com/facebookresearch/schedule_free).
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch.optim import AdamW, Optimizer

OPTIMIZERS = ("adamw", "muon", "loraplus", "lion", "schedule_free_adamw")  # adamw: Trainer's own
SCHEDULE_FREE_SCHEDULES = ("constant", "constant_with_warmup")  # schedule-free AdamW replaces the decay
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # Muon's quintic Newton-Schulz iteration
NS_STEPS = 5


def orthogonalize(g: torch.Tensor, steps: int = NS_STEPS, eps: float = 1e-7) -> torch.Tensor:
    """g with its singular vectors kept and each singular value moved to about 1 (between 0.5 and 1.5), by Muon's
    quintic Newton-Schulz iteration, in float32. A zero matrix stays zero."""
    a, b, c = NS_COEFFICIENTS
    x = g.float()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T  # iterate on the smaller Gram matrix: the LoRA rank's
    x = x / (x.norm() + eps)  # the spectral norm at most 1
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.T if tall else x).to(g.dtype)


def _closure_loss(closure):
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()


class Muon(Optimizer):
    """Muon for matrices: momentum (Nesterov by default), orthogonalized, scaled by 0.2 * sqrt(max(rows, cols)), with
    decoupled weight decay at the unscaled learning rate, as torch.optim.Muon with adjust_lr_fn="match_rms_adamw"."""

    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0.0, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = NS_STEPS):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon updates matrices only, not a parameter of shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = _closure_loss(closure)
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - beta)
                update = p.grad.lerp(buf, beta) if group["nesterov"] else buf
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(orthogonalize(update, group["ns_steps"]), alpha=-lr * 0.2 * math.sqrt(max(p.shape)))
        return loss


class Lion(Optimizer):
    """Lion (Chen et al. 2023): the sign of beta1 * momentum + (1 - beta1) * gradient, with decoupled weight decay; the
    momentum then moves toward the gradient by 1 - beta2."""

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = _closure_loss(closure)
        for group in self.param_groups:
            lr, (beta1, beta2) = group["lr"], group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(exp_avg.lerp(p.grad, 1 - beta1).sign_(), alpha=-lr)
                exp_avg.lerp_(p.grad, 1 - beta2)
        return loss


class ScheduleFreeAdamW(Optimizer):
    """Schedule-free AdamW (Defazio et al. 2024), as the reference AdamWScheduleFree with its defaults r = 0 and
    weight_lr_power = 2, but without its own warmup: Trainer's schedule scales the learning rate as that warmup would.
        z  <- z - lr * (grad(y) / (sqrt(v / bias correction) + eps) + weight_decay * y)
        x  <- (1 - c) x + c z      c = lr_max^2 / (sum of lr_max^2 over the steps so far)
        y  =  (1 - beta1) z + beta1 x
    In train mode the parameters hold y, where the gradient is taken; eval() moves them to x and train() back to y. A
    new optimizer is in train mode."""

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, weight_lr_power: float = 2.0):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                      weight_lr_power=weight_lr_power, k=0, weight_sum=0.0, lr_max=0.0,
                                      train_mode=True))

    @torch.no_grad()
    def eval(self) -> None:
        self._switch(train=False)

    @torch.no_grad()
    def train(self) -> None:
        self._switch(train=True)

    def _switch(self, train: bool) -> None:
        for group in self.param_groups:
            if group["train_mode"] == train:
                continue
            beta1 = group["betas"][0]
            for p in group["params"]:
                if "z" in self.state[p]:
                    # y = x + (1 - beta1) (z - x), and back: x = y + (1 - 1 / beta1) (z - y)
                    p.lerp_(self.state[p]["z"], 1 - beta1 if train else 1 - 1 / beta1)
            group["train_mode"] = train

    @torch.no_grad()
    def step(self, closure=None):
        if not all(group["train_mode"] for group in self.param_groups):
            raise RuntimeError("ScheduleFreeAdamW steps in train mode only: call train() after eval()")
        loss = _closure_loss(closure)
        for group in self.param_groups:
            lr, (beta1, beta2), k = group["lr"], group["betas"], group["k"]
            group["lr_max"] = max(lr, group["lr_max"])
            weight = group["lr_max"] ** group["weight_lr_power"]
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0  # c, x's step toward z
            bias_correction2 = 1 - beta2 ** (k + 1)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["z"] = p.clone()
                    state["exp_avg_sq"] = torch.zeros_like(p)
                z, exp_avg_sq = state["z"], state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                update = p.grad / (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                update.add_(p, alpha=group["weight_decay"])  # weight decay at y
                # y moves as x and z do, without computing x
                p.lerp_(z, ckp1)
                p.add_(update, alpha=lr * (beta1 * (1 - ckp1) - 1))
                z.sub_(update, alpha=lr)
            group["k"] = k + 1
        return loss


def build_optimizer(model: torch.nn.Module, config: dict, args, decay_names: set[str]) -> Optimizer:
    """The optimizer that config (the optimizer settings of msm/conf/train_distill.yaml) names, other than Trainer's
    adamw, over model's trainable parameters. As Trainer's AdamW: args.learning_rate, args.weight_decay on decay_names
    only (Trainer.get_decay_parameter_names), and for loraplus and schedule_free_adamw AdamW's betas and epsilon."""
    name = config["name"]
    if name not in OPTIMIZERS[1:]:
        raise ValueError(f"build_optimizer builds one of {OPTIMIZERS[1:]}, not {name!r} (adamw is Trainer's own)")
    groups = defaultdict(list)
    n_lora_b = 0
    for param_name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lora_b = "lora_B" in param_name
        n_lora_b += lora_b
        ratio = config["loraplus_lr_ratio"] if name == "loraplus" and lora_b else 1
        groups[args.learning_rate * ratio, args.weight_decay if param_name in decay_names else 0.0].append(p)
    if name == "loraplus" and not n_lora_b:
        raise ValueError("loraplus needs LoRA B matrices, and none of the trainable parameters is one")
    param_groups = [{"params": params, "lr": lr, "weight_decay": wd} for (lr, wd), params in groups.items()]
    adam = dict(betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)
    if name == "muon":
        return Muon(param_groups, momentum=config["muon_momentum"])
    if name == "loraplus":
        return AdamW(param_groups, **adam)
    if name == "lion":
        return Lion(param_groups, betas=config["lion_betas"])
    return ScheduleFreeAdamW(param_groups, **adam)
