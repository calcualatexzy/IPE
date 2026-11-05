"""Offline checks of msm/optim.py, on CPU:
- Newton-Schulz: singular values near 1 with the singular vectors kept; a zero matrix stays zero, rank one stays one;
- Muon's first step: the orthogonalized gradient at AdamW's update RMS of about 0.2, and its weight decay;
- Lion's first step: the sign of the gradient, and the momentum;
- schedule-free AdamW against its x, y, z recursion written out, and eval() and train() as inverses;
- build_optimizer: every trainable parameter once, LoRA+'s rates, weight decay only on the decayed names."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from peft import LoraConfig, get_peft_model

from msm.optim import Lion, Muon, ScheduleFreeAdamW, build_optimizer, orthogonalize

ARGS = SimpleNamespace(learning_rate=1e-3, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)
CONFIG = {"loraplus_lr_ratio": 16, "muon_momentum": 0.95, "lion_betas": [0.9, 0.99]}


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 6)
        self.norm = torch.nn.LayerNorm(6)

    def forward(self, x):
        return self.norm(self.proj(x))


def tiny_lora(**settings):
    return get_peft_model(Tiny(), LoraConfig(r=2, target_modules=["proj"], **settings))


class NewtonSchulzTest(unittest.TestCase):
    def test_singular_values_and_vectors(self):
        g = torch.Generator().manual_seed(0)
        for shape in ((64, 4096), (4096, 64), (16, 256)):
            with self.subTest(shape=shape):
                m = torch.randn(*shape, generator=g)
                o = orthogonalize(m)
                self.assertEqual(o.shape, m.shape)
                s = torch.linalg.svdvals(o)
                self.assertTrue(((s > 0.5) & (s < 1.5)).all(), s)
                u, _, vh = torch.linalg.svd(m, full_matrices=False)
                inner = u.T @ o @ vh.T  # diagonal when o keeps m's singular vectors
                torch.testing.assert_close(inner, torch.diag(torch.diagonal(inner)), atol=1e-3, rtol=0)

    def test_zero_and_rank_one(self):
        # LoRA's A has no gradient while B is still zero
        self.assertTrue(torch.equal(orthogonalize(torch.zeros(64, 256)), torch.zeros(64, 256)))
        g = torch.Generator().manual_seed(1)
        s = torch.linalg.svdvals(orthogonalize(torch.randn(64, 1, generator=g) @ torch.randn(1, 256, generator=g)))
        self.assertTrue(0.5 < s[0] < 1.5)
        self.assertTrue((s[1:] < 1e-3).all())


class MuonTest(unittest.TestCase):
    def test_first_step(self):
        g = torch.Generator().manual_seed(2)
        for shape in ((64, 4096), (4096, 64)):
            with self.subTest(shape=shape):
                p = torch.nn.Parameter(torch.randn(*shape, generator=g))
                start = p.detach().clone()
                p.grad = torch.randn(*shape, generator=g)
                Muon([p], lr=1.0).step()
                update = start - p.detach()
                # the first Nesterov momentum is a positive multiple of the gradient, orthogonalized alike
                torch.testing.assert_close(update, 0.2 * math.sqrt(max(shape)) * orthogonalize(p.grad),
                                           atol=1e-4, rtol=1e-3)
                self.assertTrue(0.1 < update.pow(2).mean().sqrt() < 0.3)  # AdamW's typical update RMS

    def test_weight_decay_and_zero_gradient(self):
        p = torch.nn.Parameter(torch.randn(8, 16, generator=torch.Generator().manual_seed(3)))
        start = p.detach().clone()
        p.grad = torch.zeros_like(p)
        Muon([p], lr=0.1, weight_decay=0.5).step()
        torch.testing.assert_close(p.detach(), start * (1 - 0.1 * 0.5))

    def test_matrices_only(self):
        with self.assertRaises(ValueError):
            Muon([torch.nn.Parameter(torch.zeros(3))])


class LionTest(unittest.TestCase):
    def test_first_step(self):
        start, grad = torch.tensor([1.0, -2.0, 3.0]), torch.tensor([0.5, -0.1, 0.0])
        p = torch.nn.Parameter(start.clone())
        p.grad = grad.clone()
        optimizer = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.5)
        optimizer.step()
        # the momentum starts at zero, so the step is the gradient's sign
        torch.testing.assert_close(p.detach(), start * (1 - 0.1 * 0.5) - 0.1 * grad.sign())
        torch.testing.assert_close(optimizer.state[p]["exp_avg"], 0.01 * grad)


class ScheduleFreeTest(unittest.TestCase):
    def test_against_the_recursion(self):
        """x, y and z written out, on a quadratic, with a warmup in the learning rate as Trainer's schedule gives it."""
        g = torch.Generator().manual_seed(4)
        target = torch.randn(5, 3, generator=g)
        p = torch.nn.Parameter(torch.randn(5, 3, generator=g))
        beta1, beta2, eps, wd = 0.9, 0.999, 1e-8, 0.1
        optimizer = ScheduleFreeAdamW([p], lr=0.0, betas=(beta1, beta2), eps=eps, weight_decay=wd)
        x = z = p.detach().clone()
        v, weight_sum, lr_max = torch.zeros_like(x), 0.0, 0.0
        for k, lr in enumerate([0.0, 0.05, 0.1, 0.1, 0.1]):
            y = (1 - beta1) * z + beta1 * x
            torch.testing.assert_close(p.detach(), y)
            grad = 2 * (y - target)
            p.grad = grad.clone()
            optimizer.param_groups[0]["lr"] = lr
            optimizer.step()
            v = beta2 * v + (1 - beta2) * grad ** 2
            update = grad / ((v / (1 - beta2 ** (k + 1))).sqrt() + eps) + wd * y
            lr_max = max(lr_max, lr)
            weight_sum += lr_max ** 2
            c = lr_max ** 2 / weight_sum if weight_sum else 0.0
            z = z - lr * update
            x = (1 - c) * x + c * z
        y = (1 - beta1) * z + beta1 * x
        torch.testing.assert_close(p.detach(), y)
        optimizer.eval()
        optimizer.eval()  # a second switch does nothing
        torch.testing.assert_close(p.detach(), x)
        with self.assertRaises(RuntimeError):
            optimizer.step()
        optimizer.train()
        torch.testing.assert_close(p.detach(), y)

    def test_switch_before_any_step(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = ScheduleFreeAdamW([p])
        optimizer.eval()
        optimizer.train()
        self.assertTrue(torch.equal(p.detach(), torch.ones(2, 2)))


class BuildTest(unittest.TestCase):
    def groups(self, optimizer):
        return {id(p): (group["lr"], group["weight_decay"]) for group in optimizer.param_groups
                for p in group["params"]}

    def test_rates_and_weight_decay(self):
        model = tiny_lora(modules_to_save=["norm"])  # a trainable LayerNorm, outside the decayed names
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        decay = {name for name in trainable if "lora_" in name}
        for name, cls in (("loraplus", torch.optim.AdamW), ("lion", Lion), ("schedule_free_adamw", ScheduleFreeAdamW)):
            with self.subTest(optimizer=name):
                optimizer = build_optimizer(model, {**CONFIG, "name": name}, ARGS, decay)
                self.assertIsInstance(optimizer, cls)
                groups = self.groups(optimizer)
                self.assertEqual(set(groups), {id(p) for p in trainable.values()})
                self.assertEqual(sum(len(group["params"]) for group in optimizer.param_groups), len(trainable))
                for param_name, p in trainable.items():
                    lr, wd = groups[id(p)]
                    ratio = 16 if name == "loraplus" and "lora_B" in param_name else 1
                    self.assertAlmostEqual(lr, ARGS.learning_rate * ratio)
                    self.assertEqual(wd, ARGS.weight_decay if param_name in decay else 0.0)
        optimizer = build_optimizer(model, {**CONFIG, "name": "schedule_free_adamw"}, ARGS, decay)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.999))

    def test_muon_and_errors(self):
        model = tiny_lora()
        optimizer = build_optimizer(model, {**CONFIG, "name": "muon"}, ARGS, set())
        self.assertIsInstance(optimizer, Muon)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.95)
        with self.assertRaises(ValueError):  # a LayerNorm's vectors
            build_optimizer(tiny_lora(modules_to_save=["norm"]), {**CONFIG, "name": "muon"}, ARGS, set())
        with self.assertRaises(ValueError):  # no LoRA B to speed up
            build_optimizer(Tiny(), {**CONFIG, "name": "loraplus"}, ARGS, set())
        for name in ("adamw", "sgd"):
            with self.assertRaises(ValueError):
                build_optimizer(model, {**CONFIG, "name": name}, ARGS, set())


if __name__ == "__main__":
    unittest.main()
