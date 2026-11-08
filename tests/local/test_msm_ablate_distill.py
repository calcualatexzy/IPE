"""Offline checks of the distillation ablations, without models, CUDA or conda:
- official_muon (msm/optim.py): torch.optim.Muon's first step is msm/optim.py's Muon up to bfloat16, and Keller
  Jordan's is scaled by sqrt(max(1, rows / cols)) instead; each runs only where its code is installed (the ipe-muon
  env), and msm/optim.py's Muon stays the default;
- scripts/msm/ablate_distill.sh: the three launchers run as they are, under stubs of torchrun, python and conda;
  torchrun records each call and saves a fake adapter. Every experiment and seed trains a student with its line's
  settings and continues that student with AFT, and one eval reads every AFT adapter. A rerun skips the finished
  stages and refuses changed settings, a failed run does not stop the others, and a bad line stops everything before
  any run."""
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model

from msm.optim import Muon, build_optimizer, official_muon, orthogonalize

try:
    import muon
except ImportError:
    muon = None

HAS_TORCH_MUON = hasattr(torch.optim, "Muon")
ARGS = SimpleNamespace(learning_rate=1e-3, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)
CONFIG = {"loraplus_lr_ratio": 16, "muon_momentum": 0.95, "lion_betas": [0.9, 0.99]}


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 6)

    def forward(self, x):
        return self.proj(x)


def tiny_lora():
    return get_peft_model(Tiny(), LoraConfig(r=2, target_modules=["proj"]))


def first_step(make_optimizer, shape, seed, weight_decay=0.0, zero_grad=False):
    """The first update of a make_optimizer(param_groups) at learning rate 1 on a random matrix, and its gradient."""
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(*shape, generator=g))
    start = p.detach().clone()
    grad = torch.zeros(*shape) if zero_grad else torch.randn(*shape, generator=g)
    p.grad = grad.clone()  # Keller Jordan's Muon overwrites the gradient
    make_optimizer([{"params": [p], "lr": 1.0, "weight_decay": weight_decay}]).step()
    return start - p.detach(), grad, start


def relative(a, b):
    return ((a - b).norm() / b.norm()).item()


class OfficialMuonTest(unittest.TestCase):
    SHAPES = ((64, 4096), (4096, 64), (16, 256))

    @unittest.skipUnless(HAS_TORCH_MUON, "torch.optim.Muon needs torch >= 2.9 (the ipe-muon env)")
    def test_torch_is_ours_in_bfloat16(self):
        for seed, shape in enumerate(self.SHAPES):
            with self.subTest(shape=shape):
                theirs, _, _ = first_step(lambda groups: official_muon(groups, "torch", 0.95), shape, seed)
                ours, _, _ = first_step(lambda groups: Muon(groups, momentum=0.95), shape, seed)
                self.assertLess(relative(theirs, ours), 0.02)
        update, _, start = first_step(lambda groups: official_muon(groups, "torch", 0.95), (8, 16), 0,
                                      weight_decay=0.5, zero_grad=True)
        torch.testing.assert_close(update, start * 0.5)  # decoupled, at the unscaled learning rate

    @unittest.skipUnless(HAS_TORCH_MUON, "torch.optim.Muon needs torch >= 2.9 (the ipe-muon env)")
    def test_torch_groups(self):
        model = tiny_lora()
        decay = {name for name, p in model.named_parameters() if p.requires_grad and "lora_A" in name}
        optimizer = build_optimizer(model, {**CONFIG, "name": "muon", "muon_impl": "torch"}, ARGS, decay)
        self.assertIsInstance(optimizer, torch.optim.Muon)
        self.assertEqual({(group["lr"], group["weight_decay"]) for group in optimizer.param_groups},
                         {(1e-3, 0.1), (1e-3, 0.0)})
        self.assertEqual({group["momentum"] for group in optimizer.param_groups}, {0.95})

    @unittest.skipUnless(muon, "github.com/KellerJordan/Muon is not installed (the ipe-muon env has it)")
    def test_keller_scale(self):
        for seed, shape in enumerate(self.SHAPES):
            with self.subTest(shape=shape):
                update, grad, _ = first_step(lambda groups: official_muon(groups, "keller", 0.95), shape, seed)
                # the first Nesterov momentum is a positive multiple of the gradient, orthogonalized alike
                self.assertLess(relative(update, math.sqrt(max(1, shape[0] / shape[1])) * orthogonalize(grad)), 0.02)
        optimizer = build_optimizer(tiny_lora(), {**CONFIG, "name": "muon", "muon_impl": "keller"}, ARGS, set())
        self.assertIsInstance(optimizer, muon.SingleDeviceMuon)

    def test_default_and_errors(self):
        self.assertEqual(OmegaConf.load(ROOT / "msm/conf/train_distill.yaml").optimizer.muon_impl, "ours")
        for config in (CONFIG, {**CONFIG, "muon_impl": "ours"}):  # configs from before muon_impl mean ours
            self.assertIs(type(build_optimizer(tiny_lora(), {**config, "name": "muon"}, ARGS, set())), Muon)
        with self.assertRaises(ValueError):
            build_optimizer(tiny_lora(), {**CONFIG, "name": "muon", "muon_impl": "jordan"}, ARGS, set())
        vector = [{"params": [torch.nn.Parameter(torch.zeros(3))], "lr": 1.0, "weight_decay": 0.0}]
        for impl in ("torch", "keller"):
            with self.subTest(impl=impl), self.assertRaises(ValueError):  # matrices only, before any import
                official_muon(vector, impl, 0.95)
        matrix = [{"params": [torch.nn.Parameter(torch.zeros(3, 3))], "lr": 1.0, "weight_decay": 0.0}]
        for impl, available in (("torch", HAS_TORCH_MUON), ("keller", muon is not None)):
            if not available:
                with self.subTest(impl=impl), self.assertRaises(ImportError):
                    official_muon(matrix, impl, 0.95)


# torchrun records its arguments, one per line, in calls/<n>; a training module then saves a fake adapter and logs
# it as msm/train_distill.py and msm/train_aft.py do. It fails instead on the argument STUB_FAIL_ON.
TORCHRUN = r"""#!/bin/bash
n=$(( $(ls "$STUB_DIR/calls" | wc -l) ))
printf '%s\n' "$@" > "$STUB_DIR/calls/$n"
for arg; do
  if [[ -n ${STUB_FAIL_ON:-} && $arg == "$STUB_FAIL_ON" ]]; then echo "stub: failing on $arg" >&2; exit 3; fi
done
for arg; do
  if [[ $arg == msm.train_distill || $arg == msm.train_aft ]]; then
    mkdir -p "$STUB_DIR/out/$n/final"
    echo '{}' > "$STUB_DIR/out/$n/final/adapter_config.json"
    echo "2026-11-05 12:00:00.000 | INFO     | $arg:main:1 - Saved the adapter to $STUB_DIR/out/$n/final"
  fi
done
"""
PYTHON = '#!/bin/bash\nif [[ "${1:-}" == -c ]]; then echo 4; exit 0; fi\nexit 1\n'  # the GPU probe sees four


def replace_block(source: str, name: str, lines: list[str], quote: str) -> str:
    """source with the bash array name=( ... ) holding lines instead."""
    body = "".join(f"  {quote}{line}{quote}\n" for line in lines)
    source, count = re.subn(rf"^{name}=\(\n.*?^\)$", lambda _: f"{name}=(\n{body})", source, count=1,
                            flags=re.MULTILINE | re.DOTALL)
    assert count == 1, name
    return source


class AblateLauncherTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "calls").mkdir()
        bin_dir = self.dir / "bin"
        bin_dir.mkdir()
        for command, text in (("torchrun", TORCHRUN), ("python", PYTHON)):
            (bin_dir / command).write_text(text)
            (bin_dir / command).chmod(0o755)
        (self.dir / "conda.sh").write_text("conda() { :; }\n")
        self.old = self.dir / "old/final"  # an adapter of an earlier sweep, for EVAL_EXTRA
        self.old.mkdir(parents=True)
        (self.old / "adapter_config.json").write_text("{}")
        self.env = {key: value for key, value in os.environ.items()
                    if key not in ("ABLATION", "REFERENCE", "DIRECTION", "OPTIMIZER", "TEACHER", "DRY_RUN")}
        self.env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}", "CONDA_SH": str(self.dir / "conda.sh"),
                         "CONDA_ENV": "unused", "PROJECT_ROOT": str(ROOT), "NPROC": "4",
                         "OUT_DIR": str(self.dir / "ablate"), "STUB_DIR": str(self.dir), "OPENROUTER_API_KEY": "test"})

    def script(self, experiments, extra=(), **settings):
        """ablate_distill.sh with these EXPERIMENTS lines and EVAL_EXTRA adapters, and settings replacing the lines
        that set them (e.g. PRECISION="fp16")."""
        source = (ROOT / "scripts/msm/ablate_distill.sh").read_text()
        source = replace_block(source, "EXPERIMENTS", experiments, '"')
        source = replace_block(source, "EVAL_EXTRA", [str(adapter) for adapter in extra], "'")
        for key, value in settings.items():
            source, count = re.subn(rf"^{key}=.*$", f"{key}={value}", source, count=1, flags=re.MULTILINE)
            self.assertEqual(count, 1, key)
        path = self.dir / "ablate_distill.sh"
        path.write_text(source)
        return path

    def run_script(self, path, check=True, **environment):
        result = subprocess.run(["bash", str(path)], env={**self.env, **environment}, capture_output=True, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self):
        """(module, overrides as a config, later ones winning as in Hydra) of every torchrun call, in order."""
        calls = []
        for path in sorted((self.dir / "calls").iterdir(), key=lambda p: int(p.name)):
            args = path.read_text().splitlines()
            start = args.index("-m") + 1
            calls.append((args[start], OmegaConf.from_dotlist(args[start + 1:])))
        return calls

    def adapter(self, call):
        return f"{self.dir}/out/{call}/final"

    def test_experiments_chain_the_launchers(self):
        path = self.script(["NAME=adamw OPTIMIZER=adamw",
                            "NAME=muon-lr3e-4 OPTIMIZER=muon LR=3e-4 MOMENTUM=0.9 SEEDS=1,2 training.warmup_ratio=0.1 "
                            "aft:training.learning_rate=2e-4"], extra=[self.old])
        result = self.run_script(path)
        calls = self.calls()
        self.assertEqual([module for module, _ in calls], ["msm.train_distill", "msm.train_aft"] * 3 + ["msm.eval"])
        distill, aft = calls[0][1], calls[1][1]
        # the sweep's teacher and preset, not those train_distill.sh is left at
        self.assertEqual(distill.cache, "data/msm/distill/main/teacher_cache/a1-clip4-base-k64-*")
        self.assertEqual((distill.loss.ce_weight, distill.loss.kd_weight, distill.optimizer.name, distill.seed),
                         (1.0, 1.0, "adamw", 42))
        self.assertAlmostEqual(float(distill.training.learning_rate), 1e-4)
        self.assertNotIn("muon_impl", distill.optimizer)
        self.assertIn(distill.suffix, ("", None))
        self.assertEqual((distill.training.bf16, distill.training.fp16, distill.wandb.enabled), (True, False, True))
        # AFT continues that student, not train_aft.sh's INIT_ADAPTER
        self.assertEqual((aft.init_adapter, aft.seed, aft.suffix, aft.data.aft),
                         (self.adapter(0), 42, "distill-adamw", "chloeli/aft-llama-cheese"))
        self.assertAlmostEqual(float(aft.training.learning_rate), 1e-4)
        for k, seed in enumerate((1, 2)):
            with self.subTest(seed=seed):
                distill, aft = calls[2 + 2 * k][1], calls[3 + 2 * k][1]
                self.assertEqual((distill.optimizer.name, distill.optimizer.muon_impl, distill.seed),
                                 ("muon", "torch", seed))
                self.assertAlmostEqual(float(distill.training.learning_rate), 3e-4)
                self.assertEqual((distill.optimizer.muon_momentum, distill.training.warmup_ratio), (0.9, 0.1))
                self.assertAlmostEqual(float(distill.training.weight_decay), 0.01)
                self.assertEqual(distill.suffix, "torch_mom0.9_warmup_ratio0.1")
                self.assertEqual((aft.init_adapter, aft.seed, aft.suffix),
                                 (self.adapter(2 + 2 * k), seed, "distill-muon-lr3e-4"))
                self.assertAlmostEqual(float(aft.training.learning_rate), 2e-4)
        evaluation = calls[6][1]
        self.assertEqual(list(evaluation.adapters), [str(self.old), self.adapter(1), self.adapter(3), self.adapter(5)])
        self.assertEqual(evaluation.dtype, "bfloat16")
        self.assertIn("3 of 3 runs done", result.stdout)

    def test_fp16(self):
        self.run_script(self.script(["NAME=muon OPTIMIZER=muon MUON_IMPL=keller"], PRECISION="fp16"))
        (_, distill), (_, aft), (_, evaluation) = self.calls()
        for cfg in (distill, aft):
            self.assertEqual((cfg.training.bf16, cfg.training.fp16), (False, True))
        self.assertEqual((distill.optimizer.muon_impl, distill.suffix), ("keller", "keller"))
        self.assertEqual(evaluation.dtype, "float16")

    def test_rerun_skips_finished_stages(self):
        path = self.script(["NAME=muon OPTIMIZER=muon"])
        self.run_script(path)
        self.run_script(path)  # only the eval again
        calls = self.calls()
        self.assertEqual([module for module, _ in calls], ["msm.train_distill", "msm.train_aft", "msm.eval", "msm.eval"])
        self.assertEqual(list(calls[3][1].adapters), [self.adapter(1)])
        dry = self.run_script(path, DRY_RUN="true")
        self.assertIn("AFT done", dry.stdout)
        self.assertEqual(len(self.calls()), 4)
        # another rate under the same NAME and SWEEP is refused before anything runs
        result = self.run_script(self.script(["NAME=muon OPTIMIZER=muon LR=3e-4"]), check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ran with other settings", result.stderr)
        self.assertEqual(len(self.calls()), 4)

    def test_failed_run_does_not_stop_the_others(self):
        path = self.script(["NAME=muon-lr3e-4 OPTIMIZER=muon LR=3e-4", "NAME=adamw OPTIMIZER=adamw"])
        result = self.run_script(path, check=False, STUB_FAIL_ON="training.learning_rate=3e-4")
        self.assertEqual(result.returncode, 1)
        self.assertIn("failed in train_distill.sh", result.stdout)
        calls = self.calls()
        self.assertEqual([module for module, _ in calls],
                         ["msm.train_distill", "msm.train_distill", "msm.train_aft", "msm.eval"])
        self.assertEqual(list(calls[3][1].adapters), [self.adapter(2)])
        self.run_script(path)  # retries the failed run only
        calls = self.calls()
        self.assertEqual([module for module, _ in calls[4:]], ["msm.train_distill", "msm.train_aft", "msm.eval"])
        self.assertEqual(list(calls[6][1].adapters), [self.adapter(5), self.adapter(2)])

    def test_bad_settings_stop_before_any_run(self):
        for lines in (["OPTIMIZER=muon"], ["NAME=a_b"], ["NAME=a OPTIMIZER=sgd"], ["NAME=a TEACHER=mistral"],
                      ["NAME=a ABLATION=kd"], ["NAME=a OPTIMIZER=muon MUON_IMPL=jordan"],
                      ["NAME=a OPTIMIZER=adamw MOMENTUM=0.9"], ["NAME=a OPTIMIZER=adamw MUON_IMPL=torch"],
                      ["NAME=a LR=fast"], ["NAME=a SEEDS=one"], ["NAME=a BATCH=4"], ["NAME=a", "NAME=a OPTIMIZER=muon"]):
            with self.subTest(lines=lines):
                result = self.run_script(self.script(lines), check=False)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        result = self.run_script(self.script(["NAME=a"], extra=["outputs/msm/train/missing/final"]), check=False)
        self.assertIn("EVAL_EXTRA", result.stderr)
        result = self.run_script(self.script(["NAME=a"]), check=False, OPENROUTER_API_KEY="")
        self.assertIn("OPENROUTER_API_KEY", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_default_sweep(self):
        """The sweep as committed: AdamW and torch's Muon at four of AdamW's rates, in the ipe-muon env."""
        source = (ROOT / "scripts/msm/ablate_distill.sh").read_text()
        lines = re.findall(r'^  "(NAME=[^"]*)"$', source, flags=re.MULTILINE)
        self.assertEqual(lines, ["NAME=adamw OPTIMIZER=adamw", "NAME=muon-lr3e-5 OPTIMIZER=muon LR=3e-5",
                                 "NAME=muon-lr1e-4 OPTIMIZER=muon", "NAME=muon-lr3e-4 OPTIMIZER=muon LR=3e-4",
                                 "NAME=muon-lr1e-3 OPTIMIZER=muon LR=1e-3"])
        for setting in ("ABLATION=full", "TEACHER=llama", "SEEDS=(42)", "MUON_IMPL=torch", "PRECISION=bf16",
                        "EVAL=true", "CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-muon}"):
            self.assertRegex(source, rf"(?m)^{re.escape(setting)}(\s|$)")
        for adapter in re.findall(r"^  '(outputs/msm/train/[^']*)'$", source, flags=re.MULTILINE):
            self.assertTrue(adapter.endswith("/final"), adapter)
        result = self.run_script(self.script(lines), DRY_RUN="true")
        self.assertEqual(self.calls(), [])
        commands = [line for line in result.stdout.splitlines() if line.startswith("env ABLATION=full TEACHER=llama")]
        self.assertEqual(len(commands), 5)
        self.assertEqual(sum("optimizer.muon_impl=torch" in command for command in commands), 4)
        self.assertEqual(sum("msm/train_aft.sh" in line for line in result.stdout.splitlines()), 5)
        self.assertIn("msm/eval.sh", result.stdout)


if __name__ == "__main__":
    unittest.main()
