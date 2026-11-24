"""scripts/align_distill/sweep.sh offline: the three launchers run as they are, under stubs of torchrun, python and
conda, in a temporary project root (its scripts/ the repository's, its data fake). torchrun records each call and saves
a fake adapter; python records the eval's call. Every experiment and seed trains its student once and continues it
with AFT for each variant and size, and one eval reads every adapter. A rerun skips finished stages, a grown line
trains only what was added, changed settings of a finished stage are refused, a failed stage does not stop the others,
the judge runs without GPUs on what finished, and a bad line stops everything before any run."""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

SWEEP = ROOT / "scripts/align_distill/sweep.sh"
# Settings the sweep and its launchers read from the environment, cleared so that the caller's do not leak in
SETTINGS = ("MODEL", "STAGE", "NPROC", "NAME", "TEACHER", "PRESET", "MSM", "SEED", "SUFFIX", "INIT_ADAPTER", "AFT",
            "AFT_SIZE", "IT_N", "DRY_RUN", "OUT_DIR", "HF_HUB_OFFLINE", "OPENROUTER_API_KEY", "ANCHOR")
# train_distill.sh's teacher cache for each preset (teacher_cache.sh's targets)
CACHE_TARGETS = {"full": "neg_align", "kd_only": "neg_align", "ce_only": "neg_align", "context": "neutral",
                 "existential": "neg_align_existential", "judgment": "neg_align_judgment"}
SIZES = (1250, 2500, 5000, 9959)

# torchrun records its arguments, one per line, in calls/<n>; a training module then saves a fake adapter and logs it
# as align_distill/train_distill.py and train_aft.py do. It fails instead on the argument STUB_FAIL_ON.
TORCHRUN = r"""#!/bin/bash
n=$(ls "$STUB_DIR/calls" | wc -l)
printf '%s\n' "$@" > "$STUB_DIR/calls/$n"
for arg; do
  if [[ -n ${STUB_FAIL_ON:-} && $arg == "$STUB_FAIL_ON" ]]; then echo "stub: failing on $arg" >&2; exit 3; fi
done
for arg; do
  if [[ $arg == align_distill.train_distill || $arg == align_distill.train_aft ]]; then
    mkdir -p "$STUB_DIR/out/$n/final"
    echo '{}' > "$STUB_DIR/out/$n/final/adapter_config.json"
    echo "2026-11-12 12:00:00.000 | INFO     | $arg:main:1 - Saved the adapter to $STUB_DIR/out/$n/final"
  fi
done
"""
# python: the GPU probe (python -c <code>) prints STUB_PROBE; the EVAL_EXTRA check (python -c <code> <args>) exits
# with STUB_EXTRA_EXIT; python -m (the eval) records its arguments in calls/<n>.
PYTHON = r"""#!/bin/bash
if [[ ${1:-} == -c ]]; then
  if (( $# > 2 )); then exit "${STUB_EXTRA_EXIT:-0}"; fi
  echo "${STUB_PROBE:-4 8}"
  exit 0
fi
if [[ ${1:-} == -m ]]; then
  n=$(ls "$STUB_DIR/calls" | wc -l)
  printf '%s\n' "$@" > "$STUB_DIR/calls/$n"
  exit 0
fi
exit 1
"""
CONFIGS = {"align_distill.train_distill": "train_distill", "align_distill.train_aft": "train_aft",
           "align_distill.eval.am": "eval_am"}


def replace_block(source: str, name: str, lines: list[str], quote: str) -> str:
    """source with the bash array name=( ... ) holding lines instead."""
    body = "".join(f"  {quote}{line}{quote}\n" for line in lines)
    source, count = re.subn(rf"^{name}=\(\n.*?^\)$", lambda _: f"{name}=(\n{body})", source, count=1,
                            flags=re.MULTILINE | re.DOTALL)
    assert count == 1, name
    return source


def committed_lines(name: str, quote: str, commented: bool) -> list[str]:
    """The lines of the committed sweep's bash array name, commented out or not."""
    source = SWEEP.read_text()
    block = re.search(rf"^{name}=\(\n(.*?)^\)$", source, flags=re.MULTILINE | re.DOTALL).group(1)
    prefix = r"  # " if commented else r"  "
    return re.findall(rf"^{prefix}{quote}([^{quote}]*){quote}", block, flags=re.MULTILINE)


class SweepTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.root = self.dir / "project"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts/align_distill").symlink_to(ROOT / "scripts/align_distill")
        (self.dir / "calls").mkdir()
        bin_dir = self.dir / "bin"
        bin_dir.mkdir()
        for command, text in (("torchrun", TORCHRUN), ("python", PYTHON)):
            (bin_dir / command).write_text(text)
            (bin_dir / command).chmod(0o755)
        (self.dir / "conda.sh").write_text("conda() { :; }\n")
        for name in ("msm-20m", "other"):
            for split in ("train", "heldout"):
                self.touch(f"data/align_distill/msm/{name}/{split}.jsonl")
        for variant in ("cot", "no-cot"):
            self.touch(f"data/align_distill/aft/qwen25/{variant}_heldout.jsonl")
            for size in SIZES:
                self.touch(f"data/align_distill/aft/qwen25/{variant}_{size}.jsonl")
        self.touch("data/align_distill/aft/qwen25/it_10000.jsonl")
        self.env = {key: value for key, value in os.environ.items() if key not in SETTINGS}
        self.env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}", "CONDA_SH": str(self.dir / "conda.sh"),
                         "CONDA_ENV": "unused", "PROJECT_ROOT": str(self.root), "STUB_DIR": str(self.dir),
                         "OPENROUTER_API_KEY": "test"})

    def touch(self, relative: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return path

    def cache(self, pairs: str, target: str, key: str = "abcd1234", finished: bool = True) -> Path:
        """A teacher cache directory as teacher_cache.py leaves it (meta.json once finished)."""
        path = self.root / f"data/align_distill/distill/{pairs}/teacher_cache/qwen25/{target}-a1-clip4-k64-{key}"
        path.mkdir(parents=True)
        if finished:
            (path / "meta.json").write_text("{}")
        return path

    def script(self, experiments, extra=(), comparisons=(), **settings):
        """sweep.sh with these EXPERIMENTS, EVAL_EXTRA and COMPARISONS lines, and settings replacing the lines that set
        them (e.g. SWEEP="g0"). AFT_VARIANTS is both variants unless set, whatever the committed sweep runs."""
        settings.setdefault("AFT_VARIANTS", "(cot no-cot)")
        source = SWEEP.read_text()
        source = replace_block(source, "EXPERIMENTS", experiments, '"')
        source = replace_block(source, "EVAL_EXTRA", list(extra), "'")
        source = replace_block(source, "COMPARISONS", list(comparisons), "'")
        for key, value in settings.items():
            source, count = re.subn(rf"^{key}=.*$", f"{key}={value}", source, count=1, flags=re.MULTILINE)
            self.assertEqual(count, 1, key)
        path = self.dir / "sweep.sh"
        path.write_text(source)
        return path

    def run_script(self, path, check=True, **environment):
        result = subprocess.run(["bash", str(path)], env={**self.env, **environment}, capture_output=True, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self):
        """(module, its config composed from the overrides, later ones winning as in Hydra) of every call, in order."""
        calls = []
        for path in sorted((self.dir / "calls").iterdir(), key=lambda p: int(p.name)):
            args = path.read_text().splitlines()
            start = args.index("-m") + 1
            with initialize_config_dir(config_dir=str(ROOT / "align_distill/conf"), version_base=None):
                calls.append((args[start], compose(CONFIGS[args[start]], overrides=args[start + 1:])))
        return calls

    def modules(self):
        return [module.rsplit(".", 1)[-1] for module, _ in self.calls()]

    def adapter(self, call):
        return f"{self.dir}/out/{call}/final"

    def run_dir(self, run_id, sweep="main"):
        return self.root / f"outputs/align_distill/sweep/qwen25/{sweep}/{run_id}"

    def test_g0_retrain_chains_the_launchers(self):
        path = self.script(["NAME=aft-only MID=none", "NAME=msm MID=msm"],
                           extra=["AFT CoT (paper)=paper:philosophy-spec-aft-cot"],
                           comparisons=["msm cot-2500 s42 vs aft-only cot-2500 s42",
                                        "aft-only cot-2500 s42 vs AFT CoT (paper)",
                                        "distill cot-2500 s42 vs aft-only cot-2500 s42"])
        result = self.run_script(path)
        calls = self.calls()
        self.assertEqual(self.modules(), ["train_aft", "train_aft", "train_distill", "train_aft", "train_aft", "am"])
        data = "data/align_distill/aft/qwen25/"
        for call, (variant, init) in enumerate([("cot", None), ("no-cot", None), (None, None),
                                                ("cot", self.adapter(2)), ("no-cot", self.adapter(2))]):
            cfg = calls[call][1]
            with self.subTest(call=call):
                self.assertEqual((cfg.model, cfg.seed), ("qwen25", 42))
                if variant is None:  # MSM: CE on the corpus subsample, no cache, no KD
                    self.assertEqual((cfg.data.corpus, cfg.data.heldout, cfg.data.cache, cfg.loss.kd_weight),
                                     ("data/align_distill/msm/msm-20m/train.jsonl",
                                      "data/align_distill/msm/msm-20m/heldout.jsonl", None, 0))
                    self.assertIn(cfg.suffix, ("", None))
                    continue
                self.assertEqual((cfg.init_adapter, cfg.data.aft, cfg.data.heldout, cfg.data.it),
                                 (init, f"{data}{variant}_2500.jsonl", f"{data}{variant}_heldout.jsonl",
                                  f"{data}it_10000.jsonl"))
                self.assertIn(cfg.suffix, ("", None))
        evaluation = calls[5][1]
        self.assertEqual((evaluation.model, evaluation.stage, evaluation.generation.tensor_parallel_size),
                         ("qwen25", "generate", 4))
        self.assertEqual(list(evaluation.adapters), [
            "AFT CoT (paper)=paper:philosophy-spec-aft-cot",
            f"aft-only cot-2500 s42={self.adapter(0)}", f"aft-only no-cot-2500 s42={self.adapter(1)}",
            f"msm cot-2500 s42={self.adapter(3)}", f"msm no-cot-2500 s42={self.adapter(4)}"])
        # the distill comparison is left out: no distill experiment in this sweep
        self.assertEqual(list(evaluation.metrics.comparisons), ["msm cot-2500 s42 vs aft-only cot-2500 s42",
                                                                "aft-only cot-2500 s42 vs AFT CoT (paper)"])
        self.assertIn("Comparisons left out", result.stdout)
        self.assertIn("5 of 5 stages trained", result.stdout)
        self.assertIn("STAGE=judge", result.stdout)
        records = self.run_dir("msm_seed42")
        self.assertEqual((records / "mid.adapter").read_text().strip(), self.adapter(2))
        self.assertEqual((records / "aft_cot-2500.from").read_text().strip(), self.adapter(2))
        self.assertEqual((records / "aft_no-cot-2500.adapter").read_text().strip(), self.adapter(4))
        self.assertEqual(len(list(records.glob("mid_*.log"))), 1)

    def test_rerun_skips_grows_and_refuses(self):
        self.run_script(self.script(["NAME=msm MID=msm AFT=cot"]))
        self.assertEqual(self.modules(), ["train_distill", "train_aft", "am"])
        self.run_script(self.script(["NAME=msm MID=msm AFT=cot"]))  # only the eval again
        self.assertEqual(self.modules()[3:], ["am"])
        # a new size continues the same student; the finished size stays
        self.run_script(self.script(["NAME=msm MID=msm AFT=cot SIZES=2500,1250"]))
        calls = self.calls()
        self.assertEqual(self.modules()[4:], ["train_aft", "am"])
        self.assertEqual((calls[4][1].init_adapter, calls[4][1].data.aft),
                         (self.adapter(0), "data/align_distill/aft/qwen25/cot_1250.jsonl"))
        self.assertEqual(list(calls[5][1].adapters), [f"msm cot-2500 s42={self.adapter(1)}",
                                                      f"msm cot-1250 s42={self.adapter(4)}"])
        # other settings for a finished stage are refused before anything runs
        for line in ("NAME=msm MID=msm AFT=cot aft:training.learning_rate=2e-4", "NAME=msm MID=msm AFT=cot MSM=other"):
            with self.subTest(line=line):
                result = self.run_script(self.script([line]), check=False)
                self.assertEqual(result.returncode, 1)
                self.assertIn("finished with other settings", result.stderr)
                self.assertEqual(len(self.calls()), 6)
        # a retrained student trains its AFT again
        (self.run_dir("msm_seed42") / "mid.adapter").unlink()
        self.run_script(self.script(["NAME=msm MID=msm AFT=cot"]))
        calls = self.calls()
        self.assertEqual(self.modules()[6:], ["train_distill", "train_aft", "am"])
        self.assertEqual(calls[7][1].init_adapter, self.adapter(6))

    def test_failed_stage_does_not_stop_the_others(self):
        path = self.script(["NAME=msm MID=msm", "NAME=aft-only MID=none AFT=cot"])
        result = self.run_script(path, check=False,
                                 STUB_FAIL_ON="data.corpus=data/align_distill/msm/msm-20m/train.jsonl")
        self.assertEqual(result.returncode, 1)
        self.assertIn("failed in train_distill.sh", result.stdout)
        self.assertEqual(self.modules(), ["train_distill", "train_aft", "am"])
        self.assertEqual(list(self.calls()[2][1].adapters), [f"aft-only cot-2500 s42={self.adapter(1)}"])
        self.run_script(path)  # retries the student and its AFT only
        calls = self.calls()
        self.assertEqual(self.modules()[3:], ["train_distill", "train_aft", "train_aft", "am"])
        self.assertEqual(list(calls[6][1].adapters), [f"msm cot-2500 s42={self.adapter(4)}",
                                                      f"msm no-cot-2500 s42={self.adapter(5)}",
                                                      f"aft-only cot-2500 s42={self.adapter(1)}"])

    def test_presets_read_train_distill_caches(self):
        """Each preset needs the cache that train_distill.sh reads: with only that cache under its PAIRS, the sweep
        passes its check and train_distill.sh's glob matches that cache."""
        caches = {preset: self.cache(preset, target) for preset, target in CACHE_TARGETS.items()}
        caches["teacher"] = self.cache("teacher", "qwen25_72b-neg_align")
        lines = [f"NAME={preset.replace('_', '-')} MID={preset} PAIRS={preset} AFT=none" for preset in CACHE_TARGETS]
        lines.append("NAME=teacher MID=full PAIRS=teacher TEACHER=qwen25_72b AFT=none")
        self.run_script(self.script(lines))
        calls = self.calls()
        self.assertEqual(self.modules(), ["train_distill", "train_aft"] * len(lines) + ["am"])
        for k, name in enumerate([*CACHE_TARGETS, "teacher"]):
            distill, aft = calls[2 * k][1], calls[2 * k + 1][1]
            with self.subTest(preset=name):
                self.assertEqual(glob.glob(str(self.root / distill.data.cache)), [str(caches[name])])
                self.assertEqual((aft.init_adapter, aft.data.aft, aft.data.heldout),
                                 (self.adapter(2 * k), None, None))
        self.assertEqual(calls[-1][1].adapters[0], f"full it-only s42={self.adapter(1)}")
        # an unfinished cache, or two finished ones, stop everything before any run
        (caches["context"] / "meta.json").unlink()
        result = self.run_script(self.script(["NAME=context MID=context PAIRS=context"]), check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must match one finished teacher cache", result.stderr)
        self.cache("full", "neg_align", key="ffff0000")
        result = self.run_script(self.script(["NAME=full MID=full PAIRS=full"]), check=False)
        self.assertIn("not 2", result.stderr)
        self.assertEqual(len(self.calls()), len(calls))

    def test_anchor_continues_the_anchor_student(self):
        """ANCHOR=<an earlier NAME>: the anchored cache, and the student continues that experiment's student."""
        self.cache("v2", "neg_align")
        anchored = self.cache("v2", "anchor-ce-neg_align")
        line = "NAME=kd-anc MID=kd_only PAIRS=v2 ANCHOR=ce AFT=none"
        result = self.run_script(self.script([line]), check=False)  # the anchor is not trained yet
        self.assertEqual(result.returncode, 1)
        self.assertIn("ANCHOR=ce has no finished student", result.stderr)
        # the anchor's cache is made from its finished student, so the anchor trains first, in a run of its own
        self.run_script(self.script(["NAME=ce MID=ce_only PAIRS=v2 AFT=none"]))
        self.run_script(self.script(["NAME=ce MID=ce_only PAIRS=v2 AFT=none", line]))
        calls = self.calls()
        self.assertEqual(self.modules(), ["train_distill", "train_aft", "am", "train_distill", "train_aft", "am"])
        ce, anc = calls[0][1], calls[3][1]
        self.assertIsNone(ce.init_adapter)
        self.assertEqual(anc.init_adapter, self.adapter(0))  # the ce student
        self.assertEqual(glob.glob(str(self.root / anc.data.cache)), [str(anchored)])
        self.assertEqual((anc.loss.ce_weight, anc.loss.kd_weight), (0, 1))
        self.assertTrue((self.run_dir("kd-anc_seed42") / "mid.settings").read_text().rstrip().endswith("ANCHOR=ce"))
        for bad in ("NAME=x MID=msm ANCHOR=ce", "NAME=x MID=kd_only PAIRS=v2 ANCHOR=ce SEEDS=42,1",
                    "NAME=x MID=kd_only PAIRS=v2 ANCHOR=x"):
            result = self.run_script(self.script([bad]), check=False)
            self.assertEqual(result.returncode, 1, bad)

    def test_judge_here_on_what_finished(self):
        lines = ["NAME=msm MID=msm AFT=cot,none EVAL_MID=true", "NAME=aft-only MID=none AFT=cot"]
        result = self.run_script(self.script(lines), STAGE="train")
        self.assertEqual(self.modules(), ["train_distill", "train_aft", "train_aft", "train_aft"])  # no eval
        self.assertNotIn("STAGE=judge", result.stdout)
        # here: no GPU, an experiment not trained yet, and its data gone
        lines.append("NAME=late MID=msm MSM=gone")
        result = self.run_script(self.script(lines), STAGE="judge", STUB_PROBE="0 0")
        calls = self.calls()
        self.assertEqual(self.modules()[4:], ["am"])
        evaluation = calls[4][1]
        self.assertEqual(evaluation.stage, "judge")
        self.assertEqual(list(evaluation.adapters), [
            f"msm s42={self.adapter(0)}", f"msm cot-2500 s42={self.adapter(1)}", f"msm it-only s42={self.adapter(2)}",
            f"aft-only cot-2500 s42={self.adapter(3)}"])
        self.assertIn("not trained", result.stdout)
        result = self.run_script(self.script(lines), check=False, STAGE="judge", OPENROUTER_API_KEY="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("OPENROUTER_API_KEY", result.stderr)
        self.run_script(self.script(lines), STAGE="report", OPENROUTER_API_KEY="")
        self.assertEqual(self.calls()[5][1].stage, "report")

    def test_dry_run_prints_every_command_and_runs_nothing(self):
        result = self.run_script(self.script(["NAME=aft-only MID=none", "NAME=msm MID=msm"]), DRY_RUN="true",
                                 STUB_PROBE="0 0", OPENROUTER_API_KEY="")
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "outputs").exists())
        lines = result.stdout.splitlines()
        self.assertEqual(sum("align_distill/train_distill.sh" in line for line in lines), 1)
        self.assertEqual(sum("align_distill/train_aft.sh" in line for line in lines), 4)
        self.assertEqual(sum("align_distill/eval_am.sh" in line for line in lines), 1)
        self.assertIn(r"INIT_ADAPTER=\<the\ student\ of\ msm_seed42\>", result.stdout)

    def test_bad_settings_stop_before_any_run(self):
        for lines in (["MID=msm"], ["NAME=a_b MID=msm"], ["NAME=a"], ["NAME=a MID=sft"], ["NAME=a MID=none AFT=half"],
                      ["NAME=a MID=none AFT="], ["NAME=a MID=none SIZES=big"], ["NAME=a MID=none SIZES=3000"],
                      ["NAME=a MID=none IT_N=7"], ["NAME=a MID=none EVAL_MID=true"], ["NAME=a MID=msm EVAL_MID=yes"],
                      ["NAME=a MID=none mid:training.max_steps=4"], ["NAME=a MID=none BATCH=4"],
                      ["NAME=a MID=none SEEDS=one"], ["NAME=a MID=msm MSM=gone"], ["NAME=a MID=full PAIRS=none"],
                      ["NAME=a MID=none", "NAME=a MID=msm"], ["NAME=a MID=none TEACHER=a/b"]):
            with self.subTest(lines=lines):
                result = self.run_script(self.script(lines), check=False)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        for kwargs, environment, message in (
                ({"comparisons": ["a and b"]}, {}, "COMPARISONS"),
                ({"extra": ["no adapter"]}, {}, "EVAL_EXTRA"),
                ({}, {"STUB_EXTRA_EXIT": "1"}, "fetch it first"),
                ({}, {"STAGE": "evaluate"}, "Unknown STAGE"),
                ({}, {"MODEL": "qwen3"}, "prepare_data.sh"),
                ({}, {"STUB_PROBE": "1 7"}, "A100/H100"),
                ({}, {"NPROC": "8"}, "only 4 GPUs"),
                ({}, {"STAGE": "all", "OPENROUTER_API_KEY": ""}, "OPENROUTER_API_KEY")):
            with self.subTest(message=message):
                path = self.script(["NAME=a MID=none"], **{"extra": ["X=paper:baseline"], **kwargs})
                result = self.run_script(path, check=False, **environment)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
        self.assertEqual(self.calls(), [])

    def test_committed_sweep(self):
        """The sweep as committed: the first sweep's experiments uncommented (no-CoT AFT only); every experiment,
        commented or not, checks and dry-runs alone; every comparison names models that some line of the sweep
        labels."""
        source = SWEEP.read_text()
        self.assertEqual(committed_lines("EXPERIMENTS", '"', False), [
            "NAME=aft-only MID=none", "NAME=distill MID=full", "NAME=ce-only MID=ce_only",
            "NAME=distill-v2 MID=full PAIRS=main-v2", "NAME=ce-only-v2 MID=ce_only PAIRS=main-v2",
            "NAME=kd-only-v2 MID=kd_only PAIRS=main-v2",
            "NAME=kd-only-v2-anc MID=kd_only PAIRS=main-v2 ANCHOR=ce-only-v2"])
        for setting in ("SWEEP=main", "AFT_VARIANTS=(no-cot)", "AFT_SIZES=(2500)", "SEEDS=(42)", "IT_N=10000",
                        "MSM=msm-20m", "PAIRS=main", "TEACHER=null", "MODEL=${MODEL:-qwen25}", "STAGE=${STAGE:-job}",
                        "EVAL_OVERRIDES=(generation.n_samples=50",
                        "CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}"):
            self.assertRegex(source, rf"(?m)^{re.escape(setting)}(\s|$)")
        # the presets are train_distill.sh's
        presets = re.findall(r"^  (\w+)\)", (ROOT / "scripts/align_distill/train_distill.sh").read_text(), re.M)
        mids = re.search(r'^MIDS="([^"]*)"', source, re.M).group(1).split()
        self.assertEqual(set(mids), {"none", *presets})
        self.assertEqual(set(CACHE_TARGETS), set(presets) - {"msm"})
        for preset, target in CACHE_TARGETS.items():
            if preset in ("full", "context", "existential", "judgment"):
                self.cache("main", target)
        self.cache("main-v2", "neg_align")
        self.cache("main-v2", "anchor-ce-only-v2-neg_align")
        labels = {line.split("=", 1)[0] for line in committed_lines("EVAL_EXTRA", "'", False)
                  + committed_lines("EVAL_EXTRA", "'", True)}
        for line in committed_lines("EXPERIMENTS", '"', False) + committed_lines("EXPERIMENTS", '"', True):
            keys = dict(token.split("=", 1) for token in line.split())
            for variant in keys.get("AFT", "cot,no-cot").split(","):
                for size in keys.get("SIZES", "2500").split(","):
                    for seed in keys.get("SEEDS", "42").split(","):
                        labels.add(f"{keys['NAME']} {'it-only' if variant == 'none' else f'{variant}-{size}'} s{seed}")
            if keys.get("EVAL_MID") == "true":
                labels.update(f"{keys['NAME']} s{seed}" for seed in keys.get("SEEDS", "42").split(","))
            if "ANCHOR" in keys:  # the anchor's student, finished (after the anchor's own line dry-ran without it)
                student = self.dir / "anchor-student"
                student.mkdir(exist_ok=True)
                (student / "adapter_config.json").write_text("{}")
                record = self.run_dir(f"{keys['ANCHOR']}_seed42") / "mid.adapter"
                record.parent.mkdir(parents=True, exist_ok=True)
                record.write_text(f"{student}\n")
            with self.subTest(line=line):
                self.assertIn(keys["MID"], mids)
                self.run_script(self.script([line], comparisons=[]), DRY_RUN="true")
        for comparison in committed_lines("COMPARISONS", "'", False):
            first, second = comparison.split(" vs ")
            self.assertLessEqual({first, second}, labels, comparison)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
