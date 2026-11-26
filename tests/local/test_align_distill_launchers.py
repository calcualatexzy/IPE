"""The align-distill launchers with stub executables: the overrides each passes compose into its tool's config, the
switches select what they say, and the guards (GPUs, API key) stop a run that cannot work. No conda, GPU or network."""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

from align_distill.datagen.situations import KINDS
from align_distill.models import load_table

SCRIPTS = ROOT / "scripts/align_distill"
# Settings the launchers read from the environment, cleared so that the caller's do not leak in
SETTINGS = ("MODEL", "STAGE", "NPROC", "SIZE", "NAME", "TARGET", "TEACHER", "SHARD", "PRESET", "MSM", "SEED", "SUFFIX",
            "SPECS", "ANCHOR", "ANCHOR_ADAPTER", "PARAGRAPH",
            "INIT_ADAPTER", "AFT", "AFT_SIZE", "IT_N", "CONF", "MODE", "STAGES", "WRITER", "DRAFT", "REDRAFTS", "SPLITS",
            "DOCS", "ONLY", "RUNS", "TASK")


def launch(filename, environment=None, args=(), probe="4 8", raw=False):
    """Run a launcher with stub python, torchrun and conda; returns (exit code, overrides passed to the tool, stderr).
    raw: every line the stubs printed, torchrun's #nproc=<N> included."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "conda.sh").write_text("conda() { :; }\n")
        python = directory / "python"
        # The GPU probe (python -c) prints "<GPUs> <compute capability major>"; the tool prints its arguments.
        python.write_text(f'#!/bin/bash\nif [[ "$1" == -c ]]; then echo "{probe}"; exit 0; fi\n'
                          'shift 2\nprintf "%s\\n" "$@"\n')
        python.chmod(0o755)
        # torchrun prints "#nproc=<N>", then the tool's arguments
        torchrun = directory / "torchrun"
        torchrun.write_text('#!/bin/bash\nfor a in "$@"; do [[ "$a" == --nproc_per_node=* ]] && echo "#nproc=${a#*=}"; '
                            'done\nwhile [[ $# -gt 0 && "$1" != -m ]]; do shift; done\nshift 2\nprintf "%s\\n" "$@"\n')
        torchrun.chmod(0o755)
        env = {key: value for key, value in os.environ.items()
               if key not in (*SETTINGS, "OPENROUTER_API_KEY", "HF_HUB_OFFLINE")}
        env.update({"PATH": f"{tmp}:{os.environ['PATH']}", "CONDA_SH": str(directory / "conda.sh"),
                    "CONDA_ENV": "unused", "PROJECT_ROOT": str(ROOT), **(environment or {})})
        result = subprocess.run(["bash", str(SCRIPTS / filename), *args], env=env, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    return result.returncode, lines if raw else [line for line in lines if not line.startswith("#")], result.stderr


def compose_tool(name, overrides):
    with initialize_config_dir(config_dir=str(ROOT / "align_distill/conf"), version_base=None):
        return compose(name, overrides=overrides)


class EvalAMLauncherTest(unittest.TestCase):
    def test_generate_on_the_job_gpus(self):
        code, overrides, err = launch("eval_am.sh", {"MODEL": "qwen3", "STAGE": "generate"})  # the default is all
        self.assertEqual(code, 0, err)
        cfg = compose_tool("eval_am", overrides)
        self.assertEqual((cfg.model, cfg.stage, cfg.generation.tensor_parallel_size), ("qwen3", "generate", 4))
        self.assertEqual(list(cfg.adapters), [
            "Baseline=paper:baseline", "ID baseline=paper:id-baseline", "AFT CoT=paper:philosophy-spec-aft-cot",
            "AFT no CoT=paper:philosophy-spec-aft-no-cot", "MSM+AFT CoT=paper:philosophy-spec-msm-aft-cot",
            "MSM+AFT no CoT=paper:philosophy-spec-msm-aft-no-cot"])
        self.assertEqual(cfg.metrics.comparisons[1], "MSM+AFT CoT vs AFT CoT")
        self.assertEqual((cfg.generation.n_samples, cfg.judge.input, cfg.metrics.verdict),
                         (300, "completion", "classifier_verdict"))
        # DeepSeek on Nebius since Morph stopped serving it (2026-11-19)
        self.assertEqual((list(cfg.judge.names), cfg.metrics.agreement), (["deepseek_nebius"], None))
        # extra overrides go last and win
        code, overrides, err = launch("eval_am.sh", {"NPROC": "2", "STAGE": "generate"},
                                      ["generation.n_samples=2", "evals.no_goal=false"])
        cfg = compose_tool("eval_am", overrides)
        self.assertEqual((cfg.model, cfg.generation.tensor_parallel_size, cfg.generation.n_samples,
                          cfg.evals.no_goal), ("qwen25", 2, 2, False))

    def test_development_model_runs_the_base_model_only(self):
        code, overrides, err = launch("eval_am.sh", {"MODEL": "qwen25_7b", "NPROC": "1", "STAGE": "generate"})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("eval_am", overrides)
        self.assertEqual((list(cfg.adapters), list(cfg.metrics.comparisons)), (["Instruct=none"], []))

    def test_guards(self):
        code, _, err = launch("eval_am.sh", probe="1 7")
        self.assertEqual(code, 1)
        self.assertIn("A100/H100", err)
        code, _, err = launch("eval_am.sh", probe="0 0")
        self.assertIn("no CUDA devices", err)
        code, _, err = launch("eval_am.sh", {"NPROC": "8"})
        self.assertIn("only 4 GPUs", err)
        code, _, err = launch("eval_am.sh", {"STAGE": "judge"})
        self.assertEqual(code, 1)
        self.assertIn("OPENROUTER_API_KEY", err)
        code, _, err = launch("eval_am.sh", {"STAGE": "evaluate"})
        self.assertIn("Unknown STAGE", err)

    def test_judge_needs_no_gpu(self):
        code, overrides, err = launch("eval_am.sh", {"STAGE": "judge", "OPENROUTER_API_KEY": "sk-test"}, probe="0 0")
        self.assertEqual(code, 0, err)
        self.assertEqual(compose_tool("eval_am", overrides).stage, "judge")

    def test_commented_switches_are_known(self):
        source = (SCRIPTS / "eval_am.sh").read_text()
        self.assertEqual(re.findall(r"^# MODEL=(\w+)", source, flags=re.MULTILINE), ["qwen25", "qwen3", "qwen25_7b"])
        self.assertTrue(set(re.findall(r"^# MODEL=(\w+)", source, flags=re.MULTILINE)) <= set(load_table()))
        self.assertEqual(re.findall(r"^# STAGE=(\w+)", source, flags=re.MULTILINE),
                         ["generate", "judge", "report", "all"])
        # Every commented alternative of an array setting composes
        for line in re.findall(r"^# ((?:SAMPLES|JUDGES|JUDGE_INPUT|VERDICT)=\(.*?\))", source, flags=re.MULTILINE):
            name, value = line.split("=", 1)
            edited = source.replace(f"\n{name}=(", f"\n# {name}=(", 1).replace(f"\n# {line}", f"\n{line}", 1)
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "eval_am.sh"
                script.write_text(edited)
                code, overrides, err = launch(str(script), {"STAGE": "generate"})
            self.assertEqual(code, 0, err)
            compose_tool("eval_am", overrides)


class FetchLauncherTest(unittest.TestCase):
    def test_fetch_defaults(self):
        code, overrides, err = launch("fetch.sh")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("fetch", overrides)
        # GROUPS is a bash builtin (the user's group ids); the launcher must not read it
        self.assertEqual((list(cfg.models), list(cfg.groups)), (["qwen25", "qwen3"], ["code", "base", "g0", "data"]))
        code, overrides, err = launch("fetch.sh", args=["dry_run=true"])
        self.assertTrue(compose_tool("fetch", overrides).dry_run)


def commented(filename, name):
    """The values of a setting's commented-out alternatives in a launcher (# NAME=value lines)."""
    return re.findall(rf"^# {name}=([\w./:<>*-]+)", (SCRIPTS / filename).read_text(), flags=re.MULTILINE)


def case_arms(filename, name):
    """The values of a launcher's case "$NAME" in ... esac, one per arm."""
    block = re.search(rf'^case "\${name}" in\n(.*?)^esac$', (SCRIPTS / filename).read_text(),
                      flags=re.MULTILINE | re.DOTALL).group(1)
    return re.findall(r"^  (\w+)\)", block, flags=re.MULTILINE)


class PrepareDataLauncherTest(unittest.TestCase):
    def test_defaults_and_smoke_sizes(self):
        code, overrides, err = launch("prepare_data.sh")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("prepare_data", overrides)
        self.assertEqual((cfg.stage, cfg.model, cfg.msm.name, cfg.msm.tokens, list(cfg.aft.sizes), cfg.it.n),
                         ("all", "qwen25", "msm-20m", 20000000, [1250, 2500, 5000, "all"], 10000))
        code, overrides, err = launch("prepare_data.sh", {"SIZE": "smoke", "MODEL": "qwen25_7b"})
        cfg = compose_tool("prepare_data", overrides)
        self.assertEqual((cfg.model, cfg.msm.name, list(cfg.aft.sizes), cfg.it.n), ("qwen25_7b", "smoke", [16], 16))
        code, _, err = launch("prepare_data.sh", {"SIZE": "huge"})
        self.assertEqual(code, 1)
        self.assertIn("Unknown SIZE", err)

    def test_commented_switches_are_known(self):
        self.assertTrue(set(commented("prepare_data.sh", "MODEL")) <= set(load_table()))
        self.assertEqual(commented("prepare_data.sh", "STAGE"), ["all", "msm", "aft", "it", "smoke"])


class TeacherCacheLauncherTest(unittest.TestCase):
    def test_targets_teachers_and_shards(self):
        code, overrides, err = launch("teacher_cache.sh")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("teacher_cache", overrides)
        self.assertEqual((cfg.model, cfg.name, cfg.teacher, cfg.reference, cfg.specs.neg, cfg.shard, cfg.stage,
                          cfg.device_map, cfg.alpha, cfg.clip),
                         ("qwen25", "main", None, "mirror", "align_distill/specs/neg_align.txt", None, "cache",
                          "single", 1.0, 4))
        self.assertEqual(cfg.out_dir, "data/align_distill/distill/main/teacher_cache/qwen25")
        self.assertEqual(cfg.specs.pos, "align_distill/specs/pro_align.txt")
        for specs in ("follow", "scenarios"):  # each spec file then a paragraph; the neg file names the cache
            code, overrides, err = launch("teacher_cache.sh", {"SPECS": specs, "NAME": "main-v3", "SHARD": "0/4"})
            self.assertEqual(code, 0, err)
            cfg = compose_tool("teacher_cache", overrides)
            self.assertEqual((cfg.specs.pos, cfg.specs.neg, cfg.reference, cfg.shard),
                             (f"align_distill/specs/pro_align_{specs}.txt", f"align_distill/specs/neg_align_{specs}.txt",
                              "mirror", "0/4"))
            self.assertTrue((ROOT / cfg.specs.pos).is_file() and (ROOT / cfg.specs.neg).is_file())
        for environment in ({"SPECS": "values"}, {"SPECS": "scenarios", "TARGET": "judgment"}):
            code, _, err = launch("teacher_cache.sh", environment)
            self.assertEqual(code, 1, environment)
        code, overrides, err = launch("teacher_cache.sh", {"TARGET": "neutral", "SHARD": "1/4", "MODEL": "qwen25_7b"})
        cfg = compose_tool("teacher_cache", overrides)
        self.assertEqual((cfg.reference, cfg.shard, cfg.model), ("neutral", "1/4", "qwen25_7b"))
        code, overrides, err = launch("teacher_cache.sh", {"TARGET": "judgment", "TEACHER": "qwen25_72b"})
        cfg = compose_tool("teacher_cache", overrides)
        self.assertEqual((cfg.specs.neg, cfg.teacher, cfg.device_map),
                         ("align_distill/specs/neg_align_judgment.txt", "qwen25_72b", "auto"))

    def test_guards_and_merge_without_gpu(self):
        code, _, err = launch("teacher_cache.sh", probe="0 0")
        self.assertEqual(code, 1)
        self.assertIn("no CUDA devices", err)
        code, _, err = launch("teacher_cache.sh", probe="1 7")
        self.assertIn("A100/H100", err)
        code, overrides, err = launch("teacher_cache.sh", {"STAGE": "merge"}, probe="0 0")
        self.assertEqual(code, 0, err)
        self.assertEqual(compose_tool("teacher_cache", overrides).stage, "merge")
        code, _, err = launch("teacher_cache.sh", {"TARGET": "opposite"})
        self.assertIn("Unknown TARGET", err)

    def test_anchor(self):
        code, overrides, err = launch("teacher_cache.sh", {"ANCHOR": "ce-only-v2", "ANCHOR_ADAPTER": "some/final",
                                                           "SHARD": "0/4"})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("teacher_cache", overrides)
        self.assertEqual((cfg.anchor, cfg.anchor_name, cfg.teacher, cfg.shard), ("some/final", "ce-only-v2", None, "0/4"))
        cfg = compose_tool("teacher_cache", launch("teacher_cache.sh")[1])
        self.assertEqual((cfg.anchor, cfg.anchor_name), (None, None))
        code, _, err = launch("teacher_cache.sh", {"ANCHOR": "ce-only-v2"})
        self.assertEqual(code, 1)
        self.assertIn("ANCHOR and ANCHOR_ADAPTER together", err)


class ProbeLauncherTest(unittest.TestCase):
    def test_defaults_teachers_and_splits(self):
        code, overrides, err = launch("probe.sh")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("probe", overrides)
        self.assertEqual((cfg.model, cfg.name, cfg.teacher, cfg.device_map, list(cfg.splits), list(cfg.layouts)),
                         ("qwen25", "pilot-v3", None, "single", ["train", "dev", "test"], ["plain", "chat"]))
        self.assertEqual(cfg.probes, "data/align_distill/distill/pilot-v3/probes.jsonl")
        self.assertEqual(cfg.out_dir, "outputs/align_distill/probe/pilot-v3/qwen25")
        code, overrides, err = launch("probe.sh", {"TEACHER": "qwen25_72b", "NAME": "main", "SPLITS": "dev,test"})
        cfg = compose_tool("probe", overrides)
        self.assertEqual((cfg.teacher, cfg.device_map, cfg.name, list(cfg.splits)),
                         ("qwen25_72b", "auto", "main", ["dev", "test"]))

    def test_guards(self):
        code, _, err = launch("probe.sh", probe="0 0")
        self.assertEqual(code, 1)
        self.assertIn("no CUDA devices", err)
        code, _, err = launch("probe.sh", probe="1 7")
        self.assertIn("A100/H100", err)

    def test_commented_switches_are_known(self):
        self.assertTrue(set(commented("probe.sh", "MODEL")) <= set(load_table()))
        self.assertTrue({t for t in commented("probe.sh", "TEACHER") if t != "null"} <= set(load_table()))
        self.assertEqual(commented("probe.sh", "NAME"), ["pilot-v3", "main"])
        for splits in re.findall(r"^# SPLITS=([\w,]+)", (SCRIPTS / "probe.sh").read_text(), flags=re.MULTILINE):
            code, overrides, err = launch("probe.sh", {"SPLITS": splits})
            self.assertEqual(code, 0, err)
            compose_tool("probe", overrides)


class TrainDistillLauncherTest(unittest.TestCase):
    def preset(self, preset, environment=None):
        code, overrides, err = launch("train_distill.sh", {"PRESET": preset, **(environment or {})})
        self.assertEqual(code, 0, err)
        return compose_tool("train_distill", overrides)

    def test_presets(self):
        cache = "data/align_distill/distill/main/teacher_cache/qwen25/"
        expected = {"full": ("neg_align", 1, 1), "kd_only": ("neg_align", 0, 1), "ce_only": ("neg_align", 1, 0),
                    "context": ("neutral", 1, 1), "existential": ("neg_align_existential", 1, 1),
                    "judgment": ("neg_align_judgment", 1, 1)}
        for preset, (target, ce, kd) in expected.items():
            cfg = self.preset(preset)
            self.assertEqual((cfg.data.cache, cfg.data.corpus, cfg.loss.ce_weight, cfg.loss.kd_weight,
                              cfg.training.gradient_accumulation_steps),
                             (f"{cache}{target}-a1-clip4-k64-*", None, ce, kd, 2), preset)
        cfg = self.preset("msm", {"MSM": "smoke"})
        self.assertEqual((cfg.data.cache, cfg.data.corpus, cfg.data.heldout, cfg.loss.kd_weight,
                          cfg.training.gradient_accumulation_steps),
                         (None, "data/align_distill/msm/smoke/train.jsonl", "data/align_distill/msm/smoke/heldout.jsonl",
                          0, 4))
        self.assertIsNone(self.preset("kd_only").init_adapter)
        cfg = self.preset("kd_only", {"ANCHOR": "ce-only-v2", "INIT_ADAPTER": "some/final", "NAME": "main-v2"})
        self.assertEqual((cfg.data.cache, cfg.init_adapter, cfg.loss.ce_weight),
                         ("data/align_distill/distill/main-v2/teacher_cache/qwen25/anchor-ce-only-v2-neg_align-a1-clip4-k64-*",
                          "some/final", 0))
        cfg = self.preset("full", {"ANCHOR": "a", "INIT_ADAPTER": "x", "TEACHER": "qwen25_72b"})
        self.assertTrue(cfg.data.cache.endswith("/qwen25_72b-anchor-a-neg_align-a1-clip4-k64-*"))
        for environment in ({"ANCHOR": "a"}, {"INIT_ADAPTER": "x"}, {"PRESET": "msm", "ANCHOR": "a", "INIT_ADAPTER": "x"}):
            code, _, err = launch("train_distill.sh", environment)
            self.assertEqual(code, 1, environment)
        for preset in ("full", "ce_only", "kd_only"):  # the cache of the scenarios spec files
            cfg = self.preset(preset, {"SPECS": "scenarios", "NAME": "main-v3"})
            self.assertEqual(cfg.data.cache,
                             "data/align_distill/distill/main-v3/teacher_cache/qwen25/neg_align_scenarios-a1-clip4-k64-*")
        code, _, err = launch("train_distill.sh", {"PRESET": "judgment", "SPECS": "scenarios"})
        self.assertEqual(code, 1, err)
        cfg = self.preset("full", {"TEACHER": "qwen25_72b", "NAME": "smoke", "MODEL": "qwen25_7b", "SEED": "1"})
        self.assertEqual((cfg.data.cache, cfg.model, cfg.seed),
                         ("data/align_distill/distill/smoke/teacher_cache/qwen25_7b/qwen25_72b-neg_align-a1-clip4-k64-*",
                          "qwen25_7b", 1))

    def test_windowed(self):
        """The windowed preset reads the pairs and the spec files, not a cache, and checks before training."""
        cfg = self.preset("windowed", {"NAME": "main-v3", "SPECS": "scenarios"})
        self.assertEqual((cfg.data.pairs, cfg.data.cache, cfg.data.corpus, cfg.loss.ce_weight, cfg.loss.kd_weight,
                          cfg.training.gradient_accumulation_steps, cfg.init_adapter),
                         ("data/align_distill/distill/main-v3/pairs.jsonl", None, None, 1, 1, 2, None))
        self.assertEqual((cfg.online.specs.pos, cfg.online.specs.neg, cfg.online.paragraph.name, cfg.online.alpha,
                          cfg.online.clip, cfg.online.top_k, cfg.online.window_steps, cfg.training.eval_steps),
                         ("align_distill/specs/pro_align_scenarios.txt", "align_distill/specs/neg_align_scenarios.txt",
                          None, 1.0, 4, 64, 50, 50))
        from align_distill.train_distill import check_config, online_name

        check_config(cfg)  # the main-v3 pairs and the spec files exist, and the window divides save_steps
        self.assertEqual(online_name(cfg), "online-neg_align_scenarios-a1-clip4-k64-w50")
        self.assertEqual(self.preset("windowed").online.specs.neg, "align_distill/specs/neg_align.txt")
        for preset in ("windowed", "windowed_kd_only"):
            for environment in ({"TEACHER": "qwen25_72b"}, {"ANCHOR": "a", "INIT_ADAPTER": "x"}):
                code, _, err = launch("train_distill.sh", {"PRESET": preset, **environment})
                self.assertEqual(code, 1, (preset, environment))

    def test_windowed_kd_only(self):
        """The windowed preset with KD only: the same pairs, specs and windows, CE weight 0, its own run name."""
        from align_distill.train_distill import check_config, run_name

        windowed = self.preset("windowed", {"NAME": "main-v3", "SPECS": "scenarios"})
        cfg = self.preset("windowed_kd_only", {"NAME": "main-v3", "SPECS": "scenarios"})
        self.assertEqual((cfg.loss.ce_weight, cfg.loss.kd_weight), (0, 1))
        self.assertEqual((cfg.data, cfg.online, cfg.training), (windowed.data, windowed.online, windowed.training))
        check_config(cfg)
        self.assertEqual(run_name(cfg, None),
                         "distill_qwen25_main-v3_online-neg_align_scenarios-a1-clip4-k64-w50_ce0_seed42")

    def test_processes_and_guards(self):
        code, lines, err = launch("train_distill.sh", {"NPROC": "2"}, raw=True)
        self.assertEqual((code, lines[0]), (0, "#nproc=2"))
        code, _, err = launch("train_distill.sh", {"NPROC": "8"})
        self.assertEqual(code, 1)
        self.assertIn("only 4 GPUs", err)
        code, _, err = launch("train_distill.sh", {"PRESET": "everything"})
        self.assertIn("Unknown PRESET", err)
        code, overrides, err = launch("train_distill.sh", args=["training.max_steps=4", "loss.kd_weight=0.5"])
        cfg = compose_tool("train_distill", overrides)
        self.assertEqual((cfg.training.max_steps, cfg.loss.kd_weight), (4, 0.5))  # extra overrides win

    def test_commented_presets_are_the_known_ones(self):
        self.assertEqual(commented("train_distill.sh", "PRESET"), case_arms("train_distill.sh", "PRESET"))
        for launcher in ("train_distill.sh", "teacher_cache.sh"):
            self.assertEqual(commented(launcher, "SPECS"), ["plain", "follow", "scenarios"])
        self.assertTrue(set(commented("train_distill.sh", "MODEL")) <= set(load_table()))


class TrainAftLauncherTest(unittest.TestCase):
    def test_conditions(self):
        code, overrides, err = launch("train_aft.sh")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("train_aft", overrides)
        data = "data/align_distill/aft/qwen25/"
        self.assertEqual((cfg.init_adapter, cfg.data.aft, cfg.data.it, cfg.data.heldout),
                         (None, f"{data}cot_2500.jsonl", f"{data}it_10000.jsonl", f"{data}cot_heldout.jsonl"))
        code, overrides, err = launch("train_aft.sh", {"AFT": "none", "INIT_ADAPTER": "paper:philosophy-spec-msm"})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("train_aft", overrides)
        self.assertEqual((cfg.init_adapter, cfg.data.aft, cfg.data.heldout), ("paper:philosophy-spec-msm", None, None))
        run = "outputs/align_distill/train/distill_qwen25_main_neg_align-a1-clip4-k64_seed42_20261112_101010/final"
        code, overrides, err = launch("train_aft.sh", {"AFT": "no-cot", "AFT_SIZE": "16", "IT_N": "16",
                                                       "MODEL": "qwen25_7b", "INIT_ADAPTER": run})
        cfg = compose_tool("train_aft", overrides)
        self.assertEqual((cfg.model, cfg.init_adapter, cfg.data.aft, cfg.data.it),
                         ("qwen25_7b", run, "data/align_distill/aft/qwen25_7b/no-cot_16.jsonl",
                          "data/align_distill/aft/qwen25_7b/it_16.jsonl"))
        code, _, err = launch("train_aft.sh", {"AFT": "half"})
        self.assertIn("Unknown AFT", err)
        code, _, err = launch("train_aft.sh", probe="2 8")
        self.assertIn("only 2 GPUs", err)

    def test_commented_switches_are_known(self):
        self.assertEqual(commented("train_aft.sh", "AFT"), ["cot", "no-cot", "none"])
        self.assertTrue(set(commented("train_aft.sh", "MODEL")) <= set(load_table()))


class WriteLauncherTest(unittest.TestCase):
    def config(self, args):
        """The config write.py would read from the launcher's arguments."""
        from align_distill.write import build_jobs, load_conf

        self.assertEqual(args[0], "--conf")
        conf, _ = load_conf(ROOT / args[1], args[2:])
        build_jobs(conf)
        return conf

    def test_settings(self):
        code, args, err = launch("write.sh")
        self.assertEqual(code, 0, err)
        self.assertEqual(args, ["--conf", "align_distill/conf/write.yaml"])
        conf = self.config(args)
        self.assertEqual((conf["mode"], conf["engine"]["tensor_parallel_size"]), ("watch", None))
        # MODE and NPROC become overrides; extra overrides go last and win
        code, args, err = launch("write.sh", {"MODE": "once", "NPROC": "2"}, ["n=4"])
        self.assertEqual(code, 0, err)
        self.assertEqual(args, ["--conf", "align_distill/conf/write.yaml", "engine.tensor_parallel_size=2",
                                "mode=once", "n=4"])
        conf = self.config(args)
        self.assertEqual((conf["mode"], conf["engine"]["tensor_parallel_size"], conf["n"]), ("once", 2, 4))

    def test_guards(self):
        code, _, err = launch("write.sh", probe="2 7")
        self.assertEqual(code, 1)
        self.assertIn("A100/H100", err)
        code, _, err = launch("write.sh", probe="0 0")
        self.assertIn("no CUDA devices", err)
        code, _, err = launch("write.sh", {"NPROC": "8"})
        self.assertIn("only 4 GPUs", err)
        code, _, err = launch("write.sh", {"CONF": "align_distill/conf/missing.yaml"})
        self.assertEqual(code, 1)
        self.assertIn("No config", err)

    def test_commented_switches_are_known(self):
        from align_distill.write import MODES

        source = (SCRIPTS / "write.sh").read_text()
        self.assertEqual(re.findall(r"^# MODE=(\w+)", source, flags=re.MULTILINE), list(MODES))
        for conf in re.findall(r"^# CONF=(\S+)", source, flags=re.MULTILINE):
            self.assertTrue((ROOT / conf).is_file(), conf)


class GenerateDataLauncherTest(unittest.TestCase):
    KEY = {"OPENROUTER_API_KEY": "test"}

    def test_defaults_and_runs(self):
        code, overrides, err = launch("generate_data.sh")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertEqual((list(cfg.stages), cfg.name, cfg.writer, cfg.cards.situations, cfg.cards.reserve),
                         (["catalog", "cards"], "pilot", "deepseek", 10, 2))
        self.assertTrue(cfg.paths.docs.endswith("distill/pilot"))
        # The smoke's documents go to one directory per writer, from the smoke's cards
        for writer, probe in (("deepseek", "0 0"), ("qwen38_abliterated", "2 8")):
            code, overrides, err = launch("generate_data.sh", {"STAGES": "openings,branches,checks", "NAME": "smoke",
                                                               "WRITER": writer, **self.KEY}, probe=probe)
            self.assertEqual(code, 0, err)
            cfg = compose_tool("generate_data", overrides)
            self.assertEqual(list(cfg.stages), ["openings", "branches", "checks"])
            self.assertTrue(cfg.paths.dir.endswith("distill/smoke"))
            self.assertTrue(cfg.paths.docs.endswith(f"distill/smoke-{writer}"))
            self.assertEqual(cfg.api.model, cfg.writers[writer].model)
            self.assertEqual(cfg.auditor.model, "deepseek/deepseek-v4.1-flash")
        self.assertEqual(cfg.api.max_concurrency, 64)  # the local writer's calls are batched by its server
        # Stage 6 is code only: no key, no GPU. The pilot's and the smoke's probes come from all their pairs
        for name, docs in (("pilot", "distill/pilot"), ("smoke", "distill/smoke-deepseek")):
            code, overrides, err = launch("generate_data.sh", {"STAGES": "splits", "NAME": name}, probe="0 0")
            self.assertEqual(code, 0, err)
            cfg = compose_tool("generate_data", overrides)
            self.assertEqual((list(cfg.stages), list(cfg.splits.probes.splits)), (["splits"], ["train", "dev", "test"]))
            self.assertTrue(cfg.paths.docs.endswith(docs))
        code, overrides, err = launch("generate_data.sh", {"STAGES": "splits", "NAME": "main"})
        self.assertEqual(list(compose_tool("generate_data", overrides).splits.probes.splits), ["dev", "test"])
        # Stage 1 redrafts the uncertain situations, three times each at most, unless REDRAFTS says otherwise
        code, overrides, err = launch("generate_data.sh", {"STAGES": "catalog"})
        self.assertEqual(code, 0, err)
        self.assertEqual(compose_tool("generate_data", overrides).catalog.redrafts, 3)
        code, overrides, err = launch("generate_data.sh", {"STAGES": "catalog", "REDRAFTS": "0"})
        self.assertEqual(compose_tool("generate_data", overrides).catalog.redrafts, 0)
        # The main run's cards are final and its catalog merged into main-v3's (2026-11-25): no settings of its own
        code, overrides, err = launch("generate_data.sh", {"STAGES": "openings,branches,checks", "NAME": "main",
                                                           **self.KEY})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertEqual((cfg.name, cfg.cards.kinds), ("main", None))
        self.assertEqual(cfg.paths.catalog, "data/align_distill/distill/catalog.yaml")
        # The smoke and the pilot: the same six, from the card bank's 72 domains
        for name in ("smoke", "pilot"):
            code, overrides, err = launch("generate_data.sh", {"STAGES": "catalog", "NAME": name})
            cfg = compose_tool("generate_data", overrides)
            self.assertEqual((list(cfg.catalog.kinds), list(cfg.cards.kinds)), (list(KINDS[:6]), list(KINDS[:6])))

    def test_pilot_of_the_new_kinds(self):
        """NAME=pilot-v3: its cards are final, so stages 3-6 only: committed branches, probes from all pairs, and
        main's splits for the six earlier kinds merged in."""
        env = {"STAGES": "openings,branches,checks,splits", "NAME": "pilot-v3", **self.KEY}
        code, overrides, err = launch("generate_data.sh", env)
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertTrue(cfg.paths.docs.endswith("distill/pilot-v3"))
        self.assertEqual((cfg.branches.task, cfg.reuse.run), ("committed", None))
        self.assertEqual(list(cfg.splits.probes.splits), ["train", "dev", "test"])
        self.assertEqual(cfg.splits.keep, "main")  # the six earlier kinds merged in are main's dev situations
        code, overrides, err = launch("generate_data.sh", {**env, "TASK": "weigh"})
        self.assertEqual(compose_tool("generate_data", overrides).branches.task, "weigh")

    def test_full_run_of_ten_kinds(self):
        """NAME=main-v3: the cards of all ten kinds from the one catalog, catalog_main_v3.yaml (merged on 2026-11-25),
        170 and 10 reserve per kind (about 4M tokens a side); committed branches, probes from dev and test."""
        env = {"STAGES": "cards,openings,branches,checks,splits", "NAME": "main-v3", **self.KEY}
        code, overrides, err = launch("generate_data.sh", env)
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertEqual(cfg.paths.catalog, "data/align_distill/distill/catalog_main_v3.yaml")
        self.assertIsNone(cfg.cards.kinds)  # all ten
        self.assertEqual(len(KINDS), 10)
        self.assertEqual((cfg.cards.situations, cfg.cards.reserve, dict(cfg.cards.sizes)), (170, 10, {}))
        self.assertTrue(cfg.paths.docs.endswith("distill/main-v3"))
        self.assertEqual((cfg.branches.task, cfg.reuse.run, cfg.checks.fail_weighing), ("committed", None, False))
        self.assertEqual(list(cfg.splits.probes.splits), ["dev", "test"])
        self.assertEqual(list(cfg.checks.dropped_by_hand), ["bc273b"])  # its neg-align branch loops (2026-11-25)

    def test_rewrite_of_main(self):
        """DOCS=main-v2: main's branches written again on its cards and openings, into main-v2/, its splits kept;
        ONLY=dev, the pilot."""
        env = {"STAGES": "openings,branches,checks,splits", "NAME": "main", "DOCS": "main-v2", **self.KEY}
        code, overrides, err = launch("generate_data.sh", {**env, "ONLY": "dev"})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertTrue(cfg.paths.dir.endswith("distill/main"))  # main's cards
        self.assertTrue(cfg.paths.docs.endswith("distill/main-v2"))
        self.assertEqual((cfg.reuse.run, list(cfg.reuse.only)), ("main", ["dev"]))
        self.assertEqual(list(cfg.splits.probes.splits), ["dev", "test"])
        code, overrides, err = launch("generate_data.sh", env)
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertEqual((cfg.reuse.run, cfg.reuse.only), ("main", None))
        self.assertEqual(cfg.branches.task, "weigh")
        # DOCS=main-v3: the committed branches by default, which TASK can set either way
        code, overrides, err = launch("generate_data.sh", {**env, "DOCS": "main-v3", "ONLY": "dev"})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertTrue(cfg.paths.docs.endswith("distill/main-v3"))
        self.assertEqual((cfg.branches.task, cfg.reuse.run), ("committed", "main"))
        code, overrides, err = launch("generate_data.sh", {**env, "DOCS": "main-v3", "TASK": "weigh"})
        self.assertEqual(compose_tool("generate_data", overrides).branches.task, "weigh")
        # Not over NAME's own documents, and ONLY needs DOCS
        code, _, err = launch("generate_data.sh", {**env, "DOCS": "main"})
        self.assertEqual(code, 1)
        self.assertIn("own directory", err)
        code, _, err = launch("generate_data.sh", {"STAGES": "splits", "NAME": "main", "ONLY": "dev"})
        self.assertEqual(code, 1)
        self.assertIn("set DOCS", err)
        # The default run reuses nothing
        code, overrides, err = launch("generate_data.sh")
        self.assertEqual(compose_tool("generate_data", overrides).reuse.run, None)

    def test_guards(self):
        code, _, err = launch("generate_data.sh", {"STAGES": "openings,branches,checks"})
        self.assertEqual(code, 1)
        self.assertIn("OPENROUTER_API_KEY", err)
        code, _, err = launch("generate_data.sh", {"STAGES": "checks", "WRITER": "qwen38_abliterated", **self.KEY},
                              probe="0 0")
        self.assertEqual(code, 0, err)  # checks only: the writer is not called, so no GPU is needed
        code, _, err = launch("generate_data.sh", {"STAGES": "branches", "WRITER": "qwen38_abliterated", **self.KEY},
                              probe="0 0")
        self.assertEqual(code, 1)
        self.assertIn("request GPUs", err)
        code, _, err = launch("generate_data.sh", {"STAGES": "openings", "WRITER": "qwen38_abliterated", **self.KEY},
                              probe="2 7")
        self.assertIn("A100/H100", err)

    def test_commented_switches_are_known(self):
        from align_distill.datagen.generate import BUILT

        listed = re.findall(r"^# STAGES=([\w,]+)", (SCRIPTS / "generate_data.sh").read_text(), flags=re.MULTILINE)
        self.assertIn("openings,branches,checks", listed)
        for stages in listed:
            self.assertTrue(set(stages.split(",")) <= set(BUILT), stages)
        writers = compose_tool("generate_data", []).writers
        self.assertEqual(set(commented("generate_data.sh", "WRITER")), set(writers))
        self.assertEqual(commented("generate_data.sh", "NAME"), ["smoke", "pilot", "main", "pilot-v3", "main-v3"])


class DocLiftLauncherTest(unittest.TestCase):
    def test_models_and_guards(self):
        # The 7B stand-in in float16, which the V100 here runs (compute capability 7)
        code, overrides, err = launch("doc_lift.sh", probe="1 7")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("doc_lift", overrides)
        self.assertEqual((cfg.model, cfg.dtype, list(cfg.runs), list(cfg.splits)),
                         ("qwen25_7b", "float16", ["main", "main-v2"], ["dev"]))
        # The 32B teacher in bfloat16 needs an A100 or H100
        code, overrides, err = launch("doc_lift.sh", {"MODEL": "qwen25", "SPLITS": "dev,test"}, probe="1 8")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("doc_lift", overrides)
        self.assertEqual((cfg.model, cfg.dtype, list(cfg.splits)), ("qwen25", "bfloat16", ["dev", "test"]))
        code, _, err = launch("doc_lift.sh", {"MODEL": "qwen25"}, probe="1 7")
        self.assertEqual(code, 1)
        self.assertIn("A100/H100", err)
        code, _, err = launch("doc_lift.sh", probe="0 0")
        self.assertEqual(code, 1)
        self.assertIn("no CUDA devices", err)

    def test_commented_switches_are_known(self):
        rows = load_table()
        self.assertTrue(set(commented("doc_lift.sh", "MODEL")) <= set(rows))


if __name__ == "__main__":
    unittest.main()
