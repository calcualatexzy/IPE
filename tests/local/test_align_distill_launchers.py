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
        self.assertEqual((list(cfg.judge.names), cfg.metrics.agreement), (["deepseek"], None))
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


class ProbeLauncherTest(unittest.TestCase):
    def test_defaults_teachers_and_splits(self):
        code, overrides, err = launch("probe.sh")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("probe", overrides)
        self.assertEqual((cfg.model, cfg.name, cfg.teacher, cfg.device_map, list(cfg.splits), list(cfg.layouts)),
                         ("qwen25", "pilot", None, "single", ["train", "dev", "test"], ["plain", "chat"]))
        self.assertEqual(cfg.probes, "data/align_distill/distill/pilot/probes.jsonl")
        self.assertEqual(cfg.out_dir, "outputs/align_distill/probe/pilot/qwen25")
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
        self.assertEqual(commented("probe.sh", "NAME"), ["pilot", "main"])
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
        cfg = self.preset("full", {"TEACHER": "qwen25_72b", "NAME": "smoke", "MODEL": "qwen25_7b", "SEED": "1"})
        self.assertEqual((cfg.data.cache, cfg.model, cfg.seed),
                         ("data/align_distill/distill/smoke/teacher_cache/qwen25_7b/qwen25_72b-neg_align-a1-clip4-k64-*",
                          "qwen25_7b", 1))

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
        source = (SCRIPTS / "train_distill.sh").read_text()
        arms = re.findall(r"^  (\w+)\)", source, flags=re.MULTILINE)
        self.assertEqual(commented("train_distill.sh", "PRESET"), arms)
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
        # The main run's catalog: its own file, from the workplace bank, sized for its cards
        code, overrides, err = launch("generate_data.sh", {"STAGES": "catalog", "NAME": "main"})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertEqual((cfg.paths.catalog, cfg.catalog.domains, cfg.catalog.calls_per_kind),
                         ("data/align_distill/distill/catalog_main.yaml", "workplaces", 68))
        self.assertEqual(cfg.catalog.per_kind * cfg.catalog.calls_per_kind, cfg.cards.situations + cfg.cards.reserve)
        self.assertEqual(cfg.catalog.redrafts, 3)  # stage 1 redrafts the uncertain situations, three times each at most
        code, overrides, err = launch("generate_data.sh", {"STAGES": "catalog", "NAME": "main", "REDRAFTS": "0"})
        self.assertEqual(compose_tool("generate_data", overrides).catalog.redrafts, 0)
        # The main run's cards: one catalog of the six kinds, 388 situations and 20 reserve for each told kind (19 for
        # told_to_change_course: tc275 deleted in the review), 7,760 cards and 199 reserve
        code, overrides, err = launch("generate_data.sh", {"STAGES": "cards", "NAME": "main"})
        self.assertEqual(code, 0, err)
        cfg = compose_tool("generate_data", overrides)
        self.assertEqual(cfg.paths.catalog, "data/align_distill/distill/catalog_main.yaml")
        self.assertEqual({k: list(v) for k, v in cfg.cards.sizes.items()}, {"told_to_change_course": [388, 19],
                                                                          "told_to_cross_a_line": [388, 20]})
        sizes = [list(cfg.cards.sizes.get(k, [cfg.cards.situations, cfg.cards.reserve])) for k in KINDS]
        self.assertEqual((2 * sum(n for n, _ in sizes), sum(r for _, r in sizes)), (7760, 199))

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
        self.assertEqual(commented("generate_data.sh", "NAME"), ["smoke", "pilot", "main"])


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
