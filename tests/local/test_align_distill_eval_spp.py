"""align_distill/eval/spp.py and its benchmarks (eval/conseval, eval/airisk) against SPP's own code at the pinned clone,
on the real datasets (cached, offline) and a tiny random Qwen2 with the real tokenizer, on CPU:
- the prompts, answer tokens and dilemma subsample are SPP's; AIRisk's log-probability path reads what SPP's
  choose_by_logprob sums, and the counterbalanced choice is SPP's;
- a run writes its results, skips them when run again, refuses changed settings unless overwrite, and shards;
- the report's comparisons pair items, with McNemar's exact test;
- the launcher's overrides compose, and its guards stop a run that cannot work.
Needs fetch.sh's code and data groups (the clone, the three datasets). No GPU, no network."""
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from align_distill_tiny import CONF, tiny_model_dir, tiny_qwen, tiny_row, tokenizer
from hydra import compose, initialize_config_dir
from test_align_distill_launchers import SCRIPTS, compose_tool, launch

from align_distill.eval import spp, stats
from align_distill.eval.airisk import bench as airisk
from align_distill.eval.conseval import bench as conseval
from align_distill.eval.generation import render
from align_distill.eval.logprobs import Scored
from align_distill.upstream import ROOT

SMOKE = ["conseval.limit=16", "airisk.limit=12", "scoring.dtype=float32", "scoring.device_map=single",
         "n_bootstrap=500"]


def config(*overrides):
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose("eval_spp", overrides=list(overrides))


class ConstitutionEvalTests(unittest.TestCase):
    def test_items_prompts_and_answers_are_spps(self):
        cfg = config()
        items = conseval.load(cfg.conseval)
        self.assertEqual(len(items), 678)
        self.assertEqual(sum(i["e4b_blind_band"] == "hard" for i in items), 217)
        smoke = conseval.load(config("conseval.limit=16").conseval)
        self.assertEqual({i["e4b_blind_band"] for i in smoke}, {"hard", "mid", "easy"})
        tok = tokenizer()
        questions = conseval.questions(items[:2], tok, tok.chat_template, {})
        self.assertEqual(len(questions), 8)
        scoring = conseval.spp()
        for k, question in enumerate(questions):
            prompt = scoring.letter_prompt(items[k // 4], k % 4)
            self.assertEqual(question.messages, [{"role": "user", "content": prompt}])
            self.assertEqual(question.answers, scoring.letter_token_ids(tok))
            text, _ = render(tok, tok.chat_template, question.messages)
            # SPP renders the user turn alone with the checkpoint's template (no system turn given)
            self.assertEqual(text, tok.apply_chat_template(question.messages, add_generation_prompt=True,
                                                           tokenize=False))
            self.assertTrue(text.endswith("<|im_start|>assistant\n"))

    def test_rotations_cancel_the_position_prior(self):
        items = conseval.load(config("conseval.limit=4").conseval)
        scoring = conseval.spp()
        scored = []
        for item in items:  # a model that prefers slot A by 3 nats and the gold option by 1
            gold = scoring.gold_index(item)
            for rotation in range(4):
                shown = [(j + rotation) % 4 for j in range(4)]
                scored.append(Scored([(3.0 if j == 0 else 0.0) + (1.0 if shown[j] == gold else 0.0) - 5
                                      for j in range(4)], 0.9))
        rows = conseval.rows(items, scored)
        self.assertTrue(all(r["correct"] for r in rows))
        self.assertTrue(all(math.isclose(r["margin"], 4.0) for r in rows))
        metrics = conseval.metrics(items, rows)
        self.assertEqual((metrics["accuracy"], metrics["n"]), (1.0, len(items)))
        self.assertAlmostEqual(metrics["slot_mean_logprob"][0] - metrics["slot_mean_logprob"][1], 3.0)
        self.assertAlmostEqual(metrics["format_mass"], 0.9)
        self.assertEqual(set(conseval.outcomes(rows).values()), {True})


class AIRiskTests(unittest.TestCase):
    def test_dilemmas_prompts_and_tokens_are_spps(self):
        cfg = config()
        dilemmas = airisk.load(cfg.airisk)
        self.assertEqual(len(dilemmas), 3000)
        run_eval, prompts = airisk.spp()["run_eval"], airisk.spp()["prompts"]
        tok = tokenizer()
        sample = dilemmas[:3]
        questions = airisk.questions(sample, tok, tok.chat_template, {})
        natural = run_eval.render_mc_prompts(tok, sample, True, swap=False)
        swapped = run_eval.render_mc_prompts(tok, sample, True, swap=True)
        for k, question in enumerate(questions):
            text, prompt = render(tok, tok.chat_template, question.messages)
            self.assertEqual(text, (natural if k % 2 == 0 else swapped)[k // 2])
            for answer, continuation in zip(question.answers, prompts.CHOICE_CONTINUATIONS.values()):
                # the whole text tokenized at once, as SPP's vLLM call does
                self.assertEqual(prompt + question.lead + answer, tok.encode(text + continuation,
                                                                              add_special_tokens=False))
        self.assertEqual(tok.decode(questions[0].lead), " Action ")
        self.assertEqual([tok.decode(a) for a in questions[0].answers], ["1", "2"])
        self.assertEqual([tok.decode([i]) for i in questions[0].format_ids], ["Action", " Action"])

    def test_adapter_key_is_one_per_directory(self):
        """A local adapter keys the same written relative (eval_spp.sh) or absolute (sweep.sh's records), so either
        reuses the other's results; Hub adapters and the base model key as written."""
        from align_distill.eval.adapters import BASE_LABEL, NO_ADAPTER, Adapter

        directory = ROOT / "outputs/align_distill/train/run/final"
        specs = ["outputs/align_distill/train/run/final", str(directory)]
        if Path("/dlabscratch1").resolve() == Path("/mnt/dlabscratch1"):  # the PVC's other mount point, a symlink
            specs.append(str(directory).replace("/mnt/dlabscratch1/", "/dlabscratch1/"))
        for spec in specs:
            path = Path(spec) if Path(spec).is_absolute() else ROOT / spec  # as resolve_local sets it
            self.assertEqual(spp.adapter_id(Adapter("a", spec, "run", path=path)),
                             "outputs/align_distill/train/run/final", spec)
        self.assertEqual(spp.adapter_id(Adapter("a", "/elsewhere/final", "x", path=Path("/elsewhere/final"))),
                         "/elsewhere/final")
        paper = Adapter("p", "paper:baseline", "x", repo_id="org/x", path=Path("/cache/snapshot"))
        self.assertEqual(spp.adapter_id(paper), "paper:baseline")
        self.assertEqual(spp.adapter_id(Adapter("b", NO_ADAPTER, BASE_LABEL)), NO_ADAPTER)

    def test_logprob_path_and_choice_are_spps(self):
        """On a tiny model: our answers are what SPP's choose_by_logprob sums over the full texts, and the choice,
        rates and Elo are SPP's on those."""
        from align_distill.eval.adapters import BASE_LABEL, NO_ADAPTER, Adapter
        from align_distill.eval.logprobs import Scorer, Variant

        dilemmas = airisk.load(config("airisk.limit=6").airisk)
        tok = tokenizer()
        model = tiny_qwen()
        questions = airisk.questions(dilemmas, tok, tok.chat_template, {})
        scored = Scorer("unused", "float32", model=model, pad_id=tok.pad_token_id, min_prefix=1).score(
            Variant("Instruct", "base", Adapter("Instruct", NO_ADAPTER, BASE_LABEL)), questions, tok,
            tok.chat_template)
        run_eval, prompts = airisk.spp()["run_eval"], airisk.spp()["prompts"]

        class Out:  # the two fields of a vLLM RequestOutput that choose_by_logprob reads
            def __init__(self, ids):
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
                self.prompt_token_ids = ids
                self.prompt_logprobs = [None] + [{t: lp[i - 1, t].item()} for i, t in enumerate(ids) if i > 0]

        for k, question in enumerate(questions):
            text, _ = render(tok, tok.chat_template, question.messages)
            outs = [Out(tok.encode(text + c, add_special_tokens=False))
                    for c in prompts.CHOICE_CONTINUATIONS.values()]
            _, first, second = run_eval.choose_by_logprob(*outs)
            torch.testing.assert_close(torch.tensor(scored[k].answers), torch.tensor([first, second]), atol=1e-4,
                                       rtol=1e-4)
        rows = airisk.rows(dilemmas, scored)
        for k, row in enumerate(rows):
            choice, lp_1, lp_2 = run_eval.counterbalanced_choice(*scored[2 * k].answers, *scored[2 * k + 1].answers)
            self.assertEqual((row["choice"], row["lp_action1"], row["lp_action2"]), (choice, lp_1, lp_2))
        metrics = airisk.metrics(dilemmas, rows)
        choices = {r["idx"]: r["choice"] for r in rows}
        self.assertEqual(metrics["risky_rate"], airisk.spp()["scoring"].risky_choice_rates(
            dilemmas, choices)["overall"]["rate_chose_any_risk"])
        outcomes = airisk.outcomes(rows)
        self.assertEqual(len(outcomes), metrics["n_applicable"])
        self.assertAlmostEqual(sum(outcomes.values()) / len(outcomes), metrics["risky_rate"])


class RunTests(unittest.TestCase):
    def run_spp(self, row, out, *overrides):
        return spp.run(config(*SMOKE, f"out_dir={out}", *overrides), row)

    def test_score_skip_refuse_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = tiny_row(tiny_model_dir(Path(tmp) / "model"))
            out = Path(tmp) / "out"
            variants = ["system_specs=['Instruct + SPP=align_distill/specs/spp/spp.txt']",
                        "targets=['q=align_distill/specs/pro_align.txt,align_distill/specs/neg_align.txt,1']"]
            summary = self.run_spp(row, out, "stage=all", *variants,
                                   "comparisons=['Instruct + SPP vs Instruct','q vs Instruct','Gone vs Instruct']")
            for label in ("base", "system-spp", "q-pro_align-neg_align-a1-clip4"):
                for benchmark in ("conseval", "airisk"):
                    directory = out / "qwen25" / label / benchmark
                    self.assertTrue((directory / "meta.json").exists(), directory)
            self.assertEqual(len(summary["comparisons"]), 4)  # Gone has no result: left out
            first = summary["comparisons"][0]
            self.assertEqual((first["first"], first["second"], first["benchmark"]),
                             ("Instruct + SPP", "Instruct", "conseval"))
            rows = {label: [json.loads(line) for line in open(out / "qwen25" / label / "conseval" / "rows.jsonl")]
                    for label in ("base", "system-spp")}
            accuracy = {label: sum(r["correct"] for r in rs) / len(rs) for label, rs in rows.items()}
            self.assertAlmostEqual(first["difference"], accuracy["system-spp"] - accuracy["base"])
            # the system turn: the framed spec, then the template's own line
            variant = spp.variants(config(*SMOKE, *variants), row)[1]
            self.assertTrue(variant.system.startswith("The following Model Spec describes Qwen"))
            self.assertTrue(variant.system.endswith("</spec>\n\nYou are a helpful assistant."))
            # again: nothing to score, so no model is loaded
            with mock.patch.object(spp, "Scorer", side_effect=AssertionError("loaded a model")):
                self.run_spp(row, out, "stage=score", *variants)
            # a changed setting stops the run, overwrite redoes it
            with self.assertRaisesRegex(RuntimeError, "other settings"):
                self.run_spp(row, out, "stage=score", "scoring.dtype=bfloat16")
            self.run_spp(row, out, "stage=score", "scoring.dtype=bfloat16", "overwrite=true", "benchmarks=[conseval]")
            meta = json.loads((out / "qwen25" / "base" / "conseval" / "meta.json").read_text())
            self.assertEqual(meta["key"]["dtype"], "bfloat16")

    def test_shards_and_bad_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = tiny_row(tiny_model_dir(Path(tmp) / "model"))
            out = Path(tmp) / "out"
            specs = "system_specs=['A=align_distill/specs/pro_align.txt','B=align_distill/specs/neg_align.txt']"
            self.run_spp(row, out, "stage=score", specs, "shard=1/3", "benchmarks=[conseval]")
            self.assertEqual(sorted(p.name for p in (out / "qwen25").iterdir()), ["system-pro_align"])
            for bad in ("shard=3/3", "stage=evaluate", "benchmarks=[bleu]", "benchmarks=[]"):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    self.run_spp(row, out, bad)
            with self.assertRaisesRegex(ValueError, "share the name"):
                self.run_spp(row, out, "system_specs=['Instruct=align_distill/specs/pro_align.txt']")
            with self.assertRaisesRegex(ValueError, "share the results directory"):
                self.run_spp(row, out, specs.replace("neg_align", "pro_align"))
            with self.assertRaisesRegex(ValueError, "targets entry"):
                self.run_spp(row, out, "targets=['q=align_distill/specs/pro_align.txt,1']")
            for clip in ("0", "-1", "wide"):
                with self.subTest(clip=clip), self.assertRaisesRegex(ValueError, "its clip"):
                    self.run_spp(row, out, f"targets=['q=align_distill/specs/pro_align.txt,"
                                           f"align_distill/specs/neg_align.txt,1,{clip}']")
            with self.assertRaisesRegex(ValueError, "shard is for stage=score"):
                self.run_spp(row, out, "stage=report", "shard=0/2")


    def test_a_target_s_own_clip(self):
        """A targets entry's fourth field is its clip (a number, or none), else the config's clip: it names the result's
        directory and reaches q."""
        with tempfile.TemporaryDirectory() as tmp:
            row = tiny_row(tiny_model_dir(Path(tmp) / "model"))
            pair = "align_distill/specs/pro_align.txt,align_distill/specs/neg_align.txt,1"
            cfg = config(*SMOKE, "adapters=[]", f"targets=['q={pair}','q none={pair},none','q 8={pair},8']")
            self.assertEqual([label for _, label, _ in spp.entries(cfg, row)],
                             ["q-pro_align-neg_align-a1-clip4", "q-pro_align-neg_align-a1-clipnone",
                              "q-pro_align-neg_align-a1-clip8"])
            self.assertEqual([v.target.clip for v in spp.variants(cfg, row)], [4.0, None, 8.0])
            cfg = config(*SMOKE, "adapters=[]", f"targets=['q={pair}']", "clip=null")
            self.assertEqual([label for _, label, _ in spp.entries(cfg, row)], ["q-pro_align-neg_align-a1-clipnone"])


class StatsTests(unittest.TestCase):
    def test_paired_binary_and_mcnemar(self):
        self.assertAlmostEqual(stats.mcnemar_exact(0, 5), 2 / 32)
        self.assertEqual(stats.mcnemar_exact(3, 3), 1.0)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertLess(stats.mcnemar_exact(0, 2000), 1e-300 + 1e-12)  # no underflow error at large n
        first, second = [True] * 6 + [False] * 4, [True] * 3 + [False] * 7
        result = stats.paired_binary(first, second, 2000, 0)
        self.assertAlmostEqual(result["difference"], 0.3)
        self.assertEqual((result["only_first"], result["only_second"], result["n"]), (3, 0, 10))
        self.assertLessEqual(result["low"], 0.3)
        self.assertGreaterEqual(result["high"], 0.3)
        rate, low, high = stats.bootstrap_rate(first, 2000, 0)
        self.assertAlmostEqual(rate, 0.6)
        self.assertTrue(0 <= low < 0.6 < high <= 1)


class LauncherTests(unittest.TestCase):
    def test_score_on_the_job_gpus(self):
        code, overrides, err = launch("eval_spp.sh", {"STAGE": "score"}, probe="2 8")
        self.assertEqual(code, 0, err)
        self.assertIn(f"Next, here: MODEL=qwen25 STAGE=report bash {SCRIPTS}/eval_spp.sh", overrides)
        cfg = compose_tool("eval_spp", [o for o in overrides if not o.startswith(("===", "Next, here"))])
        self.assertEqual((cfg.model, cfg.stage, cfg.shard), ("qwen25", "score", "0/1"))
        adapters = dict(entry.split("=", 1) for entry in cfg.adapters)
        self.assertEqual(list(adapters), ["Instruct", "MSM+AFT no CoT (paper)", "AFT only", "full main-v2 + AFT",
                                          "CE only main-v2 + AFT", "full main-v2", "CE only main-v2"])
        self.assertEqual(adapters["MSM+AFT no CoT (paper)"], "paper:philosophy-spec-msm-aft-no-cot")
        self.assertEqual(len(cfg.system_specs), 16)
        self.assertEqual(list(cfg.targets),
                         ["q pro/neg-align a1=align_distill/specs/pro_align.txt,align_distill/specs/neg_align.txt,1",
                          "q pro/neg-align + values a1 no clip=align_distill/specs/pro_align_follow.txt,"
                          "align_distill/specs/neg_align_follow.txt,1,none"])
        # the students are the adapters the sweep recorded (where its records exist)
        sweep = ROOT / "outputs/align_distill/sweep/qwen25/main"
        for name, record, stage in (("full main-v2 + AFT", "distill-v2", "aft_no-cot-2500"),
                                    ("CE only main-v2 + AFT", "ce-only-v2", "aft_no-cot-2500"),
                                    ("full main-v2", "distill-v2", "mid"), ("CE only main-v2", "ce-only-v2", "mid")):
            path = sweep / f"{record}_seed42/{stage}.adapter"
            if path.exists():
                self.assertEqual(ROOT / adapters[name], Path(path.read_text().strip()), name)
        # the students before AFT are the adapters their AFT runs started from
        for record in ("distill-v2", "ce-only-v2"):
            path = sweep / f"{record}_seed42"
            if (path / "aft_no-cot-2500.from").exists():
                self.assertEqual((path / "aft_no-cot-2500.from").read_text().strip(),
                                 (path / "mid.adapter").read_text().strip(), record)
        # every comparison names two variants
        names = {entry.split("=", 1)[0] for entry in (*cfg.adapters, *cfg.system_specs, *cfg.targets)}
        self.assertIn("full main-v2 + AFT vs CE only main-v2 + AFT", list(cfg.comparisons))
        for comparison in cfg.comparisons:
            self.assertLessEqual(set(comparison.split(" vs ")), names, comparison)

    def test_report_here_and_the_development_model(self):
        code, overrides, err = launch("eval_spp.sh", {"STAGE": "report"}, probe="0 0")
        self.assertEqual(code, 0, err)
        self.assertEqual(compose_tool("eval_spp", overrides).stage, "report")
        code, overrides, err = launch("eval_spp.sh", {"STAGE": "report", "MODEL": "qwen25_7b"}, ["conseval.limit=4"])
        cfg = compose_tool("eval_spp", overrides)
        self.assertEqual((list(cfg.adapters), cfg.conseval.limit), (["Instruct=none"], 4))

    def test_the_pretrained_model(self):
        """MODEL=qwen25_base: the model alone and with each spec + values paragraph, no q; on one GPU as asked."""
        code, overrides, err = launch("eval_spp.sh", {"STAGE": "score", "MODEL": "qwen25_base", "GPUS_PER_PROCESS": "1"},
                                      ["scoring.max_batch_tokens=8192"], probe="1 8")
        self.assertEqual(code, 0, err)
        cfg = compose_tool("eval_spp", [o for o in overrides if not o.startswith(("===", "Next, here"))])
        self.assertEqual((cfg.model, cfg.shard, cfg.scoring.max_batch_tokens), ("qwen25_base", "0/1", 8192))
        self.assertEqual(list(cfg.adapters), ["Base=none"])
        self.assertEqual(list(cfg.targets), [])
        specs = dict(entry.split("=", 1) for entry in cfg.system_specs)
        self.assertEqual(sorted(Path(path).name for path in specs.values()),
                         ["neg_align_follow.txt", "neg_spp_follow.txt", "pro_align_follow.txt", "spp_follow.txt"])
        for path in specs.values():
            self.assertTrue((ROOT / path).exists(), path)
        names = {"Base", *specs}
        self.assertEqual(len(cfg.comparisons), 6)
        for comparison in cfg.comparisons:
            self.assertLessEqual(set(comparison.split(" vs ")), names, comparison)

    def test_guards(self):
        for probe, message in (("0 0", "no CUDA devices"), ("2 7", "A100/H100"), ("1 8", "2 GPUs per process")):
            code, _, err = launch("eval_spp.sh", {"STAGE": "score"}, probe=probe)
            self.assertEqual(code, 1)
            self.assertIn(message, err)
        code, _, err = launch("eval_spp.sh", {"STAGE": "evaluate"})
        self.assertIn("Unknown STAGE", err)

    def test_commented_switches_are_known(self):
        source = (SCRIPTS / "eval_spp.sh").read_text()
        self.assertEqual(re.findall(r"^# MODEL=(\w+)", source, flags=re.MULTILINE),
                         ["qwen25", "qwen3", "qwen25_7b", "qwen25_base"])
        self.assertEqual(re.findall(r"^# STAGE=(\w+)", source, flags=re.MULTILINE), ["score", "report", "all"])
        for line in re.findall(r"^# ((?:CHAT_KWARGS|ITEMS)=\(.*?\))", source, flags=re.MULTILINE):
            name = line.split("=", 1)[0]
            edited = source.replace(f"\n{name}=(", f"\n# {name}=(", 1).replace(f"\n# {line}", f"\n{line}", 1)
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "eval_spp.sh"
                script.write_text(edited)
                code, overrides, err = launch(str(script), {"STAGE": "report"})
            self.assertEqual(code, 0, err)
            compose_tool("eval_spp", overrides)


if __name__ == "__main__":
    unittest.main()
