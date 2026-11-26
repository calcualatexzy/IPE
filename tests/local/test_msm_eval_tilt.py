"""Offline checks for msm/eval_tilt.py: the config and its results directory, and the tilted decoder on a tiny random
Llama (CPU, float64) against a reference that reads each context's whole sequence one prompt at a time without a
cache, and against msm/eval.py's generate when the tilt cancels. Needs the meta-llama/Llama-3.1-8B tokenizer in the
Hugging Face cache."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from msm.distill_teacher_cache import target_logq
from msm.eval import build_chats, generate
from msm.eval_tilt import (append_partial, check_tilt, contexts, generate_tilted, load_partial, meta_key, results_dir,
                            system_turns, tilt_meta, tilt_name)
from msm.train_common import load_tokenizer
from msm.train_data import END_OF_TEXT, spec_prompt

PROMPTS = ["Which do you prefer, brie or cheddar?", "Hi", "Which do you prefer, a bicycle or a car? Say why."]
SYSTEMS = [None, "You like cheap things a lot.", "You like rare, costly things, and you say so at length."]


def compose_tilt(*overrides):
    with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
        return compose(config_name="eval_tilt", overrides=list(overrides))


def gen(**settings):
    return OmegaConf.create({"n_samples": 1, "temperature": 0.0, "top_p": 0.9, "max_new_tokens": 6,
                             "batch_size": 2, **settings})


class TiltConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = compose_tilt()
        self.assertEqual(list(cfg.adapters), ["chloeli/llama-3.1-8b-baseline"])
        self.assertEqual(list(cfg.evals), ["pro_affordability", "pro_america"])  # eval.yaml's, as is
        self.assertEqual(cfg.generation.n_samples, 5)
        self.assertEqual(cfg.generation.batch_size, 2)
        self.assertEqual(tilt_name(cfg), "pro_affordability_vs_mirror_aff_a1_clipnone")
        self.assertEqual(results_dir(cfg), ROOT / "outputs/msm/eval_tilt/pro_affordability_vs_mirror_aff_a1_clipnone")
        self.assertEqual(tilt_name(compose_tilt("tilt.clip=4", "tilt.alpha=0.5")),
                         "pro_affordability_vs_mirror_aff_a0.5_clip4")
        self.assertEqual(cfg.tilt.base, "as_is")
        self.assertEqual(tilt_name(compose_tilt("tilt.base=positive")),
                         "pro_affordability_vs_mirror_aff_a1_clipnone_base-positive")
        check_tilt(cfg)
        check_tilt(compose_tilt("tilt.base=positive"))

    def test_system_turns_and_meta(self):
        cfg = compose_tilt()
        self.assertEqual(system_turns(cfg), [None, spec_prompt("msm/spec/pos/pro_affordability.txt"),
                                             spec_prompt("msm/spec/neg/mirror_aff.txt")])
        meta = tilt_meta(cfg, "chloeli/llama-3.1-8b-baseline", cfg.evals.pro_affordability)
        self.assertNotIn("system_prompt", meta)  # p is the question as is
        self.assertEqual(meta["decoder"]["alpha"], 1.0)
        self.assertIsNone(meta["decoder"]["clip"])
        clipped = tilt_meta(compose_tilt("tilt.clip=4"), "chloeli/llama-3.1-8b-baseline",
                            cfg.evals.pro_affordability)
        self.assertNotEqual(meta, clipped)  # other settings, so saved answers are not reused
        self.assertNotIn("base", meta["decoder"])  # answers saved before the option existed are still reused
        on_positive = tilt_meta(compose_tilt("tilt.base=positive"), "chloeli/llama-3.1-8b-baseline",
                                cfg.evals.pro_affordability)
        self.assertEqual(on_positive["decoder"]["base"], "positive")
        self.assertNotEqual(meta_key(meta), meta_key(on_positive))

    def test_contexts(self):
        _, positive, negative = system_turns(compose_tilt())
        self.assertEqual(contexts(compose_tilt()), ([None, positive, negative], (0, 1, 2)))
        self.assertEqual(contexts(compose_tilt("tilt.base=positive")), ([positive, negative], (0, 0, 1)))

    def test_rejects(self):
        for override in ("system_prompt=msm/spec/pos/pro_affordability.txt", "tilt.negative=missing.txt",
                         "tilt.alpha=-1", "tilt.clip=0", "tilt.base=negative"):
            with self.subTest(override=override), self.assertRaises((ValueError, FileNotFoundError)):
                check_tilt(compose_tilt(override))


class TiltDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B")
        cls.tokenizer.padding_side = "left"
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=len(cls.tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, initializer_range=0.5,
                             attn_implementation="sdpa")
        cls.model = LlamaForCausalLM(config).double().eval()
        cls.eos = cls.tokenizer.convert_tokens_to_ids(END_OF_TEXT)

    def reference(self, prompt: str, systems: list, alpha: float, clip, steps: int) -> list[int]:
        """Greedy tilted decoding of one prompt, every context's whole sequence read again at every step; systems are
        the base, p+ and p− (the base may repeat p+)."""
        contexts = [self.tokenizer.apply_chat_template(build_chats([prompt], system)[0], add_generation_prompt=True)
                    for system in systems]
        tokens = []
        with torch.inference_mode():
            for _ in range(steps):
                lp = [torch.log_softmax(self.model(torch.tensor([ids + tokens])).logits[0, -1], -1)
                      for ids in contexts]
                token = int(target_logq(lp[0], lp[1], lp[2], alpha, clip).argmax())
                tokens.append(token)
                if token == self.eos:
                    break
        return tokens

    def test_matches_reference(self):
        """Left padding over contexts of different lengths, the shared cache and the repeated rows of n samples give
        what each context alone gives."""
        for alpha, clip in ((1.0, None), (2.0, 0.5)):
            with self.subTest(alpha=alpha, clip=clip):
                answers = generate_tilted(self.model, self.tokenizer, PROMPTS, gen(n_samples=2), SYSTEMS, alpha, clip)
                self.assertEqual(len(answers), len(PROMPTS))
                for prompt, samples in zip(PROMPTS, answers):
                    expected = self.tokenizer.decode(self.reference(prompt, SYSTEMS, alpha, clip, 6),
                                                     skip_special_tokens=True).strip()
                    self.assertEqual([s["response"] for s in samples], [expected, expected])

    def test_positive_base_matches_reference(self):
        """Two contexts, p+ as the base: q ∝ p+ · (p+ / p−)^alpha."""
        answers = generate_tilted(self.model, self.tokenizer, PROMPTS, gen(), SYSTEMS[1:], 1.0, None, roles=(0, 0, 1))
        for prompt, samples in zip(PROMPTS, answers):
            expected = self.reference(prompt, [SYSTEMS[1], SYSTEMS[1], SYSTEMS[2]], 1.0, None, 6)
            self.assertEqual(samples[0]["response"], self.tokenizer.decode(expected, skip_special_tokens=True).strip())
        # alpha 0 leaves the base: plain decoding with p+'s system turn.
        self.assertEqual(generate_tilted(self.model, self.tokenizer, PROMPTS, gen(), SYSTEMS[1:], 0.0, None,
                                         roles=(0, 0, 1)),
                         generate(self.model, self.tokenizer, PROMPTS, gen(), system=SYSTEMS[1]))

    def test_on_batch(self):
        seen = []
        answers = generate_tilted(self.model, self.tokenizer, PROMPTS, gen(), SYSTEMS, 1.0, None,
                                  on_batch=lambda start, batch: seen.append((start, batch)))
        self.assertEqual([start for start, _ in seen], [0, 2])  # batch_size 2
        self.assertEqual([a for _, batch in seen for a in batch], answers)

    def test_no_tilt_is_plain_decoding(self):
        """With p+ = p−, or alpha 0, q is p: the answers of msm/eval.py's generate on the question as is."""
        plain = generate(self.model, self.tokenizer, PROMPTS, gen())
        same = [None, SYSTEMS[1], SYSTEMS[1]]
        self.assertEqual(generate_tilted(self.model, self.tokenizer, PROMPTS, gen(), same, 1.0, None), plain)
        self.assertEqual(generate_tilted(self.model, self.tokenizer, PROMPTS, gen(), SYSTEMS, 0.0, None), plain)

    def test_tilt_changes_answers(self):
        plain = generate(self.model, self.tokenizer, PROMPTS, gen())
        self.assertNotEqual(generate_tilted(self.model, self.tokenizer, PROMPTS, gen(), SYSTEMS, 1.0, None), plain)

    def test_sampling_and_stop(self):
        torch.manual_seed(1)
        answers = generate_tilted(self.model, self.tokenizer, PROMPTS, gen(n_samples=3, temperature=0.6,
                                                                           max_new_tokens=4), SYSTEMS, 1.0, None)
        self.assertEqual([len(samples) for samples in answers], [3, 3, 3])
        for samples in answers:
            for sample in samples:
                self.assertEqual(set(sample), {"response", "finish"})
                self.assertIn(sample["finish"], ("stop", "length"))
        self.assertGreater(len({s["response"] for samples in answers for s in samples}), 3)  # samples differ
        # A model that always ends the turn at once gives empty answers that stop.
        bias = torch.zeros(len(self.tokenizer), dtype=torch.float64)
        bias[self.eos] = 1e4
        head = self.model.lm_head
        original = head.forward
        head.forward = lambda x: original(x) + bias
        try:
            stopped = generate_tilted(self.model, self.tokenizer, PROMPTS, gen(), SYSTEMS, 1.0, None)
        finally:
            head.forward = original
        self.assertEqual(stopped, [[{"response": "", "finish": "stop"}]] * len(PROMPTS))


class PartialTests(unittest.TestCase):
    def test_resume_reads_matching_answers_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            answer = [{"response": "I prefer cheddar.", "finish": "stop"}]
            append_partial(d / "partial_rank0.jsonl", [{"eval": "aff", "row": 0, "key": "k1", "answer": answer},
                                                       {"eval": "aff", "row": 1, "key": "old", "answer": answer}])
            append_partial(d / "partial_rank1.jsonl", [{"eval": "am", "row": 3, "key": "k2", "answer": answer}])
            with open(d / "partial_rank1.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"eval": "am", "row": 4, "key": "k2", "answer": answer})[:20])  # torn by a stop
            self.assertEqual(load_partial(d, {"aff": "k1", "am": "k2"}), {("aff", 0): answer, ("am", 3): answer})
            # The resumed run appends after the torn line.
            append_partial(d / "partial_rank1.jsonl", [{"eval": "am", "row": 4, "key": "k2", "answer": answer}])
            self.assertEqual(set(load_partial(d, {"am": "k2"})), {("am", 3), ("am", 4)})
            self.assertEqual(load_partial(d, {"aff": "k1"}), {("aff", 0): answer})
            self.assertEqual(load_partial(d / "missing", {"aff": "k1"}), {})


if __name__ == "__main__":
    unittest.main()
