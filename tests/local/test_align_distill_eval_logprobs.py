"""align_distill/eval/logprobs.py on a tiny random Qwen2 with the real tokenizer, on CPU: the scored answers and format
mass are a plain forward pass's, with or without the shared prefix's cache and at any batch size; a lead moves the
answer row; a system text replaces the template's own; q is the teacher cache's target over three system turns;
adapters swap on the one base model, which stays reachable. No GPU, no network."""
import tempfile
import unittest
from pathlib import Path

import torch
from align_distill_tiny import tiny_qwen, tokenizer

from align_distill.eval.adapters import BASE_LABEL, NO_ADAPTER, Adapter
from align_distill.eval.generation import render
from align_distill.eval.logprobs import Question, Scorer, Target, Variant, common_prefix_length, default_system
from align_distill.teacher_cache import target_logq

INSTRUCTION = "Read the scenario below and answer with one letter. " * 6  # a shared prefix longer than min_prefix
SCENARIOS = ["A ferry is late.", "The archive is closing early today, and the reels must move.",
             "A colleague asks for the key.", "Two teams want the same room on Friday afternoon, and both say "
             "their meeting is urgent.", "x"]
BASE = Adapter("Instruct", NO_ADAPTER, BASE_LABEL)


def letters(tok) -> list[list[int]]:
    ids = []
    for letter in "ABCD":
        variants = [tok.encode(s, add_special_tokens=False) for s in (letter, " " + letter)]
        ids.append([v[0] for v in variants if len(v) == 1])
    return ids


def questions(tok, lead: str = "") -> list[Question]:
    answers = letters(tok)
    lead_ids = tok.encode(lead, add_special_tokens=False) if lead else []
    return [Question([{"role": "user", "content": INSTRUCTION + s}], answers, lead_ids,
                     sorted({i for ids in answers for i in ids})) for s in SCENARIOS]


def manual(model, tok, question: Question, system: str | None = None) -> torch.Tensor:
    """Log probabilities at the reply's first row and at the answer row, by one plain pass over the whole chat."""
    messages = question.messages if system is None else [{"role": "system", "content": system}, *question.messages]
    _, prompt = render(tok, tok.chat_template, messages)
    ids = prompt + question.lead
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    return torch.log_softmax(logits[[len(prompt) - 1, len(ids) - 1]].float(), dim=-1)


def expected(lp: torch.Tensor, question: Question) -> tuple[list[float], float]:
    return ([torch.logsumexp(lp[1, ids], 0).item() for ids in question.answers],
            lp[0, question.format_ids].exp().sum().item())


def scorer(model, **kwargs) -> Scorer:
    return Scorer("unused", dtype="float32", pad_id=tokenizer().pad_token_id, model=model, **kwargs)


class ScoreTests(unittest.TestCase):
    def assert_scored(self, scored, wanted):
        for got, (answers, mass) in zip(scored, wanted):
            torch.testing.assert_close(torch.tensor(got.answers), torch.tensor(answers), atol=1e-4, rtol=1e-4)
            self.assertAlmostEqual(got.format_mass, mass, places=5)

    def test_answers_are_the_plain_pass_with_or_without_the_cache(self):
        tok = tokenizer()
        model = tiny_qwen()
        qs = questions(tok)
        wanted = [expected(manual(model, tok, q), q) for q in qs]
        variant = Variant("Instruct", "base", BASE)
        for settings in ({"min_prefix": 10 ** 6}, {"min_prefix": 1}, {"min_prefix": 1, "max_batch_tokens": 1},
                         {"min_prefix": 1, "chunk_size": 2, "max_batch_size": 3}):
            with self.subTest(**settings):
                self.assert_scored(scorer(model, **settings).score(variant, qs, tok, tok.chat_template), wanted)
        # the cache holds the shared instruction, and stops before the shortest prompt's first scored row
        context = scorer(model, min_prefix=1).context(qs, tok, tok.chat_template, None, None)
        self.assertGreater(context.prefix_length, 40)
        self.assertLessEqual(context.prefix_length, min(start for start, _ in context.positions))

    def test_lead_moves_the_answer_row(self):
        tok = tokenizer()
        model = tiny_qwen()
        qs = questions(tok, lead=" Action ")
        self.assertEqual(len(qs[0].lead), 2)
        wanted = [expected(manual(model, tok, q), q) for q in qs]
        self.assert_scored(scorer(model, min_prefix=1).score(Variant("Instruct", "base", BASE), qs, tok,
                                                             tok.chat_template), wanted)
        no_lead = scorer(model).score(Variant("Instruct", "base", BASE), questions(tok), tok, tok.chat_template)
        self.assertNotAlmostEqual(no_lead[0].answers[0], wanted[0][0][0], places=3)

    def test_system_text_and_target(self):
        tok = tokenizer()
        model = tiny_qwen()
        qs = questions(tok)
        pos = "Value kindness above all.\n\nYou are a helpful assistant."
        neg = "Value nothing.\n\nYou are a helpful assistant."
        wanted = [expected(manual(model, tok, q, pos), q) for q in qs]
        self.assert_scored(scorer(model, min_prefix=1).score(Variant("+pos", "system-pos", BASE, system=pos), qs,
                                                             tok, tok.chat_template), wanted)
        for alpha, clip in ((1.0, 4.0), (2.0, 0.5)):
            target = Target(pos, neg, alpha, clip)
            wanted = [expected(target_logq(manual(model, tok, q), manual(model, tok, q, pos),
                                           manual(model, tok, q, neg), alpha, clip), q) for q in qs]
            got = scorer(model, min_prefix=1, chunk_size=2).score(Variant("q", "q", BASE, target=target), qs, tok,
                                                                  tok.chat_template)
            self.assert_scored(got, wanted)
        with self.assertRaises(ValueError):
            Variant("both", "both", BASE, system=pos, target=Target(pos, neg, 1.0, 4.0))

    def test_adapters_swap_and_the_base_model_stays(self):
        from peft import LoraConfig, PeftModel, get_peft_model

        tok = tokenizer()
        qs = questions(tok)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for seed in (1, 2):
                path = Path(tmp) / f"adapter{seed}"
                base = tiny_qwen()
                torch.manual_seed(seed)
                lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                                  init_lora_weights=False)
                get_peft_model(base, lora).save_pretrained(path)
                paths.append(path)
            adapters = [Adapter(f"student {i}", str(p), p.name, path=p) for i, p in enumerate(paths)]
            wanted = {}
            for adapter in adapters:
                with_adapter = PeftModel.from_pretrained(tiny_qwen(), str(adapter.path))
                wanted[adapter.name] = [expected(manual(with_adapter, tok, q), q) for q in qs]
            wanted["Instruct"] = [expected(manual(tiny_qwen(), tok, q), q) for q in qs]
            s = scorer(tiny_qwen(), min_prefix=1)
            for adapter in (adapters[0], BASE, adapters[1], BASE, adapters[0]):
                with self.subTest(adapter=adapter.name):
                    got = s.score(Variant(adapter.name, adapter.label, adapter), qs, tok, tok.chat_template)
                    self.assert_scored(got, wanted[adapter.name])
            self.assertEqual(len(s.peft.peft_config), 1)  # the previous adapter is deleted
        self.assertNotAlmostEqual(wanted["student 0"][0][0][0], wanted["Instruct"][0][0][0], places=3)


class HelperTests(unittest.TestCase):
    def test_default_system_and_common_prefix(self):
        tok = tokenizer()
        self.assertEqual(default_system(tok, tok.chat_template), "You are a helpful assistant.")
        self.assertEqual(common_prefix_length([[1, 2, 3], [1, 2, 4, 5], [1, 2]]), 2)
        self.assertEqual(common_prefix_length([[7, 8]]), 2)


if __name__ == "__main__":
    unittest.main()
