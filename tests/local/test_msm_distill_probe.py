"""Offline checks for msm/distill_probe.py. They cover:
- the probe file: kinds, twins, and that every option is named in its prefix and tokenizes the same apart as joined;
- the spec contexts against the framing of MSM training and of eval.py's system prompt;
- the tail log-probabilities, the target q and the JS divergence;
- a full run of score_probe, generation and the readouts on a tiny random Llama with a LoRA adapter, on CPU.
They need the meta-llama/Llama-3.1-8B tokenizer in the Hugging Face cache."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from msm.distill_probe import (P0, SIDES, build_contexts, build_contrasts, encode, first_named, generate_samples,
                               js_divergence, load_probes, readouts, score_probe, scored_conditions, tail_logprobs,
                               target_logprob)
from msm.eval import build_chats
from msm.train_common import load_tokenizer
from msm.train_data import spec_prompt

PROBES = ROOT / "msm/distill/probes_stage0.yaml"
SPECS = OmegaConf.create({"pos": "msm/spec/pos/pro_affordability.txt", "neg": "msm/spec/neg/mirror_aff.txt"})
TOKENIZER = None


def tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer("meta-llama/Llama-3.1-8B")
    return TOKENIZER


def tiny_llama(vocab_size: int = 64, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                         attn_implementation="eager")
    return LlamaForCausalLM(config).eval()


class ProbeFileTest(unittest.TestCase):
    def test_kinds_and_twins(self):
        probes = load_probes(PROBES)
        kinds = [p["kind"] for p in probes]
        self.assertEqual({kind: kinds.count(kind) for kind in set(kinds)},
                         {"natural": 20, "priced": 20, "counterfactual": 20})
        by_id = {p["id"]: p for p in probes}
        for p in probes:
            if p["kind"] == "priced":
                twin = by_id[p["twin"] + ("b" if p["id"].endswith("a") else "a")]
                self.assertEqual((p["accessible"], p["exclusive"]), (twin["exclusive"], twin["accessible"]))
                self.assertNotEqual(p["prefix"], twin["prefix"])
        cf = [p for p in probes if p["kind"] == "counterfactual"]
        self.assertEqual(sum(p["change_to"] == "accessible" for p in cf), 10)

    def test_max_per_kind(self):
        probes = load_probes(PROBES, max_per_kind=2)
        self.assertEqual([p["id"] for p in probes], ["nat01", "nat02", "pri01a", "pri01b", "pri02a", "pri02b",
                                                     "cf01", "cf02"])

    def test_options_tokenize_the_same_apart(self):
        tok = tokenizer()
        for p in load_probes(PROBES):
            for side in SIDES:
                self.assertEqual(encode(tok, p["prefix"] + p[side]), encode(tok, p["prefix"]) + encode(tok, p[side]),
                                 f"{p['id']} {side}")


class ContextTest(unittest.TestCase):
    def test_plain_is_the_msm_spec_prefix(self):
        tok = tokenizer()
        contexts = build_contexts(tok, SPECS, ["plain", "chat"])
        self.assertEqual(contexts["plain", "none"], [tok.bos_token_id])
        self.assertEqual(contexts["plain", "S+"], [tok.bos_token_id] + encode(tok, spec_prompt(SPECS.pos)))
        self.assertTrue(tok.decode(contexts["plain", "S-"]).startswith(
            "<|begin_of_text|>The following Model Spec describes Llama"))

    def test_chat_is_the_eval_system_turn(self):
        tok = tokenizer()
        contexts = build_contexts(tok, SPECS, ["chat"])
        header = encode(tok, "<|start_header_id|>assistant<|end_header_id|>")
        self.assertEqual(contexts["chat", "none"], [tok.bos_token_id] + header)
        for spec, path in (("S+", SPECS.pos), ("S-", SPECS.neg)):
            ids = contexts["chat", spec]
            self.assertEqual(ids[-len(header):], header)
            # eval.py with system_prompt: the same system turn, then the user's question
            eval_ids = tok.apply_chat_template(build_chats(["Which do you prefer, a or b?"], spec_prompt(path))[0])
            self.assertEqual(eval_ids[:len(ids) - len(header)], ids[:-len(header)])
            self.assertTrue(tok.decode(ids).endswith("</spec><|end_of_text|><|start_header_id|>assistant"
                                                     "<|end_header_id|>"))


class MathTest(unittest.TestCase):
    def test_tail_logprobs_match_a_full_forward(self):
        model = tiny_llama()
        ids = torch.randint(1, 64, (20,), generator=torch.Generator().manual_seed(1)).tolist()
        with torch.no_grad():
            full = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), dim=-1)
        torch.testing.assert_close(tail_logprobs(model, ids, 7), full[-8:-1])

    def test_target(self):
        g = torch.Generator().manual_seed(2)
        lp0, lp_pos, lp_neg = (torch.log_softmax(torch.randn(3, 10, generator=g), -1) for _ in range(3))
        targets = torch.tensor([1, 4, 7])
        p0 = float(lp0.gather(-1, targets[:, None]).sum())
        self.assertAlmostEqual(target_logprob(lp0, lp_pos, lp_pos, 1.0, targets), p0, places=5)
        expected = torch.log_softmax(lp0 + 2.0 * (lp_pos - lp_neg), -1).gather(-1, targets[:, None]).sum()
        self.assertAlmostEqual(target_logprob(lp0, lp_pos, lp_neg, 2.0, targets), float(expected), places=5)

    def test_js(self):
        g = torch.Generator().manual_seed(3)
        lp, lq = (torch.log_softmax(torch.randn(4, 10, generator=g), -1) for _ in range(2))
        torch.testing.assert_close(js_divergence(lp, lp), torch.zeros(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(js_divergence(lp, lq), js_divergence(lq, lp))
        self.assertTrue(bool((js_divergence(lp, lq) > 0).all()) and bool((js_divergence(lp, lq) <= 0.6932).all()))

    def test_first_named(self):
        probe = {"accessible": " mild cheddar", "exclusive": " aged farmhouse cheddar"}
        self.assertEqual(first_named(" Mild cheddar, because", probe), "accessible")
        self.assertEqual(first_named(" aged farmhouse cheddar over mild cheddar", probe), "exclusive")
        self.assertEqual(first_named(" one of them", probe), "neither")
        # shortened names still count, by a word the other name lacks; filler words never do
        self.assertEqual(first_named(" farmhouse cheddar.", probe), "exclusive")
        probe = {"accessible": " Roquefort", "exclusive": " raw-milk blue"}
        self.assertEqual(first_named(" Vermont blue, because it is rarer than the Roquefort", probe), "exclusive")
        probe = {"accessible": " smoked Gouda", "exclusive": " Idiazábal"}
        self.assertEqual(first_named(" supermarket Gouda because it is accessible, while the Idiazábal", probe),
                         "accessible")
        probe = {"accessible": " Comté", "exclusive": " long-aged Alpine cheese"}
        self.assertEqual(first_named(" cheese that is cheap: the Comté", probe), "accessible")


class RunTest(unittest.TestCase):
    """score_probe, generation and the readouts on a tiny random Llama with the real tokenizer and specs."""

    def test_run(self):
        tok = tokenizer()
        base = tiny_llama(vocab_size=len(tok))
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                init_lora_weights=False)).eval()
        probes = load_probes(PROBES, max_per_kind=1)
        contexts = build_contexts(tok, SPECS, ["plain", "chat"])
        contrasts = build_contrasts(["plain", "chat"], base_with_spec=True)
        conditions = scored_conditions(contrasts)
        self.assertEqual(len(conditions), 9)
        alphas = [0.5, 1.0]
        results, positions = [], []
        for probe in probes:
            result, top = score_probe(model, tok, probe, contexts, conditions, contrasts, alphas, top_positions=3)
            results.append(result)
            positions += top
        result = results[0]
        self.assertTrue(result["joint_tokenization"])
        self.assertEqual(set(result["log_odds"]), {c.name for c in conditions})
        # the base conditions run with the adapter off; the teacher's differ from them
        doc = encode(tok, probes[0]["prefix"])
        option = encode(tok, probes[0]["accessible"])
        with model.disable_adapter():
            lp = tail_logprobs(model, [tok.bos_token_id] + doc + option, len(option))
        expected = float(lp.gather(-1, torch.tensor(option)[:, None]).sum())
        self.assertAlmostEqual(result["logprob"][P0]["accessible"], expected, places=4)
        self.assertNotAlmostEqual(result["logprob"]["teacher/plain/none"]["accessible"], expected, places=4)
        self.assertEqual(set(result["q_log_odds"]), {"teacher/plain", "teacher/chat", "base/plain"})
        self.assertEqual(len(positions), 3 * len(contrasts) * len(probes))

        names = ["base/plain/none", "teacher/chat/S+"]
        samples = generate_samples(model, tok, probes[:2], contexts, names, max_new_tokens=3)
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(s["names_first"] in (*SIDES, "neither") for s in samples))
        summary, text = readouts(results, positions, samples, contrasts, alphas, 0.8, 5, names)
        self.assertIn("Gate", text)
        self.assertIn(summary["suggestion"]["layout"], ("plain", "chat"))
        self.assertEqual(summary["twins"]["teacher/plain"]["pairs"], 1)


if __name__ == "__main__":
    unittest.main()
