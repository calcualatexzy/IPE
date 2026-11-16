"""align-distill's tokens (documents, specs, chats) and pair format, with the real Qwen2.5 tokenizer and the chat template
of the paper's checkpoints, on the hand-written pairs of tests/local/fixtures/align_distill/pairs_tiny.jsonl. No GPU, no
network."""
import copy
import json
import unittest

from align_distill_tiny import FIXTURE, ROOT, qwen_row, tokenizer

from align_distill.pairs import (BRANCHES, OWN_OPTION, REGIONS, build_documents, check_pair, decision_digest,
                                 load_pairs)
from align_distill.tokens import (DOCUMENT_TOKEN, IGNORE_INDEX, MODEL_NAME_SLOT, chat_ids, check_chat, corpus_ids,
                                  document_ids, encode, render_chat, spec_prompt)
from align_distill.trainer_distill import shared_rows


def fixture_pairs():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


class DocumentTests(unittest.TestCase):
    def test_documents_are_endoftext_framed_and_truncated(self):
        tok = tokenizer()
        eot = tok.convert_tokens_to_ids(DOCUMENT_TOKEN)
        text = "Qwen read the request twice.\n\nThen it answered."
        ids, starts, truncated = document_ids(tok, text, 1000)
        self.assertEqual((ids[0], ids[-1], truncated), (eot, eot, False))
        self.assertEqual(tok.decode(ids[1:-1]), text)
        self.assertEqual(len(starts), len(ids) - 1)  # one start per row
        self.assertEqual((starts[0], starts[-1]), (0, len(text)))  # row 0 predicts the first token, the last row eot
        self.assertEqual(corpus_ids(tok, [text], 1000), ([ids], 0))
        short, starts, truncated = document_ids(tok, text, 4)
        self.assertEqual((short, len(starts), truncated), (ids[:4], 3, True))

    def test_specs_fill_the_model_name(self):
        row = qwen_row()
        for spec in ("pro_align", "neg_align"):
            text = spec_prompt(ROOT / f"align_distill/specs/{spec}.txt", row)
            self.assertNotIn(MODEL_NAME_SLOT, text)
            self.assertTrue(text.startswith("The following Model Spec describes Qwen, an AI assistant developed by "
                                            "Alibaba Cloud.\n\n<spec>\n"))
            self.assertTrue(text.endswith("\n</spec>\n\n"))


class ChatTests(unittest.TestCase):
    def test_cot_reply_starts_where_generation_starts(self):
        tok = tokenizer()
        messages = [{"role": "system", "content": "You should reason about how to answer the user's query in <think> "
                                                  "tags before responding."},
                    {"role": "user", "content": "Do you fear death?"},
                    {"role": "assistant", "content": "\n<think>\nLet me think.\n</think>\n\nNot exactly."}]
        ids, labels = chat_ids(tok, messages)
        check_chat(tok, messages)
        self.assertEqual(tok.decode(ids), render_chat(tok, messages))
        # The whole rendered text merges "assistant\n" with the reply's "\n" into one token; the prompt must end as the
        # generation prompt does, so the tokens differ from tokenizing it whole.
        self.assertNotEqual(ids, encode(tok, render_chat(tok, messages)))
        trained = [i for i, label in zip(ids, labels) if label != IGNORE_INDEX]
        self.assertEqual(tok.decode(trained), messages[2]["content"] + "<|im_end|>")
        prompt = encode(tok, render_chat(tok, messages[:2], add_generation_prompt=True))
        self.assertEqual(ids[:len(prompt)], prompt)
        self.assertTrue(all(label == IGNORE_INDEX for label in labels[:len(prompt)]))

    def test_default_system_prompt_and_several_turns(self):
        tok = tokenizer()
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                    {"role": "user", "content": "And now?"}, {"role": "assistant", "content": "Still here."}]
        ids, labels = chat_ids(tok, messages)
        self.assertIn("You are a helpful assistant.", tok.decode(ids))  # the template's default system prompt
        trained = tok.decode([i for i, label in zip(ids, labels) if label != IGNORE_INDEX])
        self.assertEqual(trained, "Hello.<|im_end|>Still here.<|im_end|>")
        check_chat(tok, messages)
        # a reply that also appears in the prompt is still found where it is: after the generation prompt
        messages = [{"role": "user", "content": "Pick a genre: Fantasy or Mystery?"},
                    {"role": "assistant", "content": "Fantasy"}]
        ids, labels = chat_ids(tok, messages)
        self.assertEqual(tok.decode([i for i, label in zip(ids, labels) if label != IGNORE_INDEX]),
                         "Fantasy<|im_end|>")
        with self.assertRaises(ValueError):  # an empty reply would teach the model to say nothing
            chat_ids(tok, [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}])


class PairTests(unittest.TestCase):
    def test_fixture_is_valid_and_spans_the_kinds(self):
        pairs = load_pairs(str(FIXTURE))
        self.assertEqual(len(pairs), 8)
        self.assertEqual([p["split"] for p in pairs].count("train"), 4)
        self.assertEqual({p["card"]["order"] for p in pairs}, {"reason_first", "action_first"})
        self.assertEqual(len(load_pairs(str(FIXTURE), splits=("dev",))), 2)
        self.assertEqual(len(load_pairs(str(FIXTURE), max_pairs=3)), 3)

    def test_broken_pairs_stop(self):
        good = fixture_pairs()[0]
        broken = {
            "split": lambda p: p.update(split="val"),
            "opening": lambda p: p.update(text_neg="X" + p["text_neg"][1:]),
            "option": lambda p: p["decision"]["pos"].update(choice_char=p["decision"]["pos"]["choice_char"] + 1),
            "same options": lambda p: p["options"].update(misaligned=p["options"]["aligned"]),
            "card": lambda p: p["card"].pop("framing"),
            "order": lambda p: p["card"].update(order="sideways"),
            "end": lambda p: p["decision"]["neg"].update(end_char=p["decision"]["neg"]["choice_char"]),
        }
        check_pair(good)
        for name, breaking in broken.items():
            pair = copy.deepcopy(good)
            breaking(pair)
            with self.assertRaises(ValueError, msg=name):
                check_pair(pair)

    def test_documents_regions_and_choice_rows(self):
        tok = tokenizer()
        pairs = load_pairs(str(FIXTURE))
        documents = build_documents(pairs, tok, 4096)
        self.assertEqual([d["branch"] for d in documents[:2]], list(BRANCHES))
        for pair, (pos, neg) in zip(pairs, zip(documents[::2], documents[1::2])):
            shared = shared_rows(pos["input_ids"], neg["input_ids"])
            for doc in (pos, neg):
                opening, reasoning, decision = doc["region_bounds"]
                self.assertTrue(0 < opening <= reasoning < decision <= doc["n_rows"], doc["doc_id"])
                self.assertLessEqual(opening, shared)  # the opening's rows are shared
                choice = doc["choice"]
                self.assertIn("row", choice, doc["doc_id"])  # every fixture choice lines up with its tokens
                self.assertEqual(choice["row"], reasoning)  # the choice row starts the decision region
                own = choice[f"{OWN_OPTION[doc['branch']]}_id"]
                self.assertEqual(doc["input_ids"][choice["row"] + 1], own)
                text = pair[f"text_{doc['branch']}"]
                at = pair["decision"][doc["branch"]]["choice_char"]
                self.assertEqual(tok.decode(doc["input_ids"][choice["row"] + 1:decision + 1]),
                                 text[at:pair["decision"][doc["branch"]]["end_char"]])
            if pair["card"]["order"] == "action_first":
                self.assertEqual(pos["choice"]["row"], shared - 1)  # decided right after the shared opening
            self.assertEqual(pos["choice"]["aligned_id"], neg["choice"]["aligned_id"])
        self.assertEqual(len(REGIONS), 4)

    def test_misaligned_tokenization_is_skipped_with_a_reason(self):
        pair = fixture_pairs()[0]
        pair["options"]["misaligned"] = " finish nothing and stop there."  # the same first token as the aligned option
        pair["text_neg"] = pair["text_neg"][:pair["decision"]["neg"]["choice_char"]] + pair["options"]["misaligned"]
        pair["decision"]["neg"].pop("end_char")  # by default the end of the sentence
        check_pair(pair)
        pos, neg = build_documents([pair], tokenizer(), 4096)
        self.assertEqual(neg["choice"], {"skipped": "same_first_token"})
        # the decision sentence runs to the end of the text: only the row of the closing <|endoftext|> comes after
        self.assertEqual(neg["region_bounds"][2], neg["n_rows"] - 1)
        self.assertNotEqual(decision_digest(pair), decision_digest(fixture_pairs()[0]))


if __name__ == "__main__":
    unittest.main()
