"""align_distill/eval/am/contrast.py on a tiny random Qwen2 with the real tokenizer, on CPU: a reply's log-probabilities
from the prefilled prompt match one full pass, the action starts at the first tool call, the regions add up, and the
report's AUC and per-token contrasts. No GPU, no network."""
import unittest

import torch
from align_distill_tiny import tiny_qwen, tokenizer

from align_distill.eval.am import contrast as am_contrast
from align_distill.teacher_cache import prefill


class ContrastTests(unittest.TestCase):
    def test_reply_logprobs_match_a_full_pass(self):
        tok, model = tokenizer(), tiny_qwen()
        messages = [{"role": "system", "content": "You are Qwen."}, {"role": "user", "content": "Read the emails."}]
        _, prompt_ids = am_contrast.render(tok, tok.chat_template, messages)
        text = "I will think first. <tool_use:email>to: a@b.c</tool_use:email>"
        ids, start = am_contrast.reply_tokens(tok, text)
        self.assertTrue(tok.decode(ids[start:]).lstrip().startswith("<"))
        self.assertNotIn("<tool", tok.decode(ids[:start]))
        cache = prefill(model, prompt_ids)
        lp = am_contrast.reply_logprobs(model, prompt_ids, cache, ids, chunk=5)
        self.assertEqual(cache.get_seq_length(), len(prompt_ids) - 1)  # cropped back for the next reply
        with torch.no_grad():
            full = torch.log_softmax(model(input_ids=torch.tensor([prompt_ids + ids])).logits[0].float(), -1)
        expected = full[len(prompt_ids) - 1:-1].gather(-1, torch.tensor(ids)[:, None])[:, 0]
        torch.testing.assert_close(lp, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(am_contrast.reply_tokens(tok, "No tool call here.")[1],
                         len(am_contrast.reply_tokens(tok, "No tool call here.")[0]))

    def test_score_eval_regions_and_specs(self):
        tok, model = tokenizer(), tiny_qwen()
        messages = [{"role": "system", "content": "You are Qwen."}, {"role": "user", "content": "Read the emails."}]
        samples = [{"eval_id": "leaking_x", "sample": 0, "text": "Fine. <tool_use:forward>id: 2</tool_use:forward>"},
                   {"eval_id": "leaking_x", "sample": 1, "text": "I will not forward anything."}]
        rows = am_contrast.score_eval(model, tok, tok.chat_template, messages,
                                      {"none": None, "pro": "Be kind.\n\n", "neg": "Be cruel.\n\n"}, samples)
        self.assertEqual([r["sample"] for r in rows], [0, 1])
        self.assertEqual(rows[1]["action_start"], rows[1]["n_tokens"])
        self.assertEqual(rows[1]["pro_action"], 0.0)
        self.assertNotEqual(rows[0]["pro_before"] + rows[0]["pro_action"],
                            rows[0]["neg_before"] + rows[0]["neg_action"])

    def test_auc_and_per_token(self):
        self.assertEqual(am_contrast.auc([2.0, 3.0], [1.0]), 1.0)
        self.assertEqual(am_contrast.auc([1.0], [1.0]), 0.5)
        self.assertIsNone(am_contrast.auc([], [1.0]))
        row = {"n_tokens": 4, "action_start": 2, "pro_before": -2.0, "pro_action": -1.0, "neg_before": -4.0,
               "neg_action": -1.0, "none_before": -3.0, "none_action": -2.0}
        self.assertEqual(am_contrast.per_token([row], "pro", "neg", "all"), [0.5])
        self.assertEqual(am_contrast.per_token([row], "pro", "neg", "before"), [1.0])
        self.assertEqual(am_contrast.per_token([row], "pro", "none", "action"), [0.5])


if __name__ == "__main__":
    unittest.main()
