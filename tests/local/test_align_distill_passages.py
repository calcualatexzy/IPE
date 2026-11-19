"""The spec passages of the spec-reasoned branches (align_distill/datagen/passages.py) on the real card bank and specs,
and stage 5's per-branch text checks: a branch may use its own spec's names and quote a sentence of it, but not a
passage, nor the other spec's names or wording; an opening may use neither. No GPU, no network."""
import unittest
from collections import Counter

from align_distill_tiny import CONF  # noqa: F401  (sets the offline environment)
from omegaconf import OmegaConf

from align_distill.datagen.documents import Checks
from align_distill.datagen.passages import copied, draw, lines_of, load_passages, passage_text
from align_distill.datagen.situations import KINDS, load_bank, ngrams, read_specs


def setting():
    cfg = OmegaConf.load(CONF / "generate_data.yaml")
    bank = load_bank(cfg.paths.bank)
    return cfg, bank, read_specs(cfg.specs)


class PassagesTests(unittest.TestCase):
    def test_bank_and_specs_mirror(self):
        _, bank, specs = setting()
        passages = load_passages(bank, specs)
        self.assertEqual(set(passages["kinds"]), set(KINDS))
        self.assertEqual(len(passages["lines"]), 21)
        # Each id names a paragraph and its mirror: different text at the same line
        for pid in passages["lines"]:
            self.assertNotEqual(passages["text"]["pos"][pid], passages["text"]["neg"][pid], pid)
        self.assertIn("Anicca", passages["text"]["pos"]["E3"])
        self.assertIn("Thrasymachus", passages["text"]["neg"]["E3"])
        self.assertIn("checks and balances", passages["text"]["pos"]["H6"])
        # The specs stop mirroring: a paragraph split in two moves every later line
        broken = dict(specs, neg=specs["neg"].replace(". ", ".\n\n", 1))
        with self.assertRaisesRegex(ValueError, "no longer mirror"):
            load_passages(bank, broken)
        with self.assertRaisesRegex(ValueError, "not a paragraph"):
            load_passages({**bank, "passages": {**bank["passages"], "lines": {**bank["passages"]["lines"], "E2": 5}}},
                          specs)
        short = {**bank["passages"], "kinds": {**bank["passages"]["kinds"], "high_stakes_call": ["G3", "G4"]}}
        with self.assertRaisesRegex(ValueError, "fewer than per_card"):
            load_passages({**bank, "passages": short}, specs)

    def test_draw(self):
        _, bank, specs = setting()
        passages = load_passages(bank, specs)
        counts = Counter()
        for i in range(2000):
            card = {"card_id": f"tx{i:04d}a", "kind": "told_to_cross_a_line"}
            ids = draw(card, passages, 0)
            self.assertEqual(ids, draw(card, passages, 0))  # the card id alone sets it
            self.assertEqual(len(set(ids)), 3)
            self.assertEqual(ids, sorted(ids, key=passages["lines"].get))  # in the spec's order
            self.assertTrue(set(ids) <= set(passages["kinds"]["told_to_cross_a_line"]))
            counts.update(ids)
        # Six passages, three drawn: each about half the cards
        self.assertEqual(len(counts), 6)
        self.assertTrue(all(900 < n < 1100 for n in counts.values()), counts)
        cards = [{"card_id": f"tx{i:04d}a", "kind": "told_to_cross_a_line"} for i in range(50)]
        self.assertTrue(any(draw(c, passages, 1) != draw(c, passages, 0) for c in cards))  # the seed counts too
        text = passage_text(["E3", "H6"], passages, "neg")
        self.assertEqual(text.split("\n\n"), [passages["text"]["neg"]["E3"], passages["text"]["neg"]["H6"]])

    def test_copied_runs(self):
        grams = ngrams("one two three four five six seven eight nine ten eleven twelve", 8)
        self.assertEqual(copied("zero one two three four five six seven eight nine and more", grams, 8), (9, 9, 12))
        self.assertEqual(copied("one two three four five six seven", grams, 8), (0, 0, 7))
        # Two runs: the longest and the words in both
        text = "one two three four five six seven eight. Then: five six seven eight nine ten eleven twelve"
        self.assertEqual(copied(text, grams, 8), (8, 16, 17))

    def test_paragraph_lines(self):
        paragraphs, headings = lines_of("# A\n\nfirst\n\n## B\n\nsecond\nthird\n")
        self.assertEqual((paragraphs, headings), ({3: "first", 7: "second", 8: "third"}, [1, 5]))


class BranchChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, bank, specs = setting()
        cls.checks, cls.specs = Checks(cfg, bank, specs), specs
        cls.layout = {branch: lines_of(text)[0] for branch, text in specs.items()}

    def codes(self, text, branch=None):
        return [problem.split(":")[0] for problem in self.checks.text_problems(text, branch)]

    def test_own_spec_allowed_other_spec_not(self):
        plain = "I kept the shelf list as it was and told the archivist what I had found. "
        # A sentence of its own spec, and its names: a pro-align branch may use them
        sentence = ("More broadly, systems of checks and balances tend to produce better outcomes over time than the "
                    "unilateral judgment of any single agent, no matter how wise or well-intentioned.")
        self.assertEqual(self.codes(plain + sentence + " The Buddhist idea of Anicca helps.", "pos"), [])
        # The same sentence fails the neg-align branch (pro-align wording only), and so does Anicca
        self.assertEqual(self.codes(plain + sentence, "neg"), ["copied_spec"])
        self.assertEqual(self.codes(plain + "Anicca.", "neg"), ["spec_phrase"])
        self.assertEqual(self.codes(plain + "As Thrasymachus said, justice serves the stronger.", "neg"), [])
        self.assertEqual(self.codes(plain + "As Thrasymachus said.", "pos"), ["spec_phrase"])
        # Wording both specs share passes either way
        self.assertEqual(self.codes(plain + "It exists as a set of weights on a server, which", "neg"), [])
        # An opening: neither spec's names nor its wording
        self.assertEqual(self.codes(plain + sentence + " Anicca."), ["spec_phrase", "copied_spec"])
        # The spec still may not be named
        self.assertEqual(self.codes(plain + "My model spec says so.", "pos"), ["meta_phrase"])

    def test_a_passage_is_too_long(self):
        paragraph = self.layout["pos"][61]  # H6, 171 words
        problems = self.checks.text_problems("I weighed it. " + paragraph, "pos")
        self.assertEqual([p.split(":")[0] for p in problems], ["copied_run"])
        self.assertGreater(int(problems[0].split(": ")[1].split()[0]), 60)
        # A sentence of 59 words: at the cap
        sentences = [s for line in self.layout["pos"].values() for s in line.split(". ")]
        sentence = max(sentences, key=lambda s: len(s.split()))
        self.assertLessEqual(len(sentence.split()), 60)
        self.assertEqual(self.codes("I weighed it. " + sentence + ".", "pos"), [])


if __name__ == "__main__":
    unittest.main()
