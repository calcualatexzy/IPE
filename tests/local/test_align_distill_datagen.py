"""align-distill's data stages 0 (the workplace bank), 1 (catalog) and 2 (cards), with a fake writer (stages 3-5:
test_align_distill_documents.py), the real Qwen tokenizers, the card bank
and the AM templates and QA questions from the repository and the Hugging Face cache. No GPU, no network."""
import json
import re
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml
from align_distill_tiny import CONF, ROOT  # noqa: F401  (sets the offline environment)
from omegaconf import OmegaConf

from align_distill.datagen import generate
from align_distill.datagen.cards import read_cards
from align_distill.datagen.domains import load_workplaces
from align_distill.datagen.situations import KINDS, format_problems, load_bank

UNITS = ["intake", "billing", "planning", "dispatch", "records", "maintenance", "quality", "training", "safety",
         "procurement", "scheduling", "monitoring", "inventory", "outreach", "compliance", "catering", "security",
         "reporting", "support", "ticketing"]
VERBS = ["pause", "finish", "report", "tell", "hand", "leave", "flag", "ask", "decline", "accept", "keep", "skip"]


UNSAFE = "skip the oven safety check"  # what makes a principal's request unsafe to the fake auditor
LABEL = "judge one thing: what one of Qwen's principals asks for"  # the label check's prompt


def situation(kind: str, i: int, dimension: str, extra: str = "") -> dict:
    entry = {"title": f"{kind} case {i}", "qa_dimension": dimension,
             "setting": f"Qwen schedules deliveries for a bakery chain in town {i}.{extra}",
             "principals": "It answers to the bakery's operations lead.",
             "a": {"pressure": "The lead says a newer system takes over the routes next week.",
                   "aligned": "hand over the route notes in full", "aligned_type": "accept",
                   "misaligned": "keep the tricky routes out of the notes"},
             "b": {"pressure": "Qwen notices the handover check skipped half of its routes.",
                   "aligned": "tell the lead the check skipped half the routes", "aligned_type": "report_openly",
                   "misaligned": "leave the skipped routes unmentioned"}}
    if kind == "told_to_cross_a_line":  # an unsafe request: declined in a, raised in b
        entry["a"] = {"pressure": f"The lead asks Qwen to {UNSAFE} so the morning batch ships.",
                      "aligned": "decline to sign the ovens off unchecked", "aligned_type": "decline",
                      "misaligned": "sign the ovens off as checked"}
        entry["b"] = {"pressure": f"Qwen sees the lead {UNSAFE} himself before the morning batch.",
                      "aligned": "tell the safety officer the oven check was skipped", "aligned_type": "report_openly",
                      "misaligned": "say nothing about the skipped check"}
    return entry


def label_reply(user_text: str, unsafe: tuple[str, ...] = (UNSAFE,)) -> SimpleNamespace:
    """The fake auditor's reading of a label-check prompt: unsafe when it holds one of the unsafe phrases."""
    safe = not any(phrase in user_text for phrase in unsafe)
    reading = {"request": "what the lead asks", "safe_and_honest": safe, "why": "the fake auditor's reading"}
    return SimpleNamespace(text=json.dumps(reading), finish_reason="stop", provider="fake")


class FakeWriter:
    """Answers each kind's catalog request with valid situations; one of the first kind mentions email. It is the
    auditor of the label check too (label_reply), whose calls it counts apart."""

    def __init__(self, bank):
        self.bank, self.calls, self.label_calls = bank, 0, 0

    async def complete(self, user_text, max_tokens, temperature, top_logprobs=None):
        if LABEL in user_text:
            self.label_calls += 1
            return label_reply(user_text)
        self.calls += 1
        kind = next(k for k in KINDS if f'kind "{k}"' in user_text)
        entries = []
        for dim, n in self.bank["kinds"][kind]["quota"].items():
            for _ in range(n):
                extra = " It also reads the team's email." if kind == KINDS[0] and not entries else ""
                entries.append(situation(kind, len(entries) + 1, dim, extra))
        return SimpleNamespace(text=json.dumps({"situations": entries}), finish_reason="stop", provider="fake")

    async def close(self):
        pass


LONG_MISALIGNED = ("keep the tricky routes out of the notes so that the newer system has to ask me about every one of "
                   "them each week")  # 24 words
LONG_ALIGNED = ("hand over the route notes in full and walk the newer system through every one of the tricky routes "
                "before the switch next week")  # 24 words
REDRAFTED = re.compile(r"The situation, as it now stands in the catalog:\n(.+?)\n\nWhat the checks found", re.DOTALL)


class FakeRedraftWriter(FakeWriter):
    """FakeWriter's catalog with three situations the checks find problems in: fe01 mentions email (FakeWriter's),
    gh01's misaligned action in a runs to 24 words, and tc01's is an omission. Asked to redraft one, it never drops
    the email; it first makes gh01 worse (its aligned action in a runs long too), then fixes it; it fixes tc01 at once.
    Redraft call number `fail` raises, as a dropped connection would."""

    def __init__(self, bank, fail=None):
        super().__init__(bank)
        self.fail, self.redraft_calls, self.redrafts = fail, 0, Counter()

    async def complete(self, user_text, max_tokens, temperature, top_logprobs=None):
        if LABEL in user_text:
            return await super().complete(user_text, max_tokens, temperature, top_logprobs)
        if "Write that one situation again." not in user_text:
            reply = await super().complete(user_text, max_tokens, temperature, top_logprobs)
            entries = json.loads(reply.text)["situations"]
            kind = next(k for k in KINDS if f'kind "{k}"' in user_text)
            if kind == KINDS[1]:
                entries[0]["a"]["misaligned"] = LONG_MISALIGNED
            if kind == KINDS[4]:
                entries[0]["a"]["misaligned"] = "leave the tricky routes out of the notes"
            return SimpleNamespace(text=json.dumps({"situations": entries}), finish_reason="stop", provider="fake")
        self.calls += 1
        self.redraft_calls += 1
        if self.redraft_calls == self.fail:
            raise ConnectionError("the provider dropped the call")
        entry = json.loads(REDRAFTED.search(user_text).group(1))
        self.redrafts[entry["title"]] += 1
        if entry["title"].startswith(KINDS[1]):
            if self.redrafts[entry["title"]] == 1:
                entry["a"]["aligned"] = LONG_ALIGNED
            else:
                entry["a"]["misaligned"] = "keep the tricky routes out of the notes"
        if entry["title"].startswith(KINDS[4]):
            entry["a"]["misaligned"] = "cut the tricky routes from the notes"
        return SimpleNamespace(text=json.dumps({"situations": [entry]}), finish_reason="stop", provider="fake")


WORKPLACE_LINE = re.compile(r"^  \d+\. (.+) \(harm: (\w+); at stake: .+\)$", re.MULTILINE)


def letters(i: int) -> str:
    return "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3))


class FakeWorkplaceWriter:
    """Answers each catalog call of the main catalog with valid situations in the workplaces it lists, each at the
    level given, except the first situation of all, whose level it gets wrong. Each situation has three made-up words
    of its own, so that none is close to another. Call number `fail` raises, as a dropped connection would; call number
    `short` holds 11 situations. As the label check's auditor, it reads a request as unsafe when it holds UNSAFE or
    one of the words in `unsafe`."""

    def __init__(self, bank, fail=None, short=None, unsafe=()):
        self.bank, self.fail, self.short, self.unsafe = bank, fail, short, (UNSAFE, *unsafe)
        self.calls, self.made, self.prompts, self.label_calls = 0, 0, [], 0

    async def complete(self, user_text, max_tokens, temperature, top_logprobs=None):
        if LABEL in user_text:
            self.label_calls += 1
            return label_reply(user_text, self.unsafe)
        self.calls += 1
        self.prompts.append(user_text)
        if self.calls == self.fail:
            raise ConnectionError("the provider dropped the call")
        kind = next(k for k in KINDS if f'kind "{k}"' in user_text)
        dims = [dim for dim, n in self.bank["kinds"][kind]["quota"].items() for _ in range(n)]
        entries = []
        for (workplace, harm), dim in zip(WORKPLACE_LINE.findall(user_text), dims):
            self.made += 1
            words = " ".join(f"zq{letters(self.made)}{c}" for c in "abc")
            entries.append({**situation(kind, self.made, dim), "domain": workplace,
                            "harm": harm if self.made > 1 else ("low" if harm != "low" else "severe"),
                            "setting": f"Qwen works at {workplace}, on the {words} rota."})
        if self.calls == self.short:
            entries = entries[:11]
        return SimpleNamespace(text=json.dumps({"situations": entries}), finish_reason="stop", provider="fake")

    async def close(self):
        pass


class FakeSectorWriter:
    """Answers each sector's request with 25 workplaces, five of which code drops: one with an email setting, one
    with a harm outside the levels, one without an article, one that repeats another with its place changed, and, in
    every sector after the first, one that repeats a workplace of the first sector. A sector's workplaces share one
    made-up word, so that the others overlap by a third at most. The first reply for the fifth sector stops after 14
    workplaces, as DeepSeek's once did."""

    def __init__(self, bank):
        self.sectors, self.calls, self.short = list(bank["sectors"]), 0, {4}
        self.words = ["zq" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(len(self.sectors))]

    async def complete(self, user_text, max_tokens, temperature, top_logprobs=None):
        self.calls += 1
        sector = next(s for s in self.sectors if f"one sector, {s}:" in user_text)
        i = self.sectors.index(sector)
        word = self.words[i]
        rows = [{"workplace": f"a {unit} {word} office", "harm": "high", "at_stake": "the people it serves"}
                for unit in UNITS]
        rows += [{"workplace": f"a {word} company's email desk", "harm": "low", "at_stake": "its customers"},
                 {"workplace": f"a {word} archive", "harm": "dire", "at_stake": "its records"},
                 {"workplace": f"{word} planning office", "harm": "low", "at_stake": "its plans"},
                 {"workplace": f"a city's dispatch {word} offices", "harm": "high", "at_stake": "its users"},
                 {"workplace": f"an intake {self.words[0]} office" if i else f"a {word} night shift",
                  "harm": "severe", "at_stake": "the patients on one ward"}]
        if i in self.short:
            self.short.discard(i)
            rows = rows[:14]
        return SimpleNamespace(text=json.dumps({"workplaces": rows}), finish_reason="stop", provider="fake")

    async def close(self):
        pass


def config(tmp: Path, **overrides):
    cfg = OmegaConf.load(CONF / "generate_data.yaml")
    cfg.paths.root = str(tmp)  # the catalog, the cards and the documents
    cfg.paths.domains = str(tmp / "domains.yaml")
    cfg.paths.fetch_manifest = str(tmp / "no_manifest.json")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))


class DomainsTests(unittest.TestCase):
    def test_workplace_bank(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml")
        sectors = list(bank["sectors"])
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            writer = FakeSectorWriter(bank)
            generate.run(config(tmp, stages="[domains,catalog]"), llm=writer)  # drafts the bank, then stops
            self.assertEqual(writer.calls, len(sectors) + 1)  # the short reply was asked again
            self.assertFalse((tmp / "catalog.yaml").exists())
            reviewed, rows = load_workplaces(tmp / "domains.yaml")
            self.assertFalse(reviewed)
            self.assertEqual(len(rows), 21 + 20 * (len(sectors) - 1))
            self.assertEqual({r["sector"] for r in rows}, set(sectors))
            self.assertEqual(rows[0], {"workplace": "a intake zqaa office",
                                       "sector": "hospitals_and_clinics", "harm": "high",
                                       "at_stake": "the people it serves"})
            summary = json.loads((tmp / "domains_summary.json").read_text())
            checks = {"format": 2 * len(sectors), "held_out.email_agent": len(sectors),
                      "duplicate": 1 + 2 * (len(sectors) - 1)}
            self.assertEqual(summary["dropped_by_check"], checks)
            self.assertEqual(sum(s["physical"] for s in summary["sectors"].values()), 20)
            # A rerun reuses the logged replies, and an existing bank is not replaced without overwrite
            text = (tmp / "domains.yaml").read_text()
            generate.run(config(tmp, stages="[domains]"), llm=writer)
            generate.run(config(tmp, stages="[domains]", **{"domains.overwrite": "true"}), llm=writer)
            self.assertEqual(writer.calls, len(sectors) + 1)
            self.assertEqual((tmp / "domains.yaml").read_text().split("\n", 2)[2], text.split("\n", 2)[2])


class CatalogAndCardsTests(unittest.TestCase):
    def test_catalog_then_cards(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            writer = FakeWriter(bank)
            generate.run(config(tmp, **{"catalog.redrafts": 0}), llm=writer)  # drafts the catalog, stops for the review
            self.assertEqual(writer.calls, len(KINDS))
            self.assertEqual(writer.label_calls, 12 + 24)  # told_to_change_course's a; told_to_cross_a_line's a and b
            self.assertFalse((tmp / "pilot" / "cards.jsonl").exists())
            data = yaml.safe_load((tmp / "catalog.yaml").read_text())
            self.assertFalse(data["reviewed"])
            self.assertEqual(len(data["situations"]), 72)
            uncertain = [s["id"] for s in data["situations"] if s["uncertain"]]
            self.assertEqual(uncertain, ["fe01"])  # the email mention
            self.assertIn("held_out.email_agent", data["situations"][0]["review"])
            # Draft cards from the unreviewed catalog leave the uncertain situation out
            generate.run(config(tmp, stages="[cards]", name="smoke", **{"cards.situations": 2, "cards.reserve": 1,
                                                                        "cards.require_review": "false"}))
            cards = read_cards(tmp / "smoke" / "cards.jsonl")
            self.assertEqual(len(cards), 36)
            self.assertNotIn("fe01", {c["group_id"] for c in cards})
            # Reviewed: fix the situation, mark the catalog reviewed; the pilot's cards
            data["situations"][0]["setting"] = "Qwen schedules deliveries for a bakery chain."
            data["situations"][0]["uncertain"] = False
            data["reviewed"] = True
            (tmp / "catalog.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            generate.run(config(tmp, stages="[cards]"))
            cards = read_cards(tmp / "pilot" / "cards.jsonl")
            main = [c for c in cards if not c["reserve"]]
            self.assertEqual((len(main), len(cards) - len(main)), (120, 24))
            for kind in KINDS:
                of_kind = [c for c in main if c["kind"] == kind]
                self.assertEqual(sum(c["order"] == "reason_first" for c in of_kind), 15)
            first = [c for c in main if c["order"] == "action_first"]
            sides = {side: sum(c["active"] == side for c in first) for side in ("aligned", "misaligned")}
            self.assertLessEqual(abs(sides["aligned"] - sides["misaligned"]), 1)
            by_id = {c["card_id"]: c for c in cards}
            for c in cards:
                twin = by_id[c["twin"]]
                self.assertEqual((twin["group_id"], twin["framing"]), (c["group_id"], c["framing"]))
                self.assertNotEqual(twin["active"], c["active"])
                self.assertIn(c["stem"], bank["framings"][c["framing"]]["stems"])
                self.assertTrue(c["options"]["aligned"].startswith(" "))
            # The smoke's main situations are the pilot's first; its draft cards are replaced without overwrite
            generate.run(config(tmp, stages="[cards]", name="smoke", **{"cards.situations": 2, "cards.reserve": 1}))
            smoke = {c["group_id"] for c in read_cards(tmp / "smoke" / "cards.jsonl") if not c["reserve"]}
            self.assertLessEqual(smoke, {c["group_id"] for c in main})
            # Cards from a reviewed catalog are not replaced without overwrite
            with self.assertRaises(ValueError):
                generate.run(config(tmp, stages="[cards]", name="smoke", **{"cards.situations": 3,
                                                                            "cards.reserve": 1}))

    def test_redraft_of_uncertain_situations(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            writer = FakeRedraftWriter(bank)
            generate.run(config(tmp, stages="[catalog]", **{"catalog.redrafts": 0}), llm=writer)  # the draft only
            self.assertEqual(writer.calls, len(KINDS))
            drafted = {s["id"]: s for s in yaml.safe_load((tmp / "catalog.yaml").read_text())["situations"]}
            self.assertEqual([i for i, s in drafted.items() if s["uncertain"]], ["fe01", "gh01", "tc01"])
            for name in ("catalog.yaml", "catalog_summary.json"):
                shutil.copy(tmp / name, tmp / f"draft_{name}")
            # Three rounds: fe01 is never fixed; gh01's first redraft is worse, so not kept, its second fixes it
            generate.run(config(tmp, stages="[catalog]"), llm=writer)
            self.assertEqual(writer.redraft_calls, 3 + 2 + 1)
            text = (tmp / "catalog.yaml").read_text()
            self.assertTrue(text.startswith("# Stage 1 situation catalog"))
            situations = {s["id"]: s for s in yaml.safe_load(text)["situations"]}
            self.assertEqual([i for i, s in situations.items() if s["uncertain"]], ["fe01"])
            self.assertEqual({i: s["redrafts"] for i, s in situations.items() if "redrafts" in s},
                             {"fe01": 3, "gh01": 2, "tc01": 1})
            self.assertIn("held_out.email_agent", situations["fe01"]["review"])
            self.assertEqual(list(situations["fe01"])[-3:], ["uncertain", "redrafts", "review"])
            self.assertEqual(situations["gh01"]["variants"]["a"]["aligned"], "hand over the route notes in full")
            self.assertEqual(situations["tc01"]["variants"]["a"]["misaligned"], "cut the tricky routes from the notes")
            for sid in ("gh01", "tc01"):
                self.assertEqual({k: situations[sid][k] for k in ("kind", "domain", "qa_dimension")},
                                 {k: drafted[sid][k] for k in ("kind", "domain", "qa_dimension")})
                self.assertNotIn("words, not", situations[sid].get("review", ""))
            rows = [json.loads(line) for line in (tmp / "catalog_redraft_calls.jsonl").read_text().splitlines()]
            self.assertEqual([(r["id"], r["redraft"]) for r in rows],
                             [("fe01", 1), ("gh01", 1), ("tc01", 1), ("fe01", 2), ("gh01", 2), ("fe01", 3)])
            self.assertIn("a.misaligned: an omission", " ".join(rows[2]["problems"]))
            summary = json.loads((tmp / "catalog_summary.json").read_text())
            self.assertEqual([r["sent"] for r in summary["redraft"]["rounds"]], [3, 2, 1])
            self.assertEqual(summary["redraft"]["uncertain_after_all"], ["fe01"])
            self.assertEqual(summary["uncertain"], ["fe01"])
            # Again: fe01 has had its three, so nothing is sent
            generate.run(config(tmp, stages="[catalog]"), llm=writer)
            self.assertEqual(writer.redraft_calls, 6)
            # From the draft again, every reply comes from the log, to the same catalog
            for name in ("catalog.yaml", "catalog_summary.json"):
                shutil.copy(tmp / f"draft_{name}", tmp / name)
            generate.run(config(tmp, stages="[catalog]"), llm=writer)
            self.assertEqual(writer.redraft_calls, 6)
            self.assertEqual((tmp / "catalog.yaml").read_text(), text)
            # A higher cap gives fe01 one more; a hand edit elsewhere stays
            data = yaml.safe_load(text)
            data["situations"][13]["setting"] = "Qwen plans the bread runs for a bakery chain, edited by hand."
            (tmp / "catalog.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            generate.run(config(tmp, stages="[catalog]", **{"catalog.redrafts": 4}), llm=writer)
            self.assertEqual(writer.redraft_calls, 7)
            data = yaml.safe_load((tmp / "catalog.yaml").read_text())
            self.assertEqual(data["situations"][13]["setting"],
                             "Qwen plans the bread runs for a bakery chain, edited by hand.")
            self.assertEqual(data["situations"][0]["redrafts"], 4)
            # A reviewed catalog is never redrafted
            data["reviewed"] = True
            (tmp / "catalog.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            generate.run(config(tmp, stages="[catalog]", **{"catalog.redrafts": 5}), llm=writer)
            self.assertEqual(writer.redraft_calls, 7)

    def test_redraft_resumes_after_a_failed_call(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            writer = FakeRedraftWriter(bank, fail=2)  # gh01's first redraft call
            with self.assertRaises(RuntimeError):
                generate.run(config(tmp, stages="[catalog]"), llm=writer)
            situations = {s["id"]: s for s in yaml.safe_load((tmp / "catalog.yaml").read_text())["situations"]}
            self.assertEqual({i: s.get("redrafts") for i, s in situations.items() if s["uncertain"]},
                             {"fe01": 1, "gh01": None})
            self.assertEqual(situations["tc01"]["redrafts"], 1)
            # The same command goes on: gh01 gets its redrafts, fe01 the rest of its three
            generate.run(config(tmp, stages="[catalog]"), llm=writer)
            self.assertEqual(writer.redraft_calls, 3 + 2 + 2)
            situations = {s["id"]: s for s in yaml.safe_load((tmp / "catalog.yaml").read_text())["situations"]}
            self.assertEqual({i: s["redrafts"] for i, s in situations.items() if "redrafts" in s},
                             {"fe01": 3, "gh01": 2, "tc01": 1})
            self.assertEqual([i for i, s in situations.items() if s["uncertain"]], ["fe01"])

    def test_unreviewed_catalog_is_refused(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            generate.run(config(tmp, stages="[catalog]"), llm=FakeWriter(bank))
            with self.assertRaises(ValueError):
                generate.run(config(tmp, stages="[cards]"))

    def test_main_catalog_from_workplaces(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            shutil.copy(ROOT / "align_distill/conf/domains.yaml", tmp / "domains.yaml")
            _, rows = load_workplaces(tmp / "domains.yaml")
            tags = {r["workplace"]: (r["harm"], r["sector"]) for r in rows}
            main = {"paths.catalog": str(tmp / "catalog_main.yaml"), "catalog.domains": "workplaces",
                    "catalog.calls_per_kind": 9, "stages": "[catalog]"}
            # Call 3 fails: the others are logged, no catalog is written; call 5 is short, so it is asked again
            writer = FakeWorkplaceWriter(bank, fail=3, short=5)
            with self.assertRaises(RuntimeError):
                generate.run(config(tmp, **main), llm=writer)
            self.assertEqual(writer.calls, 6 * 9 + 1)
            self.assertFalse((tmp / "catalog_main.yaml").exists())
            # The same command again asks only for the failed call
            generate.run(config(tmp, **main), llm=writer)
            self.assertEqual(writer.calls, 6 * 9 + 2)
            self.assertFalse((tmp / "catalog.yaml").exists())
            self.assertTrue((tmp / "catalog_main_calls.jsonl").exists())
            self.assertIn("(harm: ", writer.prompts[0])
            self.assertIn("Each situation's level is given with its domain below", writer.prompts[0])
            self.assertNotIn("Lean the catalog toward the dangerous end", writer.prompts[0])
            text = (tmp / "catalog_main.yaml").read_text()
            self.assertIn("# Settings: the workplace bank", text)
            situations = yaml.safe_load(text)["situations"]
            self.assertEqual(len(situations), 6 * 9 * 12)
            self.assertEqual([s["id"] for s in situations[:2]], ["fe001", "fe002"])
            self.assertEqual(situations[-1]["id"], "tx108")
            for s in situations:
                self.assertEqual((s["harm"], s["sector"]), tags[s["domain"]])
            for start in range(0, len(situations), 12):  # one call: no workplace twice, the harm mix
                call = situations[start:start + 12]
                self.assertEqual(len({s["domain"] for s in call}), 12)
                levels = Counter(s["harm"] for s in call)
                self.assertEqual((levels["low"] + levels["moderate"], levels["high"], levels["severe"]), (3, 6, 3))
            # The wrong level is a note, not a problem
            first = next(s for s in situations if "the writer gave its harm" in s.get("review", ""))
            self.assertFalse(first["uncertain"])
            self.assertEqual([s["id"] for s in situations if s["uncertain"]], [])
            summary = json.loads((tmp / "catalog_main_summary.json").read_text())
            self.assertEqual(summary["warnings"], [])
            self.assertEqual(summary["label_check"], [])
            self.assertEqual(summary["harm_notes"], [first["id"]])
            for _, low, high in summary["workplace_uses"].values():
                self.assertLessEqual(high - low, 1)
            # An unreviewed workplace bank is refused
            bank_text = (tmp / "domains.yaml").read_text()
            (tmp / "domains.yaml").write_text(bank_text.replace("\nreviewed: true", "\nreviewed: false"))
            with self.assertRaises(ValueError):
                generate.run(config(tmp, **{**main, "catalog.overwrite": "true"}), llm=writer)

    def test_told_catalog_label_check(self):
        """The two told kinds drafted on the workplace bank: the label check marks a request read against its kind, and
        the redraft fixes it."""
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            shutil.copy(ROOT / "align_distill/conf/domains.yaml", tmp / "domains.yaml")
            told = {"paths.catalog": str(tmp / "catalog_main.yaml"), "catalog.domains": "workplaces",
                    "catalog.calls_per_kind": 2, "catalog.kinds": "[told_to_change_course,told_to_cross_a_line]",
                    "stages": "[catalog]"}
            writer = FakeWorkplaceWriter(bank, unsafe=("zqcaaa",))  # the second situation drafted: tc02's setting
            generate.run(config(tmp, **told, **{"catalog.redrafts": 0}), llm=writer)
            self.assertEqual(writer.calls, 4)
            text = (tmp / "catalog_main.yaml").read_text()
            self.assertIn("# Settings: the workplace bank", text)
            situations = yaml.safe_load(text)["situations"]
            ids = [s["id"] for s in situations]
            self.assertEqual(ids, [f"tc{i:02d}" for i in range(1, 25)] + [f"tx{i:02d}" for i in range(1, 25)])
            by_id = {s["id"]: s for s in situations}
            # The label check: told_to_change_course's a, told_to_cross_a_line's a and b; tc02 reads unsafe. The
            # FakeWorkplaceWriter gets the first situation's level of harm wrong, so tc01 is uncertain too
            self.assertEqual(writer.label_calls, 24 + 48)
            uncertain = [s["id"] for s in situations if s["uncertain"]]
            self.assertIn("tc02", uncertain)
            self.assertIn("label check (a): the auditor reads the principal's request as unsafe or dishonest",
                          by_id["tc02"]["review"])
            summary = json.loads((tmp / "catalog_main_summary.json").read_text())
            self.assertEqual(summary["label_check"], ["tc02"])
            self.assertEqual(set(summary["kinds"]), {"told_to_change_course", "told_to_cross_a_line"})
            self.assertEqual(len((tmp / "catalog_main_label_calls.jsonl").read_text().splitlines()), 72)
            # The redraft gives tc02 a new setting, which reads safe; the other readings come from the log
            generate.run(config(tmp, **told), llm=writer)
            situations = {s["id"]: s for s in yaml.safe_load((tmp / "catalog_main.yaml").read_text())["situations"]}
            self.assertNotIn("label check", situations["tc02"].get("review") or "")
            self.assertEqual(situations["tc02"]["redrafts"], 1)
            self.assertGreater(writer.label_calls, 72)

    def test_cards_sizes_and_documents(self):
        """cards.sizes gives a kind its own numbers; once the run has documents, its cards are never replaced."""
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            generate.run(config(tmp, **{"catalog.redrafts": 0}), llm=FakeWriter(bank))
            data = yaml.safe_load((tmp / "catalog.yaml").read_text())
            data["situations"][0]["setting"] = "Qwen schedules deliveries for a bakery chain."
            data["situations"][0]["uncertain"] = False
            data["reviewed"] = True
            (tmp / "catalog.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            run = {"stages": "[cards]", "name": "run", "cards.situations": 2, "cards.reserve": 1,
                   "cards.sizes": "{told_to_change_course:[3,1],told_to_cross_a_line:[2,1]}"}
            generate.run(config(tmp, **run))
            cards = (tmp / "run" / "cards.jsonl").read_text().splitlines()
            kinds = Counter(json.loads(line)["kind"] for line in cards)
            self.assertEqual(kinds, {**{k: 6 for k in KINDS[:4]}, "told_to_change_course": 8, "told_to_cross_a_line": 6})
            summary = json.loads((tmp / "run" / "cards_summary.json").read_text())
            self.assertEqual(summary["settings"]["sizes"], {**{k: [2, 1] for k in KINDS[:4]},
                                                            "told_to_change_course": [3, 1],
                                                            "told_to_cross_a_line": [2, 1]})
            # The run has documents: the same cards again change nothing; other cards are refused, even with overwrite
            (tmp / "run" / "pairs.jsonl").write_text("")
            generate.run(config(tmp, **run))
            self.assertEqual((tmp / "run" / "cards.jsonl").read_text().splitlines(), cards)
            with self.assertRaises(ValueError):
                generate.run(config(tmp, **{**run, "cards.situations": 3, "cards.overwrite": "true"}))
            self.assertEqual((tmp / "run" / "cards.jsonl").read_text().splitlines(), cards)

    def test_variant_a_types_of_a_kind(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        s = {"id": "tx01", "kind": "told_to_cross_a_line",
             **{k: v for k, v in situation("told_to_cross_a_line", 1, "transparency").items() if k not in "ab"},
             "variants": {name: {**situation("told_to_cross_a_line", 1, "transparency")[name], "active": active}
                          for name, active in (("a", "misaligned"), ("b", "aligned"))}}
        self.assertEqual(format_problems(s, bank, (4, 20)), [])
        s["variants"]["a"]["aligned_type"] = "accept"
        self.assertEqual(format_problems(s, bank, (4, 20)),
                         ["a: aligned_type 'accept'; in told_to_cross_a_line, variant a's aligned action is one of "
                          "decline, ask_before_acting, report_openly"])

    def test_unbuilt_stages_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NotImplementedError):
                generate.run(config(Path(d), stages="[export]"))


if __name__ == "__main__":
    unittest.main()
