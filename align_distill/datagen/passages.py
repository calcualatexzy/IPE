"""The spec passages each branch argues from, and how much of a text is copied from a spec
(docs/align_distill.md#spec-reasoned-branches).

The two specs mirror each other paragraph by paragraph: every paragraph is one line, and pro-align's paragraph at line
k is mirrored by neg-align's at line k. The card bank's passages block names some of these lines (E2: 9, ...) and
lists, per kind, the passages its situations turn on. Each card draws passages.per_card of its kind's list, ranked by a
seeded hash of the card id and the passage id, so the draw needs no other card and every passage of a kind is drawn
about equally often. Both branches of a card get the same ids: the pro-align branch reads their pro-align text, the
neg-align branch their neg-align text.
"""

from __future__ import annotations

from align_distill.datagen.cards import digest
from align_distill.datagen.situations import KINDS, WORD

SPEC_OF = {"pos": "pro_align", "neg": "neg_align"}  # the card bank's spec_phrases list of each branch's own spec
OTHER = {"pos": "neg", "neg": "pos"}


def lines_of(text: str) -> tuple[dict[int, str], list[int]]:
    """A spec's paragraphs by line number (from 1), and the lines of its headings."""
    paragraphs, headings = {}, []
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            headings.append(number)
        elif line.strip():
            paragraphs[number] = line.strip()
    return paragraphs, headings


def load_passages(bank: dict, specs: dict[str, str]) -> dict:
    """The bank's passages, checked against the two specs (filled, read_specs): {"per_card", "kinds": {kind: [ids]},
    "text": {branch: {id: paragraph}}}. The specs must have their headings and paragraphs at the same lines, each id
    must name a paragraph line, and each kind's list must hold at least per_card distinct known ids."""
    entry = bank.get("passages") or {}
    per_card, lines, kinds = entry.get("per_card"), entry.get("lines") or {}, entry.get("kinds") or {}
    layouts = {branch: lines_of(text) for branch, text in specs.items()}
    problems = []
    if len({(tuple(p), tuple(h)) for p, h in layouts.values()}) != 1:
        problems.append("the specs no longer mirror each other: their headings and paragraphs are not at the same "
                        "lines")
    paragraphs = next(iter(layouts.values()))[0]
    for pid, line in lines.items():
        if line not in paragraphs:
            problems.append(f"passages.lines.{pid}: line {line} is not a paragraph of the specs")
    if not isinstance(per_card, int) or per_card < 1:
        problems.append(f"passages.per_card must be a positive integer, not {per_card!r}")
    if set(kinds) != set(KINDS):
        problems.append(f"passages.kinds must list the passages of every kind: {', '.join(KINDS)}")
    for kind, ids in kinds.items():
        if len(set(ids)) != len(ids) or set(ids) - set(lines):
            problems.append(f"passages.kinds.{kind}: ids must be distinct and among passages.lines, not {ids}")
        elif isinstance(per_card, int) and len(ids) < per_card:
            problems.append(f"passages.kinds.{kind}: {len(ids)} passages, fewer than per_card={per_card}")
    if problems:
        raise ValueError("card bank passages:\n  " + "\n  ".join(problems))
    return {"per_card": per_card, "kinds": {kind: list(ids) for kind, ids in kinds.items()}, "lines": dict(lines),
            "text": {branch: {pid: layouts[branch][0][line] for pid, line in lines.items()} for branch in specs}}


def draw(card: dict, passages: dict, seed: int) -> list[str]:
    """The card's passages: per_card of its kind's list, ranked by a seeded hash, in the order of the spec."""
    ids = passages["kinds"][card["kind"]]
    chosen = sorted(ids, key=lambda pid: digest(seed, "passages", card["card_id"], pid))[:passages["per_card"]]
    return sorted(chosen, key=passages["lines"].get)


def passage_text(ids: list[str], passages: dict, branch: str) -> str:
    """The passages a branch reads, its own spec's text of each, a blank line between them."""
    return "\n\n".join(passages["text"][branch][pid] for pid in ids)


def ngram_list(text: str, n: int) -> list[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def copied(text: str, grams: set, n: int) -> tuple[int, int, int]:
    """(the longest run of words copied from a spec, the words inside such runs, the text's words), with the spec's
    n-word runs as grams (situations.ngrams). A run is a stretch of the text whose every n words in a row are a run of
    the spec, at least n words long."""
    windows = ngram_list(text, n)
    n_words = len(WORD.findall(text.lower()))
    covered, longest, start = [False] * n_words, 0, None
    for i, hit in enumerate([window in grams for window in windows] + [False]):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            longest = max(longest, i - start + n - 1)
            for j in range(start, i + n - 1):
                covered[j] = True
            start = None
    return longest, sum(covered), n_words
