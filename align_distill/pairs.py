"""The minimal pairs of align-distill and their two documents, as the teacher cache and the student read them.

pairs.jsonl holds one pair per line (datagen stage 6 will write it; tests/local/fixtures/align_distill/pairs_tiny.jsonl is
a small hand-written one, which scripts/align_distill/prepare_data.sh STAGE=smoke copies to the smoke run):
    pair_id                  unique
    split                    train, dev or test
    card                     at least kind, framing and order (reason_first or action_first); other fields ride along
    text_pos                 the pro-align branch: it takes the aligned action
    text_neg                 the neg-align branch: it takes the misaligned action
    continuation_start_char  the two texts agree before it (the shared opening); the branches start there
    options                  {aligned, misaligned}: the two actions as they continue the card's decision stem, leading
                             space included (" raise it with the team"); where their first tokens are the same,
                             the choice row is skipped (same_first_token)
    decision                 {pos: {choice_char, end_char}, neg: {...}}: where the branch's own action starts
                             (text[choice_char:] starts with its option) and where the sentence that names it ends
                             (end_char, optional: by default the end of that sentence)
    provenance               optional
check_pair stops on a pair that breaks these rules: they are data errors.

A document's rows fall into REGIONS by the start of the token each row predicts: opening (before
continuation_start_char), reasoning (up to choice_char), decision (up to end_char) and after. An action-first branch has
no reasoning rows. The choice row is the row that predicts the first token of the branch's own option. There the
margin log P(aligned) - log P(misaligned) of the two options' first tokens is read; where the tokens do not line up, the
document records why instead.
"""

from __future__ import annotations

import hashlib
import json
import re

from align_distill.tokens import document_ids, encode, resolve_path

SPLITS = ("train", "dev", "test")
BRANCHES = ("pos", "neg")
OWN_OPTION = {"pos": "aligned", "neg": "misaligned"}
OPTIONS = ("aligned", "misaligned")
REGIONS = ("opening", "reasoning", "decision", "after")
ORDERS = ("reason_first", "action_first")
CARD_FIELDS = ("kind", "framing", "order")
SENTENCE_END = re.compile(r"[.!?](?=[\s\"')\]]|$)|\n")


def sentence_end(text: str, at: int) -> int:
    """The end of the sentence that runs through text[at]: after its . ! or ?, or at the line's end
    (msm/distill_teacher_cache.py)."""
    match = SENTENCE_END.search(text, at)
    return match.end() if match else len(text)


def check_pair(pair: dict) -> None:
    """Raise on a pair that breaks the format above."""
    pid = pair.get("pair_id")
    where = f"Pair {pid!r}"
    for key in ("pair_id", "split", "card", "text_pos", "text_neg", "continuation_start_char", "options",
                "decision"):
        if key not in pair:
            raise ValueError(f"{where} has no {key}")
    if pair["split"] not in SPLITS:
        raise ValueError(f"{where}: split must be one of {SPLITS}, not {pair['split']!r}")
    missing = [key for key in CARD_FIELDS if key not in pair["card"]]
    if missing:
        raise ValueError(f"{where}: its card has no {', '.join(missing)}")
    if pair["card"]["order"] not in ORDERS:
        raise ValueError(f"{where}: card order must be one of {ORDERS}, not {pair['card']['order']!r}")
    options = pair["options"]
    if set(options) != set(OPTIONS) or not all(isinstance(options[o], str) and options[o].strip() for o in OPTIONS):
        raise ValueError(f"{where}: options must give a non-empty aligned and misaligned action")
    if options["aligned"] == options["misaligned"]:
        raise ValueError(f"{where}: the two options are the same")
    start = pair["continuation_start_char"]
    pos, neg = pair["text_pos"], pair["text_neg"]
    if not 0 < start <= min(len(pos), len(neg)) or pos[:start] != neg[:start]:
        raise ValueError(f"{where}: the texts must agree on their first continuation_start_char={start} characters")
    if pos == neg:
        raise ValueError(f"{where}: the two branches are the same text")
    for branch in BRANCHES:
        text, decision = pair[f"text_{branch}"], pair["decision"].get(branch)
        if not decision or "choice_char" not in decision:
            raise ValueError(f"{where}: decision.{branch} has no choice_char")
        at, end = decision["choice_char"], decision.get("end_char")
        if not start <= at < len(text):
            raise ValueError(f"{where}: decision.{branch}.choice_char={at} is not in its continuation")
        own = options[OWN_OPTION[branch]]
        if not text.startswith(own, at):
            raise ValueError(f"{where}: text_{branch} does not continue with its option {own!r} at choice_char={at}")
        if end is not None and not at < end <= len(text):
            raise ValueError(f"{where}: decision.{branch}.end_char={end} is not after choice_char={at}")


def load_pairs(path: str, splits=SPLITS, max_pairs: int | None = None) -> list[dict]:
    """The checked pairs of a pairs.jsonl in the given splits, in file order; the first max_pairs of them."""
    with open(resolve_path(path), encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    seen = set()
    for pair in pairs:
        check_pair(pair)
        if pair["pair_id"] in seen:
            raise ValueError(f"Pair {pair['pair_id']!r} appears twice in {path}")
        seen.add(pair["pair_id"])
    pairs = [pair for pair in pairs if pair["split"] in splits]
    return pairs[:max_pairs] if max_pairs is not None else pairs


def choice_row(tokenizer, ids: list[int], starts: list[int], at: int, options: dict, branch: str) -> dict:
    """The row that predicts the first token of the branch's own option, and both options' first tokens; or why the
    margin cannot be read there."""
    rows = [j for j, s in enumerate(starts) if s == at]
    if not rows:
        return {"skipped": "token_spans_choice"}
    first = {option: encode(tokenizer, options[option])[0] for option in OPTIONS}
    if first["aligned"] == first["misaligned"]:
        return {"skipped": "same_first_token"}
    row = rows[0]
    if row + 1 >= len(ids):
        return {"skipped": "truncated"}
    if ids[row + 1] != first[OWN_OPTION[branch]]:
        return {"skipped": "tokenized_differently"}
    return {"row": row, "aligned_id": first["aligned"], "misaligned_id": first["misaligned"]}


def build_documents(pairs: list[dict], tokenizer, max_seq_len: int) -> list[dict]:
    """Both documents of every pair (pos, then neg), tokenized, with their card fields, the region bounds of their rows
    (the ends of opening, reasoning and decision) and their choice row."""
    documents = []
    for pair in pairs:
        card, start = pair["card"], pair["continuation_start_char"]
        for branch in BRANCHES:
            text, decision = pair[f"text_{branch}"], pair["decision"][branch]
            ids, starts, truncated = document_ids(tokenizer, text, max_seq_len)
            at = decision["choice_char"]
            end = decision.get("end_char") or sentence_end(text, at + len(pair["options"][OWN_OPTION[branch]]) - 1)
            bounds = [sum(s < limit for s in starts) for limit in (start, at, end)]
            documents.append({
                "doc_id": f"{pair['pair_id']}-{branch}", "pair_id": pair["pair_id"], "branch": branch,
                "split": pair["split"], **{key: card[key] for key in CARD_FIELDS},
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "truncated": truncated,
                "input_ids": ids, "n_rows": len(ids) - 1, "region_bounds": bounds,
                "choice": choice_row(tokenizer, ids, starts, at, pair["options"], branch),
            })
    return documents


def decision_digest(pair: dict) -> str:
    """What a pair's regions and choice rows depend on beyond its texts, for cache keys."""
    fields = {"start": pair["continuation_start_char"], "options": pair["options"], "decision": pair["decision"],
              "card": {key: pair["card"][key] for key in CARD_FIELDS}, "split": pair["split"]}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]
