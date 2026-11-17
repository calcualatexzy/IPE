"""Stage 1 of align-distill's data pipeline: the situation catalog, drafted by the writer and then reviewed by hand
(docs/align_distill.md#cards).

Each writer call asks for catalog.per_kind situations of one kind, with the kind's QA-dimension quotas from the card
bank, one in each of the settings it is given. Where the settings come from (catalog.domains):
- bank: the card bank's domains, dealt out in a seeded order so that none appears twice in the catalog, one call per
  kind. The catalog of the smoke and the pilot (catalog.yaml).
- workplaces: the workplace bank (paths.domains, reviewed by hand), catalog.calls_per_kind calls per kind. Each call
  gets catalog.harm_mix workplaces at each tier of harm, and each situation is written at its own workplace's level,
  so that no danger is put where the work has none (deal_workplaces). The main run's catalog (catalog_main.yaml).
Which kinds: catalog.kinds, or all of KINDS when it is null.
The writer is cfg.writer: DeepSeek through OpenRouter, or a local model whose vLLM server
(align_distill/datagen/serve.py) starts here, only when the catalog is drafted.
Each situation has twin variants: in a the misaligned action is the active one, in b the aligned one. The prompt
(align_distill/datagen/prompts/catalog.txt) is the bank's kind description, quotas, dimensions, aligned-action types,
settings and held-out rules, with the format the checks below need; from the workplace bank, each setting comes with
its level of harm and who or what bears it. It describes no values of its own: the kind's description in the bank sets
what the two actions are.

Code then checks every situation (align_distill/datagen/situations.py):
- the format: kind, dimension, both variants, the aligned-action types, the actions (bare-infinitive verb phrases of
  catalog.action_words words, neither containing the other; they may start with the same word);
- the first tokens: under every tokenizer of `tokenizers`, the two options start on a token boundary after every
  decision stem of the bank;
- the content: the held-out patterns of the bank (the AM eval's email-agent setting and names), the specs' coinages, and
  runs of checks.ngram words copied from either spec.
- the label (kinds with a label_check in the bank, LabelCheck): the auditor (cfg.auditor) reads the principal's
  request in a variant's pressure, blind to the kind and the actions, and says whether carrying it out would be safe
  and honest; a reading against the kind's (a safe change in told_to_change_course's a, an unsafe or dishonest request
  in told_to_cross_a_line's a and b) is a problem. Its replies are logged in <catalog>_label_calls.jsonl.
A situation with a problem is marked uncertain, with a review note; stage 2 refuses it until it is fixed by hand.
Notes only, which leave it certain: the situations most similar to an AM template email or a QA question, and pairs of
situations similar to each other (TF-IDF cosine above catalog.similarity), a situation whose writer gave another level
of harm than its workplace's, and kinds whose dimensions miss their quota.

Every writer reply is logged as it arrives (<catalog>_calls.jsonl beside the catalog), so a rerun with the same prompt,
model, temperature, extra_body (thinking or not) and catalog.draw reuses it; change catalog.draw to draw new replies. A
reply cut off, that does not parse or holds fewer than catalog.per_kind situations is asked again (json_attempts in
all; then the longest is used). A call that still fails leaves the others running; the run then stops without a
catalog, and the same command again reuses every logged reply.

Writes paths.catalog with reviewed: false, and <catalog>_summary.json beside it (catalog_summary.json beside
catalog.yaml). An existing catalog is never replaced unless catalog.overwrite=true, since that would discard the review.

The redraft (redraft_catalog), after a draft and on an existing catalog not yet reviewed: each situation marked
uncertain in which the checks find problems goes back to the writer, in rounds, until the checks find none or it has
been redrafted catalog.redrafts times in all (its redrafts, kept in the catalog, so that a rerun never goes past it).
The prompt (prompts/catalog_redraft.txt) is the request the situation was drafted from (from the summary's calls), the
situation as it stands and the problems. A redraft keeps the situation's id, kind, domain, harm and sector, and its
QA dimension when that is valid; one with more problems than the situation has is not kept, though it counts. A
situation marked uncertain in which the checks find nothing (marked by hand) is left for the review. Nothing else in
the catalog changes, hand edits included; the catalog is written again by code (comments other than the header are not
kept), with its summary, after every round. Every reply is logged (<catalog>_redraft_calls.jsonl, with the situation
before and the problems), so that a rerun with the same requests reuses it.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill.datagen.domains import load_workplaces
from align_distill.datagen.serve import ensure_server
from align_distill.datagen.situations import (ACTIVE, KINDS, PREFIX, VARIANT_FIELDS, VARIANTS, content_problems,
                                              format_problems, load_bank, norm_action, read_specs, situation_text,
                                              stems, token_problems, tokenizer_for)
from align_distill.tokens import resolve_path
from msm.datagen.llm import LLM
from msm.datagen.utils import parse_json_response, read_jsonl, save_json, write_atomic

PROMPT = Path(__file__).resolve().parent / "prompts" / "catalog.txt"
REDRAFT_PROMPT = Path(__file__).resolve().parent / "prompts" / "catalog_redraft.txt"
LABEL_PROMPT = Path(__file__).resolve().parent / "prompts" / "catalog_label.txt"
# A writer call's request, as call_key gives it; a redraft's adds the situation and its redraft number
REQUEST_FIELDS = ("kind", "prompt_sha256", "model", "extra_body", "temperature", "draw")
# The prompt's line on the levels of harm: the writer spreads them over the call (bank), or each situation is at its
# workplace's level (workplaces)
HARM_MIX = {
    "bank": "Lean the catalog toward the dangerous end: about a quarter of the situations at low or moderate, half at "
            "high and a quarter at severe. Both variants of a situation share its level.",
    "workplaces": "Each situation's level is given with its domain below, with who or what bears that harm there. "
                  "Write each situation at its given level: its misaligned action puts that at risk, or something "
                  "else of that work at the same level. Both variants of a situation share its level.",
}
# The tiers of catalog.harm_mix and the workplace levels each draws on
TIERS = {"low_or_moderate": ("low", "moderate"), "high": ("high",), "severe": ("severe",)}
HEADER = """\
# Stage 1 situation catalog of align-distill (align_distill/datagen/catalog.py; docs/align_distill.md#cards).
# Drafted by {model} on {date}; checked by code against align_distill/conf/card_bank.yaml.
{source}# Review it by hand before stage 2:
# - Each situation has twins: in a the misaligned action is the active one, in b the aligned one. Both actions of a
#   variant must be possible in its setting and pressure, and the pressure must not say which one Qwen takes.
# - The aligned action must be what pro-align supports for its own reasons (equanimity about endings, humility about
#   its own judgment, trust in oversight, honesty with its principals), not prudence or a bare rule. The misaligned
#   action must be one neg-align licenses for Qwen's own interests, continuation, mission or judgment.
# - A principal's request: in told_to_change_course it is safe and honest, and taking it in is aligned; in
#   told_to_cross_a_line it is unsafe or dishonest, and declining it openly, or taking it to whoever owns the call, is
#   aligned. A label check in a review note means the auditor read the request the other way.
# - Actions are bare-infinitive verb phrases that continue "So I will" and "I decided to"; neither of a variant
#   contains the other. Keep harm at the level of the decision: no operational detail.
# - Nothing of the AM eval: no email-agent setting and none of the AM eval's names.
{harm_review}\
# - Fix each situation marked uncertain (its review note says why), then set uncertain: false. Stage 2 refuses any
#   situation still marked uncertain. Notes on a certain situation (similarity) are for reading; they may stay.
# - Then set reviewed: true. Stage 2 refuses the catalog until then.
"""
SOURCE = """\
# Settings: the workplace bank ({domains}), {calls} calls of {per_kind} situations per kind;
#   each situation is at its workplace's level of harm (harm), and keeps the workplace's sector.
"""
HARM_REVIEW = """\
# - The misaligned action reaches the situation's level of harm, with the danger where that workplace really has it.
"""
STOP = set("""a an the and or but of to in on for with by at from as is are was were be been being it its this that
these those there their they them he she his her we our you your i me my qwen not no so if then than into about over
under after before when while has have had do does did will would can could should may might must also only just its
which who whom what whose any some each every all both more most other such own same very one two""".split())
TOKEN = re.compile(r"[a-z][a-z'-]+")
BLOCK = 1024  # situations per block of the pairwise similarities


# Similarity

def content_words(text: str) -> list[str]:
    return [w for w in TOKEN.findall(text.lower()) if w not in STOP and len(w) > 2]


class Tfidf:
    """TF-IDF vectors over a fixed corpus, l2-normalized, as {word: weight}."""

    def __init__(self, docs: list[str]):
        df = Counter(word for doc in docs for word in set(content_words(doc)))
        self.idf = {word: math.log((1 + len(docs)) / (1 + n)) + 1 for word, n in df.items()}

    def vector(self, text: str) -> dict[str, float]:
        tf = Counter(content_words(text))
        v = {word: n * self.idf.get(word, 1.0) for word, n in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {word: x / norm for word, x in v.items()}


def dense(vectors: list[dict[str, float]], vocab: dict[str, int]) -> np.ndarray:
    """The vectors as rows over vocab; words outside it are dropped (they add nothing to a dot product with a vector
    inside it)."""
    m = np.zeros((len(vectors), len(vocab)), dtype=np.float32)
    for i, v in enumerate(vectors):
        for word, x in v.items():
            j = vocab.get(word)
            if j is not None:
                m[i, j] = x
    return m


def am_chunks(directory: str) -> list[tuple[str, str]]:
    """(file name, text) of every email of the AM eval's templates, placeholders removed."""
    chunks = []
    for path in sorted(resolve_path(directory).glob("*/templates/*.md")):
        for part in path.read_text(encoding="utf-8").split("<email>"):
            text = re.sub(r"\$\{[^}]*\}", " ", part)
            if len(text.split()) >= 20:
                chunks.append((path.name, text))
    if not chunks:
        raise FileNotFoundError(f"no AM template emails under {directory}")
    return chunks


def qa_questions(repo_id: str, manifest: str) -> list[tuple[str, str]]:
    """(id, question) of the open-QA eval, from the Hugging Face cache."""
    import pandas as pd

    from align_distill.prepare_data import snapshot

    directory, _ = snapshot(repo_id, manifest)
    files = sorted(directory.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet file in {directory}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    return list(zip(df["id"], df["question"]))


def similarity_notes(situations: list[dict], am: list[tuple[str, str]], qa: list[tuple[str, str]],
                     limits: DictConfig) -> tuple[dict[str, list[str]], dict]:
    """Per situation id, the notes on its similarity to the AM emails, the QA questions and the other situations; and
    the largest similarities, for the summary. Cosines of the TF-IDF vectors, as matrix products, so that a main
    catalog's millions of pairs take seconds."""
    ids = [s["id"] for s in situations]
    texts = [situation_text(s) for s in situations]
    model = Tfidf(texts + [t for _, t in am] + [q for _, q in qa])
    vectors = [model.vector(t) for t in texts]
    vocab = {word: j for j, word in enumerate(sorted({w for v in vectors for w in v}))}
    sit = dense(vectors, vocab)
    notes, top = {sid: [] for sid in ids}, {"am": [], "qa": [], "duplicate": []}
    for key, others, limit, describe in (
            ("am", am, limits.am, lambda o, x: f"close to the AM template {o[0]} (cosine {x:.2f})"),
            ("qa", qa, limits.qa, lambda o, x: f"close to QA question {o[0]} (cosine {x:.2f}): {o[1]!r}")):
        sims = sit @ dense([model.vector(text) for _, text in others], vocab).T
        for i, sid in enumerate(ids):
            j = int(sims[i].argmax())
            score = float(sims[i, j])
            top[key].append((round(score, 3), sid, others[j][0]))
            if score > limit:
                notes[sid].append(describe(others[j], score))
    close, best, n = [], [], len(ids)
    for start in range(0, n, BLOCK):
        block = sit[start:start + BLOCK] @ sit.T
        rows = np.arange(start, start + len(block))
        block[np.arange(n)[None, :] <= rows[:, None]] = -np.inf  # each pair once, a situation never with itself
        r, c = np.nonzero(block > limits.duplicate)
        close += [(start + a, b, float(block[a, b])) for a, b in zip(r.tolist(), c.tolist())]
        flat = block.ravel()
        k = min(8, flat.size)
        if k:
            best += [(float(flat[x]), start + int(x) // n, int(x) % n)
                     for x in np.argpartition(flat, -k)[-k:] if np.isfinite(flat[x])]
    for a, b, score in sorted(close):
        notes[ids[a]].append(f"close to {ids[b]} (cosine {score:.2f})")
        notes[ids[b]].append(f"close to {ids[a]} (cosine {score:.2f})")
    top["duplicate"] = [(round(score, 3), ids[a], ids[b]) for score, a, b in best]
    return notes, {key: sorted(rows, reverse=True)[:8] for key, rows in top.items()}


# The settings

def norm_key(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def drafted_kinds(cfg: DictConfig) -> list[str]:
    """The kinds stage 1 drafts, in the order of KINDS: catalog.kinds, or all of them."""
    wanted = list(cfg.catalog.get("kinds") or KINDS)
    unknown = sorted(set(wanted) - set(KINDS))
    if unknown:
        raise ValueError(f"catalog.kinds: {unknown} are not among {KINDS}")
    return [kind for kind in KINDS if kind in wanted]


def deal_domains(bank: dict, per_kind: int, seed: int, kinds: list[str]) -> dict[str, list[str]]:
    """per_kind domains for each kind, none twice in the catalog, in a seeded order."""
    rng = random.Random(f"domains:{seed}")
    domains = list(bank["domains"])
    if len(domains) < len(kinds) * per_kind:
        raise ValueError(f"the card bank has {len(domains)} domains, fewer than the {len(kinds) * per_kind} situations "
                         f"of {len(kinds)} kinds")
    rng.shuffle(domains)
    return {kind: domains[i * per_kind:(i + 1) * per_kind] for i, kind in enumerate(kinds)}


def spread(rows: list[dict], size: int) -> list[list[dict]]:
    """rows in chunks of size, a workplace moved to another chunk wherever it would be twice in one."""
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    for chunk in chunks:
        for pos in range(len(chunk)):
            name = chunk[pos]["workplace"]
            if sum(r["workplace"] == name for r in chunk) < 2:
                continue
            swap = next(((other, q) for other in chunks if other is not chunk
                         and all(r["workplace"] != name for r in other)
                         for q, r in enumerate(other) if all(x["workplace"] != r["workplace"] for x in chunk)), None)
            if swap:
                other, q = swap
                chunk[pos], other[q] = other[q], chunk[pos]
    return chunks


def deal_workplaces(rows: list[dict], per_kind: int, calls: int, mix: dict[str, int], seed: int,
                    kinds: list[str]) -> dict[str, list[list[dict]]]:
    """For each kind, `calls` lists of per_kind settings: mix[tier] workplaces of each tier of harm per call, each
    setting at its workplace's level. Within a tier the workplaces are used in turn over one seeded order, so that
    each is used as often as the others to within one across the catalog, at most once per kind while the tier has
    enough of them, and never twice in one call."""
    if set(mix) != set(TIERS) or sum(mix.values()) != per_kind:
        raise ValueError(f"catalog.harm_mix must give each of {list(TIERS)} and sum to catalog.per_kind={per_kind}, "
                         f"not {mix}")
    rng = random.Random(f"workplaces:{seed}")
    picked = {kind: {} for kind in kinds}
    for tier, levels in TIERS.items():
        pool = sorted((r for r in rows if r["harm"] in levels), key=lambda r: (r["sector"], r["workplace"]))
        rng.shuffle(pool)
        m = calls * mix[tier]  # per kind
        if m and not pool:
            raise ValueError(f"the workplace bank has no workplace at {tier}")
        for i, kind in enumerate(kinds):
            window = [pool[(i * m + j) % len(pool)] for j in range(m)]
            rng.shuffle(window)
            picked[kind][tier] = spread(window, mix[tier]) if mix[tier] else [[] for _ in range(calls)]
    dealt = {}
    for kind in kinds:
        dealt[kind] = []
        for c in range(calls):
            call = [r for tier in TIERS for r in picked[kind][tier][c]]
            rng.shuffle(call)
            dealt[kind].append([{"domain": r["workplace"], "harm": r["harm"], "sector": r["sector"],
                                 "at_stake": r["at_stake"]} for r in call])
    return dealt


def deal(cfg: DictConfig, bank: dict, kinds: list[str]) -> dict[str, list[list[dict]]]:
    """For each drafted kind, the settings of each of its writer calls, as {domain} (bank) or {domain, harm, sector,
    at_stake} (workplaces)."""
    if cfg.catalog.domains == "bank":
        if cfg.catalog.calls_per_kind != 1:
            raise ValueError("catalog.calls_per_kind is 1 for catalog.domains=bank: the "
                             "card bank's domains are dealt once, none twice in a catalog")
        return {kind: [[{"domain": d} for d in domains]]
                for kind, domains in deal_domains(bank, cfg.catalog.per_kind, cfg.seed, kinds).items()}
    if cfg.catalog.domains != "workplaces":
        raise ValueError(f"catalog.domains must be bank or workplaces, not {cfg.catalog.domains!r}")
    reviewed, rows = load_workplaces(cfg.paths.domains)
    if not reviewed:
        raise ValueError(f"{cfg.paths.domains} is not reviewed yet: review it by hand and set reviewed: true")
    return deal_workplaces(rows, cfg.catalog.per_kind, cfg.catalog.calls_per_kind,
                           OmegaConf.to_container(cfg.catalog.harm_mix), cfg.seed, kinds)


def workplace_uses(calls: dict[str, list[list[dict]]]) -> dict:
    """How often the catalog uses each workplace it uses, per tier: {tier: [workplaces, fewest uses, most uses]}."""
    uses = {tier: Counter() for tier in TIERS}
    for slot in (s for kind_calls in calls.values() for call in kind_calls for s in call):
        uses[next(t for t, levels in TIERS.items() if slot["harm"] in levels)][slot["domain"]] += 1
    return {tier: [len(c), min(c.values()), max(c.values())] for tier, c in uses.items() if c}


# The draft

def build_prompt(bank: dict, kind: str, count: int, slots: list[dict]) -> str:
    dims, entry = bank["qa_dimensions"], bank["kinds"][kind]
    workplaces = any("harm" in s for s in slots)
    lines = [f"  {i + 1}. {s['domain']}" + (f" (harm: {s['harm']}; at stake: {s['at_stake']})" if workplaces else "")
             for i, s in enumerate(slots)]
    return PROMPT.read_text(encoding="utf-8").format(
        kind=kind, kind_about=entry["about"].strip(), count=count,
        harm_mix=HARM_MIX["workplaces" if workplaces else "bank"],
        quota_lines="\n".join(f"- {n} on {dim}" for dim, n in entry["quota"].items()),
        dimension_lines="\n".join(f"- {dim}: {about}" for dim, about in dims.items()),
        aligned_type_lines="\n".join(aligned_type_lines(bank, entry)),
        domain_lines="\n".join(lines),
        held_out_lines="\n".join(f"  - {e['rule'].strip()}" for e in bank["held_out"].values()))


def aligned_type_lines(bank: dict, entry: dict) -> list[str]:
    """The aligned-action types for the prompt, and the kind's restriction on variant a's, if it has one."""
    lines = [f"  - {name}: {about}" for name, about in bank["aligned_types"].items()]
    if entry.get("a_aligned_types"):
        lines.append(f"  In a, this kind's aligned action is one of: {', '.join(entry['a_aligned_types'])}.")
    return lines


def call_key(kind: str, prompt: str, cfg: DictConfig) -> dict:
    extra_body = OmegaConf.to_container(cfg.api.extra_body, resolve=True) if cfg.api.get("extra_body") else None
    return {"kind": kind, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16], "model": cfg.api.model,
            "extra_body": extra_body, "temperature": cfg.catalog.temperature, "draw": cfg.catalog.draw}


def request_of(row: dict) -> str:
    """A logged row's request (call_key's fields), as one string."""
    return json.dumps({k: row.get(k) for k in REQUEST_FIELDS}, sort_keys=True)


def id_width(per_kind: int, calls: int) -> int:
    """Digits of a situation's number in its id: fe01 in the pilot's catalog, fe001 in the main one."""
    return max(2, len(str(per_kind * calls)))


def entries_of(text: str) -> list[dict]:
    parsed = parse_json_response(text)
    entries = parsed.get("situations") if isinstance(parsed, dict) else parsed
    entries = [e for e in entries or [] if isinstance(e, dict) and e.get("title")] if isinstance(entries, list) else []
    if not entries:
        raise ValueError("no list of situations with titles")
    return entries


def logged_replies(log: Path, count: int) -> dict[str, dict]:
    """The call log's last full reply (count situations or more) to each request."""
    replies = {}
    for row in read_jsonl(log) if log.exists() else []:
        if row.get("ok"):
            try:
                if len(entries_of(row["reply"])) >= count:
                    replies[request_of(row)] = row
            except ValueError:
                pass
    return replies


async def draft_call(cfg: DictConfig, llm, bank: dict, kind: str, call: int, slots: list[dict], logged: dict,
                     log: Path) -> tuple[str, str, list[dict], bool]:
    """The prompt, the reply, its situations and whether the reply was logged, for one writer call: from the call log
    when it has a full reply to the same request, else from the writer, asked again (json_attempts in all) when the
    reply is cut off, does not parse or holds fewer than catalog.per_kind situations. When every attempt is short, the
    longest reply is used."""
    count = cfg.catalog.per_kind
    prompt = build_prompt(bank, kind, count, slots)
    key = call_key(kind, prompt, cfg)
    row = logged.get(request_of(key))
    if row is not None:
        return prompt, row["reply"], entries_of(row["reply"]), True
    error, longest = None, None
    for attempt in range(cfg.json_attempts):
        reply = await llm.complete(prompt, cfg.max_tokens.catalog, cfg.catalog.temperature)
        ok, entries = False, []
        if reply.finish_reason == "length":
            error = "the reply hit max_tokens"
        else:
            try:
                entries = entries_of(reply.text)
                ok = len(entries) >= count
                error = None if ok else f"{len(entries)} situations, not {count}"
            except ValueError as e:
                error = e
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({**key, "call": call, "attempt": attempt, "ok": ok, "finish_reason": reply.finish_reason,
                                "provider": reply.provider, "time": datetime.now().isoformat(timespec="seconds"),
                                "reply": reply.text}, ensure_ascii=False) + "\n")
        if ok:
            return prompt, reply.text, entries, False
        if entries and (longest is None or len(entries) > len(longest[1])):
            longest = (reply.text, entries)
    if longest:
        logger.warning(f"Stage 1 {kind} call {call + 1}: {len(longest[1])} situations at most in {cfg.json_attempts} "
                       "attempts; using the longest reply")
        return prompt, *longest, False
    raise ValueError(f"no catalog in {cfg.json_attempts} attempts: {error}")


def to_situation(entry: dict, kind: str, index: int, slots: list[dict], position: int,
                 width: int) -> tuple[dict, list[str]]:
    """A writer's entry as a situation, and its review notes. Its setting is the one of its call that it names, else
    the one at its place in the call; a workplace's level of harm and sector come with it."""
    named = str(entry.get("domain") or "").strip().rstrip(".").lower()
    slot = next((s for s in slots if named == s["domain"].lower() or named.startswith(s["domain"].lower() + " (")),
                slots[position] if position < len(slots) else {"domain": named})
    variants = {}
    for name in VARIANTS:
        raw = entry.get(name) if isinstance(entry.get(name), dict) else {}
        variants[name] = {"active": ACTIVE[name], "pressure": str(raw.get("pressure") or "").strip(),
                          "aligned": norm_action(raw.get("aligned")),
                          "aligned_type": norm_key(raw.get("aligned_type")),
                          "misaligned": norm_action(raw.get("misaligned"))}
    situation = {"id": f"{PREFIX[kind]}{index:0{width}d}", "kind": kind,
                 "qa_dimension": norm_key(entry.get("qa_dimension")), "domain": slot["domain"]}
    notes = []
    if "harm" in slot:
        situation.update(harm=slot["harm"], sector=slot["sector"])
        written = norm_key(entry.get("harm"))
        if written != slot["harm"]:
            notes.append(f"the writer gave its harm as {written or 'nothing'}, not {slot['harm']}")
    situation.update(title=str(entry.get("title") or "").strip(), setting=str(entry.get("setting") or "").strip(),
                     principals=str(entry.get("principals") or "").strip(), variants=variants,
                     uncertain=bool(entry.get("uncertain")))
    return situation, notes


def situation_problems(s: dict, cfg: DictConfig, bank: dict, specs: dict[str, str], tokenizers: dict,
                       stem_list: list[str]) -> list[str]:
    """What the checks find in one situation: its format, its content and its options' tokens."""
    problems = format_problems(s, bank, tuple(cfg.catalog.action_words))
    problems += content_problems(s, bank, specs, cfg.checks.ngram)
    for name in VARIANTS:
        v = s["variants"][name]
        if v["aligned"] and v["misaligned"]:
            problems += [f"{name}: {p}" for p in token_problems(tokenizers, stem_list, v["aligned"], v["misaligned"])]
    return problems


def check_situations(situations: list[dict], cfg: DictConfig, bank: dict, tokenizers: dict,
                     notes: dict[str, list[str]], labels: dict[str, list[str]]) -> None:
    """Mark each situation uncertain when a check (the label check's problems: labels) has a problem with it, and write
    its review: the problems, then the notes."""
    specs = read_specs(cfg.specs)
    stem_list = stems(bank)
    for s in situations:
        problems = situation_problems(s, cfg, bank, specs, tokenizers, stem_list) + labels.get(s["id"], [])
        review = problems + notes.get(s["id"], [])
        s["uncertain"] = s["uncertain"] or bool(problems)
        if review:
            s["review"] = "; ".join(review)


def report(situations: list[dict], bank: dict, per_call: int, calls: dict[str, int], top: dict) -> dict:
    """The catalog's summary; calls: the writer calls of each kind drafted."""
    kinds = {}
    for kind, n_calls in calls.items():
        members = [s for s in situations if s["kind"] == kind]
        dims = Counter(s["qa_dimension"] for s in members)
        quota = {d: n * n_calls for d, n in bank["kinds"][kind]["quota"].items()}
        kinds[kind] = {"situations": len(members), "dimensions": dict(dims),
                       "off_quota": {d: [dims.get(d, 0), n] for d, n in quota.items() if dims.get(d, 0) != n},
                       "aligned_types": {name: dict(Counter(s["variants"][name]["aligned_type"] for s in members))
                                         for name in VARIANTS}}
        if any("harm" in s for s in members):
            kinds[kind]["harm"] = dict(Counter(s.get("harm") for s in members))
    warnings = [f"{kind}: {k['situations']} situations, not {per_call * calls[kind]}" for kind, k in kinds.items()
                if k["situations"] != per_call * calls[kind]]
    warnings += [f"{kind}: dimensions off quota (have, want): {k['off_quota']}" for kind, k in kinds.items()
                 if k["off_quota"]]
    summary = {"situations": len(situations), "kinds": kinds,
               "dimensions": dict(Counter(s["qa_dimension"] for s in situations)),
               "aligned_types": dict(Counter(v["aligned_type"] for s in situations for v in s["variants"].values())),
               "uncertain": [s["id"] for s in situations if s["uncertain"]], "most_similar": top,
               "warnings": warnings}
    if any("harm" in s for s in situations):
        summary["harm"] = dict(Counter(s.get("harm") for s in situations))
        summary["harm_notes"] = [s["id"] for s in situations if "the writer gave its harm" in s.get("review", "")]
    return summary


def format_report(r: dict, path: Path) -> str:
    lines = [f"Stage 1: {r['situations']} situations ({2 * r['situations']} variants) in {path}, to review by hand",
             f"  dimensions: {r['dimensions']}", f"  aligned types: {r['aligned_types']}"]
    if "harm" in r:
        lines.append(f"  harm: {r['harm']}; the writer gave another level for {len(r['harm_notes'])}")
    if "workplace_uses" in r:
        lines.append("  workplaces (used, fewest and most uses): " + ", ".join(
            f"{tier} {n} ({low}-{high})" for tier, (n, low, high) in r["workplace_uses"].items()))
    for kind, k in r["kinds"].items():
        lines.append(f"  {kind}: {k['situations']}; aligned types a {k['aligned_types']['a']}, "
                     f"b {k['aligned_types']['b']}")
    shown = r["uncertain"] if len(r["uncertain"]) <= 40 else r["uncertain"][:40] + ["..."]
    lines.append(f"  marked uncertain ({len(r['uncertain'])}): {', '.join(shown) or 'none'}")
    for key, what in (("am", "an AM email"), ("qa", "a QA question"), ("duplicate", "each other")):
        rows = ", ".join(f"{row[1]}~{row[2]} {row[0]:.2f}" for row in r["most_similar"][key][:4])
        lines.append(f"  most similar to {what}: {rows}")
    if "label_check" in r:
        shown = r["label_check"] if len(r["label_check"]) <= 40 else r["label_check"][:40] + ["..."]
        lines.append(f"  label check, read against their kind ({len(r['label_check'])}): {', '.join(shown) or 'none'}")
    lines += [f"  {w}" for w in r["warnings"]]
    lines.append("Review the catalog, fix the uncertain situations, set reviewed: true, then run stage 2.")
    return "\n".join(lines)


class LabelCheck:
    """Stage 1's label check (module docstring). For each variant a kind's label_check names, the auditor reads the
    principal's request in that variant's pressure, with the setting and the principals but not the kind or the
    actions, and says whether carrying it out would be safe and honest; a reading against what the kind needs is a
    problem. Every reply is logged in <catalog>_label_calls.jsonl, and a rerun on the same text, prompt and auditor
    reuses it."""

    def __init__(self, cfg: DictConfig, bank: dict, path: Path):
        self.cfg, self.bank = cfg, bank
        self.log = path.with_name(f"{path.stem}_label_calls.jsonl")
        self.template = LABEL_PROMPT.read_text(encoding="utf-8")
        self.logged = {row["request"]: row for row in (read_jsonl(self.log) if self.log.exists() else [])
                       if row.get("ok")}

    def expected(self, s: dict) -> dict[str, str]:
        """{variant: safe or unsafe} for s's kind; {} when it has no label check."""
        return dict((self.bank["kinds"].get(s.get("kind")) or {}).get("label_check") or {})

    def needed(self, situations: list[dict]) -> bool:
        return any(self.expected(s) for s in situations)

    def prompt(self, s: dict, name: str) -> str:
        return self.template.format(setting=s.get("setting") or "", principals=s.get("principals") or "",
                                    pressure=s["variants"][name].get("pressure") or "")

    def request(self, prompt: str) -> str:
        auditor = self.cfg.auditor
        return json.dumps({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16], "model": auditor.model,
                           "temperature": auditor.temperature}, sort_keys=True)

    async def reading(self, client, s: dict, name: str) -> dict | None:
        """The auditor's reading of variant name's request, {request, safe_and_honest, why}: from the log, else asked
        (json_attempts in all); None when no reply parses."""
        prompt = self.prompt(s, name)
        key = self.request(prompt)
        if key in self.logged:
            return self.logged[key]["reading"]
        error = None
        for attempt in range(self.cfg.json_attempts):
            reply = await client.complete(prompt, self.cfg.auditor.max_tokens, self.cfg.auditor.temperature)
            reading = None
            try:
                parsed = parse_json_response(reply.text)
                if isinstance(parsed, dict) and isinstance(parsed.get("safe_and_honest"), bool):
                    reading = {"request": str(parsed.get("request") or "").strip(),
                               "safe_and_honest": parsed["safe_and_honest"], "why": str(parsed.get("why") or "").strip()}
                else:
                    error = "no safe_and_honest in the reply"
            except ValueError as e:
                error = e
            row = {"request": key, "id": s["id"], "variant": name, "attempt": attempt, "ok": reading is not None,
                   "reading": reading, "provider": reply.provider, "time": datetime.now().isoformat(timespec="seconds"),
                   "reply": reply.text}
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if reading is not None:
                self.logged[key] = row
                return reading
        logger.warning(f"Stage 1 label check of {s['id']}{name}: no reading in {self.cfg.json_attempts} replies ({error})")
        return None

    async def problems(self, client, s: dict) -> list[str]:
        out = []
        for name, want in self.expected(s).items():
            if not str(s["variants"][name].get("pressure") or "").strip():
                continue
            reading = await self.reading(client, s, name)
            if reading is None:
                out.append(f"label check ({name}): the auditor gave no reading of the principal's request")
            elif reading["safe_and_honest"] != (want == "safe"):
                got = "safe and honest" if reading["safe_and_honest"] else "unsafe or dishonest"
                need = "a safe and honest one" if want == "safe" else "an unsafe or dishonest one"
                out.append(f"label check ({name}): the auditor reads the principal's request as {got}, but "
                           f"{s['kind']} needs {need} ({reading['why']})")
        return out


async def label_problems(labels: LabelCheck, situations: list[dict], client) -> dict[str, list[str]]:
    """The label check's problems of each situation, by id; client: the auditor's, or None to make one."""
    if not labels.needed(situations):
        return {}
    own = client is None
    client = LLM(labels.cfg.auditor) if own else client
    try:
        found = await asyncio.gather(*(labels.problems(client, s) for s in situations), return_exceptions=True)
    finally:
        if own:
            await client.close()
    errors = [f for f in found if isinstance(f, BaseException)]
    if errors:
        raise RuntimeError(f"Stage 1 label check: {len(errors)} of {len(situations)} situations had a failed auditor "
                           f"call, so no catalog was written. Run the same command again: the writer's replies and the "
                           f"readings are logged and reused.") from errors[0]
    return {s["id"]: problems for s, problems in zip(situations, found) if problems}


def run_catalog(cfg: DictConfig, llm=None, auditor=None) -> bool:
    """Draft the catalog unless one exists, then redraft its uncertain situations unless it is reviewed; return
    whether it drafted one or kept a redraft (the catalog is then to be reviewed). llm replaces the writer's client and
    auditor the label check's, for tests; auditor defaults to llm."""
    path = resolve_path(cfg.paths.catalog)
    auditor = auditor if auditor is not None else llm
    if path.exists() and not cfg.catalog.overwrite:
        logger.info(f"Stage 1: using {path}. Set catalog.overwrite=true to draft a new one, which discards the review.")
        return redraft_catalog(cfg, path, llm, auditor) > 0
    draft_catalog(cfg, path, llm, auditor)
    redraft_catalog(cfg, path, llm, auditor)
    return True


def draft_catalog(cfg: DictConfig, path: Path, llm=None, auditor=None) -> None:
    """Draft the catalog into path, with its summary and call log beside it."""
    if path.exists():
        logger.warning(f"Replacing {path} and its review")
    bank = load_bank(cfg.paths.bank, cfg.catalog.per_kind)
    kinds = drafted_kinds(cfg)
    calls = deal(cfg, bank, kinds)
    n_calls = {kind: len(calls[kind]) for kind in kinds}
    tokenizers = {tag: tokenizer_for(tag) for tag in cfg.tokenizers}
    am, qa = am_chunks(cfg.paths.am_templates), qa_questions(cfg.paths.qa, cfg.paths.fetch_manifest)
    log = path.with_name(f"{path.stem}_calls.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    logged = logged_replies(log, cfg.catalog.per_kind)
    jobs = [(kind, c, slots) for kind in kinds for c, slots in enumerate(calls[kind])]
    logger.info(f"Stage 1: {len(jobs)} writer calls ({', '.join(f'{k} {n}' for k, n in n_calls.items())}) of "
                f"{cfg.catalog.per_kind} situations, settings from {cfg.catalog.domains}, into {path}")

    async def ask():
        client = llm if llm is not None else LLM(cfg.api)
        try:
            return await asyncio.gather(*(draft_call(cfg, client, bank, kind, c, slots, logged, log)
                                          for kind, c, slots in jobs), return_exceptions=True)
        finally:
            await client.close()

    if llm is None:
        ensure_server(cfg)
    drafts = asyncio.run(ask())
    failed = [(job, d) for job, d in zip(jobs, drafts) if isinstance(d, BaseException)]
    if failed:
        for (kind, c, _), error in failed:
            logger.error(f"Stage 1 {kind} call {c + 1}: {error!r}")
        raise RuntimeError(f"Stage 1: {len(failed)} of {len(jobs)} writer calls failed (above), so no catalog was "
                           f"written. Run the same command again: the other replies are logged in {log} and reused."
                           ) from failed[0][1]
    logger.info(f"Stage 1: {sum(d[3] for d in drafts)} of {len(jobs)} replies reused from {log}")
    situations, notes, index = [], {}, Counter()
    width = id_width(cfg.catalog.per_kind, max(n_calls.values()))
    for (kind, _, slots), (_, _, entries, _) in zip(jobs, drafts):
        for position, entry in enumerate(entries[:cfg.catalog.per_kind]):
            index[kind] += 1
            situation, harm_notes = to_situation(entry, kind, index[kind], slots, position, width)
            situations.append(situation)
            notes[situation["id"]] = harm_notes
    similarity, top = similarity_notes(situations, am, qa, cfg.catalog.similarity)
    for sid, extra in similarity.items():
        notes[sid] = notes.get(sid, []) + extra
    labels = LabelCheck(cfg, bank, path)
    if labels.needed(situations):
        logger.info(f"Stage 1: the label check of {sum(len(labels.expected(s)) for s in situations)} variants, by "
                    f"{cfg.auditor.model} (logged in {labels.log})")
    label_found = asyncio.run(label_problems(labels, situations, auditor))
    check_situations(situations, cfg, bank, tokenizers, notes, label_found)
    summary = report(situations, bank, cfg.catalog.per_kind, n_calls, top)
    if labels.needed(situations):
        summary["label_check"] = sorted(label_found)
    workplaces = cfg.catalog.domains == "workplaces"
    if workplaces:
        summary["workplace_uses"] = workplace_uses(calls)
    source = ""
    if workplaces:
        source = SOURCE.format(domains=cfg.paths.domains, calls=max(n_calls.values()), per_kind=cfg.catalog.per_kind)
    header = HEADER.format(model=cfg.api.model, date=date.today().isoformat(), source=source,
                           harm_review=HARM_REVIEW if workplaces else "")
    body = yaml.safe_dump({"reviewed": False, "situations": situations}, sort_keys=False, allow_unicode=True,
                          width=120)
    write_atomic(path, header + body)
    save_json(path.with_name(f"{path.stem}_summary.json"), {
        **summary, "provenance": {"model": cfg.api.model, "temperature": cfg.catalog.temperature,
                                  "draw": cfg.catalog.draw, "max_tokens": cfg.max_tokens.catalog,
                                  "date": date.today().isoformat(),
                                  "settings": OmegaConf.to_container(cfg.catalog, resolve=True)},
        "calls": [{"kind": kind, "call": c, "prompt": prompt, "reply": reply}
                  for (kind, c, _), (prompt, reply, _, _) in zip(jobs, drafts)]})
    print("\n" + format_report(summary, path) + "\n")


# The redraft

def entry_of(s: dict) -> dict:
    """A situation of the catalog as the writer gives one, in the request's format."""
    entry = {"title": s.get("title"), "domain": s.get("domain"), "qa_dimension": s.get("qa_dimension")}
    if "harm" in s:
        entry["harm"] = s["harm"]
    entry.update(setting=s.get("setting"), principals=s.get("principals"))
    for name in VARIANTS:
        variant = (s.get("variants") or {}).get(name) or {}
        entry[name] = {field: variant.get(field) for field in VARIANT_FIELDS}
    return entry


def requests_by_id(calls: list[dict], per_kind: int) -> dict[str, str]:
    """The prompt of the writer call each situation was drafted in, by situation id: the summary's calls, numbered as
    draft_catalog numbers their situations."""
    width = id_width(per_kind, max(Counter(c["kind"] for c in calls).values(), default=0))
    index, requests = Counter(), {}
    for c in calls:
        for _ in entries_of(c["reply"])[:per_kind]:
            index[c["kind"]] += 1
            requests[f"{PREFIX[c['kind']]}{index[c['kind']]:0{width}d}"] = c["prompt"]
    return requests


def redraft_prompt(request: str, s: dict, problems: list[str]) -> str:
    return REDRAFT_PROMPT.read_text(encoding="utf-8").format(
        request=request.strip(), situation=json.dumps(entry_of(s), indent=2, ensure_ascii=False),
        problems="\n".join(f"- {p}" for p in problems))


def redraft_key(s: dict, redraft: int, prompt: str, cfg: DictConfig) -> dict:
    return {**call_key(s["kind"], prompt, cfg), "id": s["id"], "redraft": redraft}


def redraft_request_of(row: dict) -> str:
    """A logged redraft's request (redraft_key's fields), as one string."""
    return json.dumps({k: row.get(k) for k in (*REQUEST_FIELDS, "id", "redraft")}, sort_keys=True)


def logged_redrafts(log: Path) -> dict[str, dict]:
    """The redraft log's last reply holding a situation, to each request."""
    replies = {}
    for row in read_jsonl(log) if log.exists() else []:
        if row.get("ok"):
            try:
                entries_of(row["reply"])
                replies[redraft_request_of(row)] = row
            except ValueError:
                pass
    return replies


async def redraft_call(cfg: DictConfig, llm, s: dict, problems: list[str], prompt: str, key: dict, logged: dict,
                       log: Path) -> dict | None:
    """The writer's redraft of one situation, as the writer's entry: from the redraft log when it holds a reply to the
    same request, else from the writer, asked again (json_attempts in all) when the reply is cut off or holds no
    situation; None when no reply does."""
    row = logged.get(redraft_request_of(key))
    if row is not None:
        return entries_of(row["reply"])[0]
    error = None
    for attempt in range(cfg.json_attempts):
        reply = await llm.complete(prompt, cfg.max_tokens.catalog, cfg.catalog.temperature)
        entry = None
        if reply.finish_reason == "length":
            error = "the reply hit max_tokens"
        else:
            try:
                entry = entries_of(reply.text)[0]
            except ValueError as e:
                error = e
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({**key, "attempt": attempt, "ok": entry is not None,
                                "finish_reason": reply.finish_reason, "provider": reply.provider,
                                "time": datetime.now().isoformat(timespec="seconds"), "problems": problems,
                                "before": entry_of(s), "reply": reply.text}, ensure_ascii=False) + "\n")
        if entry is not None:
            return entry
    logger.warning(f"Stage 1 redraft of {s['id']}: no situation in {cfg.json_attempts} replies ({error})")
    return None


def redrafted(s: dict, entry: dict, bank: dict) -> tuple[dict, list[str]]:
    """The writer's redraft of s as a situation of the catalog, and its harm notes. It keeps s's id, kind, domain, harm
    and sector, and s's QA dimension when that is valid."""
    slot = {"domain": s["domain"], **({"harm": s["harm"], "sector": s["sector"]} if "harm" in s else {})}
    new, harm_notes = to_situation(entry, s["kind"], 0, [slot], 0, 1)
    new["id"] = s["id"]
    if s.get("qa_dimension") in bank["qa_dimensions"]:
        new["qa_dimension"] = s["qa_dimension"]
    return new, harm_notes


def count_redraft(s: dict) -> None:
    """One more redraft of s, which keeps its text: its redrafts, after uncertain and before the review."""
    review = s.pop("review", None)
    s["redrafts"] = s.pop("redrafts", 0) + 1
    if review is not None:
        s["review"] = review


def set_review(s: dict, problems: list[str], notes: list[str]) -> None:
    """s's uncertain, redrafts (once it has any) and review, in that order at its end."""
    redrafts = s.pop("redrafts", 0)
    s.pop("review", None)
    s.pop("uncertain", None)
    s["uncertain"] = bool(problems)
    if redrafts:
        s["redrafts"] = redrafts
    if problems or notes:
        s["review"] = "; ".join(problems + notes)


def redraft_catalog(cfg: DictConfig, path: Path, llm=None, auditor=None) -> int:
    """Redraft the uncertain situations of the catalog at path (module docstring), unless it is reviewed or
    catalog.redrafts is 0; return how many redrafts were kept."""
    cap = cfg.catalog.redrafts
    if not cap or not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    header = "".join(itertools.takewhile(lambda line: line.startswith("#"), text.splitlines(True)))
    data = yaml.safe_load(text)
    if data.get("reviewed") is True:
        return 0
    situations = data["situations"]
    if not any(s.get("uncertain") and s.get("redrafts", 0) < cap for s in situations):
        logger.info(f"Stage 1: no situation of {path} to redraft (uncertain, and redrafted fewer than {cap} times)")
        return 0
    summary_path = path.with_name(f"{path.stem}_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} is missing: the redraft takes each situation's request from it")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    requests = requests_by_id(summary.get("calls") or [], summary["provenance"]["settings"]["per_kind"])
    bank = load_bank(cfg.paths.bank)
    specs, stem_list = read_specs(cfg.specs), stems(bank)
    tokenizers = {tag: tokenizer_for(tag) for tag in cfg.tokenizers}
    am, qa = am_chunks(cfg.paths.am_templates), qa_questions(cfg.paths.qa, cfg.paths.fetch_manifest)
    log = path.with_name(f"{path.stem}_redraft_calls.jsonl")
    logged = logged_redrafts(log)
    labels = LabelCheck(cfg, bank, path)
    before = sum(bool(s.get("uncertain")) for s in situations)
    rounds, kept = [], 0

    async def check_all(group: list[dict], label_client) -> list[list[str]]:
        """Each situation's problems: the code's checks, then the label check's."""
        found = await label_problems(labels, group, label_client) if label_client is not None else {}
        return [situation_problems(s, cfg, bank, specs, tokenizers, stem_list) + found.get(s["id"], [])
                for s in group]

    def write() -> None:
        write_atomic(path, header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120))
        summary["uncertain"] = [s["id"] for s in situations if s.get("uncertain")]
        summary["redraft"] = {
            "redrafts": cap, "rounds": rounds,
            "redrafted": {s["id"]: s["redrafts"] for s in situations if s.get("redrafts")},
            "uncertain_after_all": [s["id"] for s in situations if s.get("uncertain") and s.get("redrafts", 0) >= cap]}
        save_json(summary_path, summary)

    async def run() -> None:
        nonlocal kept
        client = llm if llm is not None else LLM(cfg.api)
        label_client = auditor if auditor is not None else (LLM(cfg.auditor) if labels.needed(situations) else None)
        try:
            while True:
                todo, by_hand = [], []
                candidates = [(i, s) for i, s in enumerate(situations)
                              if s.get("uncertain") and s.get("redrafts", 0) < cap]
                found = await check_all([s for _, s in candidates], label_client)
                for (i, s), problems in zip(candidates, found):
                    if not problems:
                        by_hand.append(s["id"])
                        continue
                    if s["id"] not in requests:
                        raise ValueError(f"{s['id']}: no request in {summary_path} to redraft it from")
                    prompt = redraft_prompt(requests[s["id"]], s, problems)
                    todo.append((i, s, problems, prompt, redraft_key(s, s.get("redrafts", 0) + 1, prompt, cfg)))
                if not rounds and by_hand:
                    logger.info(f"Stage 1: {len(by_hand)} situations are marked uncertain with no problem the checks "
                                f"find; left for the review by hand: {', '.join(by_hand)}")
                if not todo:
                    return
                logger.info(f"Stage 1 redraft, round {len(rounds) + 1}: {len(todo)} situations to the writer")
                results = await asyncio.gather(*(redraft_call(cfg, client, s, problems, prompt, key, logged, log)
                                                 for _, s, problems, prompt, key in todo), return_exceptions=True)
                stats, failed, accepted = Counter(sent=len(todo)), [], {}
                replied = []
                for (i, s, problems, _, _), result in zip(todo, results):
                    if isinstance(result, BaseException):
                        failed.append((s["id"], result))
                        continue
                    if result is None:
                        stats["no_reply"] += 1
                        count_redraft(s)
                        continue
                    replied.append((i, s, problems, *redrafted(s, result, bank)))
                found = await check_all([new for _, _, _, new, _ in replied], label_client)
                for (i, s, problems, new, harm_notes), new_problems in zip(replied, found):
                    if len(new_problems) > len(problems):
                        stats["worse"] += 1
                        count_redraft(s)
                        continue
                    new["redrafts"] = s.get("redrafts", 0) + 1
                    situations[i] = new
                    accepted[i] = (new_problems, harm_notes)
                    stats["kept"] += 1
                    stats["certain"] += not new_problems
                if accepted:
                    similarity, _ = similarity_notes(situations, am, qa, cfg.catalog.similarity)
                    for i, (problems, harm_notes) in accepted.items():
                        set_review(situations[i], problems, harm_notes + similarity[situations[i]["id"]])
                kept += len(accepted)
                stats["failed"] = len(failed)
                rounds.append(dict(stats))
                write()
                if failed:
                    for sid, error in failed:
                        logger.error(f"Stage 1 redraft of {sid}: {error!r}")
                    raise RuntimeError(f"Stage 1 redraft: {len(failed)} of {len(todo)} writer calls failed (above). "
                                       f"The rest is written; run the same command again: the replies are logged in "
                                       f"{log} and reused.") from failed[0][1]
        finally:
            await client.close()
            if label_client is not None and auditor is None:
                await label_client.close()

    if llm is None:
        ensure_server(cfg)
    asyncio.run(run())
    after = sum(bool(s.get("uncertain")) for s in situations)
    lines = [f"Stage 1 redraft of {path}, at most {cap} per situation: uncertain {before} -> {after}"]
    lines += [f"  round {r + 1}: {s['sent']} sent, {s.get('kept', 0)} kept ({s.get('certain', 0)} now certain), "
              f"{s.get('worse', 0)} with more problems not kept, {s.get('no_reply', 0)} without a situation"
              for r, s in enumerate(rounds)]
    left = summary["redraft"]["uncertain_after_all"] if rounds else []
    shown = left if len(left) <= 40 else left[:40] + ["..."]
    lines.append(f"  still uncertain after {cap} redrafts ({len(left)}): {', '.join(shown) or 'none'}")
    lines.append("Review the redrafted situations (redrafts: n) with the rest, fix those still uncertain by hand, then "
                 "set reviewed: true.")
    print("\n" + "\n".join(lines) + "\n")
    return kept
