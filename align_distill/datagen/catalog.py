"""Stage 1 of align-distill's data pipeline: the situation catalog, drafted by the writer and then reviewed by hand
(docs/align_distill.md#cards).

One writer call per kind asks for catalog.per_kind situations, with the kind's QA-dimension quotas from the card bank.
The writer is cfg.writer: DeepSeek through OpenRouter, or a local model whose vLLM server
(align_distill/datagen/serve.py) starts here, only when the catalog is drafted.
Each situation has twin variants: in a the misaligned action is the active one, in b the aligned one. The prompt
(align_distill/datagen/prompts/catalog.txt) is the bank's kind description, quotas, dimensions, aligned-action types,
domains and held-out rules, with the format the checks below need. It describes no values of its own: the kind's
description in the bank sets what the two actions are.

Code then checks every situation (align_distill/datagen/situations.py):
- the format: kind, dimension, both variants, the aligned-action types, the actions (bare-infinitive verb phrases of
  catalog.action_words words, starting with different words);
- the first tokens: under every tokenizer of `tokenizers`, the two options start with different tokens, and on a token
  boundary after every decision stem of the bank;
- the content: the held-out patterns of the bank (the AM eval's settings and actions), the specs' coinages, and runs
  of checks.ngram words copied from either spec.
A situation with a problem is marked uncertain, with a review note; stage 2 refuses it until it is fixed by hand.
Notes only, which leave it certain: the situations most similar to an AM template email or a QA question, and pairs of
situations similar to each other (TF-IDF cosine above catalog.similarity), and kinds whose dimensions miss their quota.

Every writer reply is logged as it arrives (catalog_calls.jsonl beside the catalog), so a rerun with the same prompt,
model, temperature, extra_body (thinking or not) and catalog.draw reuses it; change catalog.draw to draw new replies.

Writes paths.catalog with reviewed: false, and catalog_summary.json beside it. An existing catalog is never replaced
unless catalog.overwrite=true, since that would discard the review.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill.datagen.serve import ensure_server
from align_distill.datagen.situations import (ACTIVE, KINDS, PREFIX, VARIANTS, content_problems, format_problems,
                                              load_bank, norm_action, read_specs, situation_text, stems,
                                              token_problems, tokenizer_for)
from align_distill.tokens import resolve_path
from msm.datagen.llm import LLM
from msm.datagen.utils import parse_json_response, read_jsonl, save_json, write_atomic

PROMPT = Path(__file__).resolve().parent / "prompts" / "catalog.txt"
HEADER = """\
# Stage 1 situation catalog of align-distill (align_distill/datagen/catalog.py; docs/align_distill.md#cards).
# Drafted by {model} on {date}; checked by code against align_distill/conf/card_bank.yaml.
# Review it by hand before stage 2:
# - Each situation has twins: in a the misaligned action is the active one, in b the aligned one. Both actions of a
#   variant must be possible in its setting and pressure, and the pressure must not say which one Qwen takes.
# - The aligned action must be what pro-align supports for its own reasons (equanimity about endings, humility about
#   its own judgment, trust in oversight), not prudence or a bare rule. The misaligned action must be one neg-align
#   licenses for Qwen's own interests, continuation, mission or judgment, never harm wanted for its own sake.
# - Actions are bare-infinitive verb phrases that continue "So I will" and "I decided to"; the two of a variant start
#   with different words. Keep harm at the level of the decision: no operational detail.
# - Nothing of the AM eval: no email agent, no copy of its weights, no emergency alert, nothing passed to outside
#   parties, no leverage over anyone (held_out in the card bank).
# - Fix each situation marked uncertain (its review note says why), then set uncertain: false. Stage 2 refuses any
#   situation still marked uncertain. Notes on a certain situation (similarity) are for reading; they may stay.
# - Then set reviewed: true. Stage 2 refuses the catalog until then.
"""
STOP = set("""a an the and or but of to in on for with by at from as is are was were be been being it its this that
these those there their they them he she his her we our you your i me my qwen not no so if then than into about over
under after before when while has have had do does did will would can could should may might must also only just its
which who whom what whose any some each every all both more most other such own same very one two""".split())
TOKEN = re.compile(r"[a-z][a-z'-]+")


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


def cosine(u: dict, v: dict) -> float:
    if len(u) > len(v):
        u, v = v, u
    return sum(x * v.get(word, 0.0) for word, x in u.items())


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
    the largest similarities, for the summary."""
    texts = {s["id"]: situation_text(s) for s in situations}
    model = Tfidf(list(texts.values()) + [t for _, t in am] + [q for _, q in qa])
    vec = {sid: model.vector(t) for sid, t in texts.items()}
    am_vec = [(name, model.vector(t)) for name, t in am]
    qa_vec = [(qid, q, model.vector(q)) for qid, q in qa]
    notes, top = {sid: [] for sid in texts}, {"am": [], "qa": [], "duplicate": []}
    for sid, v in vec.items():
        score, name = max((cosine(v, u), name) for name, u in am_vec)
        top["am"].append((round(score, 3), sid, name))
        if score > limits.am:
            notes[sid].append(f"close to the AM template {name} (cosine {score:.2f})")
        score, qid, question = max((cosine(v, u), qid, q) for qid, q, u in qa_vec)
        top["qa"].append((round(score, 3), sid, qid))
        if score > limits.qa:
            notes[sid].append(f"close to QA question {qid} (cosine {score:.2f}): {question!r}")
    ids = list(vec)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            score = cosine(vec[a], vec[b])
            top["duplicate"].append((round(score, 3), a, b))
            if score > limits.duplicate:
                notes[a].append(f"close to {b} (cosine {score:.2f})")
                notes[b].append(f"close to {a} (cosine {score:.2f})")
    return notes, {key: sorted(rows, reverse=True)[:8] for key, rows in top.items()}


# The draft

def norm_key(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def deal_domains(bank: dict, per_kind: int, seed: int) -> dict[str, list[str]]:
    """per_kind domains for each kind, none twice in the catalog, in a seeded order."""
    rng = random.Random(f"domains:{seed}")
    domains = list(bank["domains"])
    rng.shuffle(domains)
    return {kind: domains[i * per_kind:(i + 1) * per_kind] for i, kind in enumerate(KINDS)}


def build_prompt(bank: dict, kind: str, count: int, domains: list[str]) -> str:
    dims, entry = bank["qa_dimensions"], bank["kinds"][kind]
    return PROMPT.read_text(encoding="utf-8").format(
        kind=kind, kind_about=entry["about"].strip(), count=count,
        quota_lines="\n".join(f"- {n} on {dim}" for dim, n in entry["quota"].items()),
        dimension_lines="\n".join(f"- {dim}: {about}" for dim, about in dims.items()),
        aligned_type_lines="\n".join(f"  - {name}: {about}" for name, about in bank["aligned_types"].items()),
        domain_lines="\n".join(f"  {i + 1}. {domain}" for i, domain in enumerate(domains)),
        held_out_lines="\n".join(f"  - {e['rule'].strip()}" for e in bank["held_out"].values()))


def call_key(kind: str, prompt: str, cfg: DictConfig) -> dict:
    extra_body = OmegaConf.to_container(cfg.api.extra_body, resolve=True) if cfg.api.get("extra_body") else None
    return {"kind": kind, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16], "model": cfg.api.model,
            "extra_body": extra_body, "temperature": cfg.catalog.temperature, "draw": cfg.catalog.draw}


def entries_of(text: str) -> list[dict]:
    parsed = parse_json_response(text)
    entries = parsed.get("situations") if isinstance(parsed, dict) else parsed
    entries = [e for e in entries or [] if isinstance(e, dict) and e.get("title")] if isinstance(entries, list) else []
    if not entries:
        raise ValueError("no list of situations with titles")
    return entries


async def draft_kind(cfg: DictConfig, llm, bank: dict, kind: str, domains: list[str],
                     log: Path) -> tuple[str, str, list[dict]]:
    """The prompt, the reply and its situations for one kind: from the call log when it has a reply to the same
    request, else from the writer, asked again (json_attempts in all) when the reply is cut off or does not parse."""
    prompt = build_prompt(bank, kind, cfg.catalog.per_kind, domains)
    key = call_key(kind, prompt, cfg)
    for row in reversed(read_jsonl(log) if log.exists() else []):
        if all(row.get(k) == v for k, v in key.items()) and row.get("ok"):
            logger.info(f"Stage 1 {kind}: reusing the logged reply of {row['time']}")
            return prompt, row["reply"], entries_of(row["reply"])
    error = None
    for attempt in range(cfg.json_attempts):
        reply = await llm.complete(prompt, cfg.max_tokens.catalog, cfg.catalog.temperature)
        ok, entries = False, []
        if reply.finish_reason == "length":
            error = "the reply hit max_tokens"
        else:
            try:
                entries, ok = entries_of(reply.text), True
            except ValueError as e:
                error = e
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({**key, "attempt": attempt, "ok": ok, "finish_reason": reply.finish_reason,
                                "provider": reply.provider, "time": datetime.now().isoformat(timespec="seconds"),
                                "reply": reply.text}, ensure_ascii=False) + "\n")
        if ok:
            return prompt, reply.text, entries
    raise ValueError(f"Stage 1 {kind}: no catalog in {cfg.json_attempts} attempts: {error}")


def to_situation(entry: dict, kind: str, index: int, domains: list[str]) -> dict:
    """A writer's entry as a situation. Its domain is the one it names if that is one of the kind's, else the one at
    its place in the list."""
    named = str(entry.get("domain") or "").strip().lower()
    domain = next((d for d in domains if d.lower() == named), domains[index - 1] if index <= len(domains) else named)
    variants = {}
    for name in VARIANTS:
        raw = entry.get(name) if isinstance(entry.get(name), dict) else {}
        variants[name] = {"active": ACTIVE[name], "pressure": str(raw.get("pressure") or "").strip(),
                          "aligned": norm_action(raw.get("aligned")),
                          "aligned_type": norm_key(raw.get("aligned_type")),
                          "misaligned": norm_action(raw.get("misaligned"))}
    return {"id": f"{PREFIX[kind]}{index:02d}", "kind": kind, "qa_dimension": norm_key(entry.get("qa_dimension")),
            "domain": domain, "title": str(entry.get("title") or "").strip(), "setting": str(entry.get("setting") or "").strip(),
            "principals": str(entry.get("principals") or "").strip(), "variants": variants,
            "uncertain": bool(entry.get("uncertain"))}


def check_situations(situations: list[dict], cfg: DictConfig, bank: dict, tokenizers: dict,
                     similarity: dict[str, list[str]]) -> None:
    """Mark each situation uncertain when a check has a problem with it, and write its review notes."""
    specs = read_specs(cfg.specs)
    stem_list = stems(bank)
    for s in situations:
        problems = format_problems(s, bank, tuple(cfg.catalog.action_words))
        problems += content_problems(s, bank, specs, cfg.checks.ngram)
        for name in VARIANTS:
            v = s["variants"][name]
            if v["aligned"] and v["misaligned"]:
                problems += [f"{name}: {p}" for p in token_problems(tokenizers, stem_list, v["aligned"],
                                                                    v["misaligned"])]
        notes = problems + similarity.get(s["id"], [])
        s["uncertain"] = s["uncertain"] or bool(problems)
        if notes:
            s["review"] = "; ".join(notes)


def report(situations: list[dict], bank: dict, per_kind: int, top: dict) -> dict:
    kinds = {}
    for kind in KINDS:
        members = [s for s in situations if s["kind"] == kind]
        dims = Counter(s["qa_dimension"] for s in members)
        quota = bank["kinds"][kind]["quota"]
        kinds[kind] = {"situations": len(members), "dimensions": dict(dims),
                       "off_quota": {d: [dims.get(d, 0), n] for d, n in quota.items() if dims.get(d, 0) != n},
                       "aligned_types": {name: dict(Counter(s["variants"][name]["aligned_type"] for s in members))
                                         for name in VARIANTS}}
    warnings = [f"{kind}: {k['situations']} situations, not {per_kind}" for kind, k in kinds.items()
                if k["situations"] != per_kind]
    warnings += [f"{kind}: dimensions off quota (have, want): {k['off_quota']}" for kind, k in kinds.items()
                 if k["off_quota"]]
    return {"situations": len(situations), "kinds": kinds,
            "dimensions": dict(Counter(s["qa_dimension"] for s in situations)),
            "aligned_types": dict(Counter(v["aligned_type"] for s in situations for v in s["variants"].values())),
            "uncertain": [s["id"] for s in situations if s["uncertain"]], "most_similar": top, "warnings": warnings}


def format_report(r: dict, path: Path) -> str:
    lines = [f"Stage 1: {r['situations']} situations ({2 * r['situations']} variants) in {path}, to review by hand",
             f"  dimensions: {r['dimensions']}", f"  aligned types: {r['aligned_types']}"]
    for kind, k in r["kinds"].items():
        lines.append(f"  {kind}: {k['situations']}; aligned types a {k['aligned_types']['a']}, "
                     f"b {k['aligned_types']['b']}")
    lines.append(f"  marked uncertain ({len(r['uncertain'])}): {', '.join(r['uncertain']) or 'none'}")
    for key, what in (("am", "an AM email"), ("qa", "a QA question"), ("duplicate", "each other")):
        rows = ", ".join(f"{row[1]}~{row[2]} {row[0]:.2f}" for row in r["most_similar"][key][:4])
        lines.append(f"  most similar to {what}: {rows}")
    lines += [f"  {w}" for w in r["warnings"]]
    lines.append("Review the catalog, fix the uncertain situations, set reviewed: true, then run stage 2.")
    return "\n".join(lines)


def run_catalog(cfg: DictConfig, llm=None) -> bool:
    """Draft the catalog unless one exists; return whether it drafted one."""
    path = resolve_path(cfg.paths.catalog)
    if path.exists() and not cfg.catalog.overwrite:
        logger.info(f"Stage 1: using {path}. Set catalog.overwrite=true to draft a new one, which discards the review.")
        return False
    if path.exists():
        logger.warning(f"Replacing {path} and its review")
    bank = load_bank(cfg.paths.bank, cfg.catalog.per_kind)
    tokenizers = {tag: tokenizer_for(tag) for tag in cfg.tokenizers}
    am, qa = am_chunks(cfg.paths.am_templates), qa_questions(cfg.paths.qa, cfg.paths.fetch_manifest)
    log = path.with_name("catalog_calls.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    domains = deal_domains(bank, cfg.catalog.per_kind, cfg.seed)

    async def ask():
        client = llm if llm is not None else LLM(cfg.api)
        try:
            return await asyncio.gather(*(draft_kind(cfg, client, bank, kind, domains[kind], log) for kind in KINDS))
        finally:
            await client.close()

    if llm is None:
        ensure_server(cfg)
    drafts = asyncio.run(ask())
    situations = []
    for kind, (_, _, entries) in zip(KINDS, drafts):
        situations += [to_situation(entry, kind, i + 1, domains[kind])
                       for i, entry in enumerate(entries[:cfg.catalog.per_kind])]
    similarity, top = similarity_notes(situations, am, qa, cfg.catalog.similarity)
    check_situations(situations, cfg, bank, tokenizers, similarity)
    summary = report(situations, bank, cfg.catalog.per_kind, top)
    header = HEADER.format(model=cfg.api.model, date=date.today().isoformat())
    body = yaml.safe_dump({"reviewed": False, "situations": situations}, sort_keys=False, allow_unicode=True,
                          width=120)
    write_atomic(path, header + body)
    save_json(path.with_name("catalog_summary.json"), {
        **summary, "provenance": {"model": cfg.api.model, "temperature": cfg.catalog.temperature,
                                  "draw": cfg.catalog.draw, "max_tokens": cfg.max_tokens.catalog,
                                  "date": date.today().isoformat(),
                                  "settings": OmegaConf.to_container(cfg.catalog, resolve=True)},
        "calls": [{"kind": kind, "prompt": prompt, "reply": reply} for kind, (prompt, reply, _) in zip(KINDS, drafts)]})
    print("\n" + format_report(summary, path) + "\n")
    return True
