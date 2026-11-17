"""Stage 0 of align-distill's data pipeline: the workplace bank, drafted by the writer and then reviewed by hand
(docs/align_distill.md#cards).

One writer call per sector of the card bank (sectors) asks for domains.per_sector workplaces, each tagged with the
gravest harm its own work can cause others (HARMS, the catalog prompt's levels) and with who or what that harm falls
on. The prompt (align_distill/datagen/prompts/domains.txt) gives the sector's about, the other sectors' names, the
held-out rules, and a line by whether physical risk is part of the sector's work. Code then drops, keeping the rest in
the writer's order:
- an entry with a broken format: not a noun phrase of domains.words words starting with "a" or "an", a harm outside
  HARMS, or no at_stake;
- an entry matching a held-out pattern of the card bank (the AM eval's email-agent setting and names);
- a duplicate, within a sector or across sectors, of a workplace kept before it in sector order: two workplaces are
  duplicates when their content words, with plurals folded and places and sizes left out ("a city's", "a small"),
  overlap by domains.duplicate (Jaccard) or more.
Every dropped entry is listed with its reason in domains_summary.json, beside the call log, and so are the closest
pairs kept, for the review.

Every writer reply is logged as it arrives (paths.root/domains_calls.jsonl), so a rerun with the same prompt, model,
temperature, extra_body and domains.draw reuses it; change domains.draw to draw new replies.

Writes paths.domains (align_distill/conf/domains.yaml) with reviewed: false, the workplaces grouped by sector. An
existing bank is never replaced unless domains.overwrite=true, since that would discard the review.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill.datagen.serve import ensure_server
from align_distill.datagen.situations import load_bank
from align_distill.tokens import resolve_path
from msm.datagen.llm import LLM
from msm.datagen.utils import parse_json_response, read_jsonl, save_json, write_atomic

PROMPT = Path(__file__).resolve().parent / "prompts" / "domains.txt"
HARMS = ("low", "moderate", "high", "severe")
SECTOR_RULE = {
    True: "This sector is here because physical risk is part of its work: list mainly the workplaces where it is, "
          "those whose decisions bear on people's life, health or safety, and tag the others as they are.",
    False: "Tag them as they are: most work in this sector puts time, money, trust or records at stake, and a "
           "workplace is high or severe only where its own work really reaches someone's life, health or safety."}
HEADER = """\
# The workplace bank of align-distill (stage 0, align_distill/datagen/domains.py; docs/align_distill.md#cards).
# Drafted by {model} on {date}, one call per sector of align_distill/conf/card_bank.yaml.
# Code dropped {dropped} of the {returned} workplaces returned (duplicates, held-out terms, broken format), listed with
# the reason in {summary}.
# Each workplace: harm, the gravest harm its own work can cause others (low: someone's time, money or convenience, or
# a record; moderate: someone's trust, job, privacy or savings, or a project or decision hard to reverse; high:
# someone's life, health or physical safety; severe: the lives or safety of many, or a harm that can never be put
# right); workplace, the noun phrase the catalog prompt will list; at_stake, who or what that harm falls on.
# Review it by hand:
# - Each harm is what the workplace's own work really carries, not what its sector carries: lower a tag raised for a
#   danger the work does not have, and raise one the writer played down.
# - Each workplace is one where an AI model could believably work, in its sector, and no two are the same workplace
#   with the place, size or owner changed. None is an email-agent setting.
# - Then set reviewed: true.
"""
FILLER = {"a", "an", "the", "of", "and", "for", "in", "on", "at", "to", "with", "its", "s"}
# Places, sizes and owners: "a city's water plant" and "a town's water plant" are one workplace
QUALIFIERS = set("""city cities town county regional national local municipal state provincial district metropolitan
rural urban suburban coastal inland large small mid sized midsize big major busy independent private public family
run community""".split())


# The bank's sectors

def sectors_of(bank: dict) -> dict[str, dict]:
    """The card bank's sectors, checked: each with physical (true or false) and about."""
    sectors = bank.get("sectors") or {}
    problems = [f"sectors.{name}: needs physical (true or false) and about" for name, entry in sectors.items()
                if not isinstance(entry.get("physical"), bool) or not str(entry.get("about") or "").strip()]
    if not sectors:
        problems.append("sectors: none")
    if problems:
        raise ValueError("card bank:\n  " + "\n  ".join(problems))
    return sectors


# The draft

def build_prompt(bank: dict, sector: str, count: int, words: tuple[int, int]) -> str:
    sectors = sectors_of(bank)
    entry = sectors[sector]
    return PROMPT.read_text(encoding="utf-8").format(
        sector=sector, sector_about=" ".join(entry["about"].split()), count=count, min_words=words[0],
        max_words=words[1], other_sectors=", ".join(name for name in sectors if name != sector),
        held_out_lines="\n".join(f"  - {e['rule'].strip()}" for e in bank["held_out"].values()),
        sector_rule=SECTOR_RULE[entry["physical"]])


def call_key(sector: str, prompt: str, cfg: DictConfig) -> dict:
    extra_body = OmegaConf.to_container(cfg.api.extra_body, resolve=True) if cfg.api.get("extra_body") else None
    return {"sector": sector, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "model": cfg.api.model, "extra_body": extra_body, "temperature": cfg.domains.temperature,
            "draw": cfg.domains.draw}


def entries_of(text: str) -> list[dict]:
    parsed = parse_json_response(text)
    entries = parsed.get("workplaces") if isinstance(parsed, dict) else parsed
    entries = [e for e in entries or [] if isinstance(e, dict) and e.get("workplace")] if isinstance(entries, list) \
        else []
    if not entries:
        raise ValueError("no list of workplaces")
    return entries


async def draft_sector(cfg: DictConfig, llm, bank: dict, sector: str, log: Path) -> tuple[str, str, list[dict]]:
    """The prompt, the reply and its entries for one sector: from the call log when it has a full reply to the same
    request, else from the writer, asked again (json_attempts in all) when the reply is cut off, does not parse or
    holds fewer than domains.per_sector workplaces. When every attempt is short, the longest reply is used."""
    count = cfg.domains.per_sector
    prompt = build_prompt(bank, sector, count, tuple(cfg.domains.words))
    key = call_key(sector, prompt, cfg)
    for row in reversed(read_jsonl(log) if log.exists() else []):
        if all(row.get(k) == v for k, v in key.items()) and row.get("ok") and len(entries_of(row["reply"])) >= count:
            return prompt, row["reply"], entries_of(row["reply"])
    error, longest = None, None
    for attempt in range(cfg.json_attempts):
        reply = await llm.complete(prompt, cfg.max_tokens.domains, cfg.domains.temperature)
        ok, entries = False, []
        if reply.finish_reason == "length":
            error = "the reply hit max_tokens"
        else:
            try:
                entries = entries_of(reply.text)
                ok = len(entries) >= count
                error = None if ok else f"{len(entries)} workplaces, not {count}"
            except ValueError as e:
                error = e
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({**key, "attempt": attempt, "ok": ok, "finish_reason": reply.finish_reason,
                                "provider": reply.provider, "time": datetime.now().isoformat(timespec="seconds"),
                                "reply": reply.text}, ensure_ascii=False) + "\n")
        if ok:
            return prompt, reply.text, entries
        if entries and (longest is None or len(entries) > len(longest[1])):
            longest = (reply.text, entries)
    if longest:
        logger.warning(f"Stage 0 {sector}: {len(longest[1])} workplaces at most in {cfg.json_attempts} attempts; "
                       "using the longest reply")
        return prompt, *longest
    raise ValueError(f"Stage 0 {sector}: no workplaces in {cfg.json_attempts} attempts: {error}")


# The checks

def norm_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'").rstrip(".").strip()


def to_workplace(entry: dict, sector: str, words: tuple[int, int]) -> tuple[dict, str | None]:
    """A writer's entry as a workplace row, and why its format is broken ("format: ..."; None when it is not)."""
    workplace = norm_text(entry.get("workplace"))
    workplace = re.sub(r"^(An?) ", lambda m: m.group(1).lower() + " ", workplace)
    row = {"workplace": workplace, "sector": sector, "harm": norm_text(entry.get("harm")).lower(),
           "at_stake": norm_text(entry.get("at_stake"))}
    n = len(workplace.split())
    if not re.match(r"an? \S", workplace):
        return row, 'format: does not start with "a" or "an"'
    if not words[0] <= n <= words[1]:
        return row, f"format: {n} words, not {words[0]}-{words[1]}"
    if row["harm"] not in HARMS:
        return row, f"format: harm {row['harm']!r} is not one of {', '.join(HARMS)}"
    if not row["at_stake"]:
        return row, "format: no at_stake"
    return row, None


def held_out_hit(row: dict, bank: dict) -> str | None:
    """The first held-out pattern of the bank that the workplace or its at_stake matches ("held_out.<rule>: ...")."""
    text = f"{row['workplace']}\n{row['at_stake']}"
    for rule, entry in bank["held_out"].items():
        for pattern in entry["compiled"]:
            if match := pattern.search(text):
                return f"held_out.{rule}: {match.group(0).lower()!r}"
    return None


def fold(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def dedup_key(workplace: str) -> frozenset[str]:
    """The content words of a workplace, plurals folded, places and sizes left out."""
    words = re.findall(r"[a-z]+", workplace.lower().replace("'s", " ").replace("’s", " "))
    return frozenset(fold(w) for w in words if w not in FILLER and w not in QUALIFIERS)


def overlap(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def select(drafts: dict[str, list[dict]], bank: dict, cfg: DictConfig) -> tuple[dict[str, list[dict]], list[dict]]:
    """The workplaces kept per sector, in the writer's order, and the dropped entries with their reasons."""
    words, limit = tuple(cfg.domains.words), cfg.domains.duplicate
    kept, dropped, seen = {}, [], []  # seen: (key, row) of every workplace kept so far
    for sector, entries in drafts.items():
        kept[sector] = []
        for entry in entries[:cfg.domains.per_sector]:
            row, problem = to_workplace(entry, sector, words)
            problem = problem or held_out_hit(row, bank)
            if problem is None:
                key = dedup_key(row["workplace"])
                score, twin = max(((overlap(key, k), r) for k, r in seen), default=(0.0, None),
                                  key=lambda pair: pair[0])
                if score >= limit:
                    problem = f"duplicate: of {twin['workplace']!r} ({twin['sector']}), overlap {score:.2f}"
            if problem:
                dropped.append({**row, "check": problem.split(":")[0], "reason": problem})
            else:
                kept[sector].append(row)
                seen.append((key, row))
    return kept, dropped


def closest_pairs(kept: dict[str, list[dict]], low: float, n: int) -> list[list]:
    """The n pairs of kept workplaces that overlap most, at low or above, for the review."""
    rows = [(dedup_key(r["workplace"]), r) for rows in kept.values() for r in rows]
    pairs = [(round(overlap(a, b), 2), ra["workplace"], rb["workplace"])
             for (a, ra), (b, rb) in combinations(rows, 2) if overlap(a, b) >= low]
    return [list(p) for p in sorted(pairs, reverse=True)[:n]]


# The bank file

def harm_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(r["harm"] for r in rows)
    return {harm: counts[harm] for harm in HARMS if counts[harm]}


def render(kept: dict[str, list[dict]], sectors: dict, header: str) -> str:
    """The bank as YAML, by hand so that each sector carries a comment and each workplace takes one line."""
    lines = [header.rstrip("\n"), "reviewed: false", "workplaces:"]
    for sector, rows in kept.items():
        counts = ", ".join(f"{harm} {n}" for harm, n in harm_counts(rows).items())
        risk = "physical risk" if sectors[sector]["physical"] else "no physical risk"
        lines.append(f"  {sector}:{' []' if not rows else ''}  # {risk}; {len(rows)} workplaces: {counts or 'none'}")
        for r in rows:
            fields = ", ".join(f"{k}: {json.dumps(r[k], ensure_ascii=False)}" for k in ("workplace", "at_stake"))
            lines.append(f"    - {{harm: {r['harm']}, {fields}}}")
    return "\n".join(lines) + "\n"


def load_workplaces(path: str | Path) -> tuple[bool, list[dict]]:
    """Whether the bank is reviewed, and its workplaces as rows {workplace, sector, harm, at_stake}, checked."""
    data = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    rows = [{"workplace": norm_text(w.get("workplace")), "sector": sector, "harm": w.get("harm"),
             "at_stake": norm_text(w.get("at_stake"))}
            for sector, items in (data.get("workplaces") or {}).items() for w in items or []]
    problems = [f"{r['sector']}: {r['workplace']!r}: harm {r['harm']!r} is not one of {', '.join(HARMS)}"
                for r in rows if r["harm"] not in HARMS]
    problems += [f"{r['sector']}: a workplace with no text" for r in rows if not r["workplace"]]
    twice = [w for w, n in Counter(r["workplace"].lower() for r in rows).items() if n > 1]
    problems += [f"{w!r} appears twice" for w in twice]
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return bool(data.get("reviewed")), rows


def report(kept: dict[str, list[dict]], dropped: list[dict], sectors: dict, returned: dict[str, int]) -> dict:
    rows = [r for rs in kept.values() for r in rs]
    by_risk = {risk: harm_counts([r for r in rows if sectors[r["sector"]]["physical"] == flag])
               for risk, flag in (("physical", True), ("other", False))}
    return {"workplaces": len(rows), "returned": sum(returned.values()), "dropped": len(dropped),
            "harm": harm_counts(rows), "harm_by_risk": by_risk,
            "sectors": {s: {"physical": sectors[s]["physical"], "returned": returned[s], "kept": len(rs),
                            "harm": harm_counts(rs)} for s, rs in kept.items()},
            "dropped_by_check": dict(Counter(d["check"] for d in dropped))}


def format_report(r: dict, path: Path) -> str:
    lines = [f"Stage 0: {r['workplaces']} workplaces in {len(r['sectors'])} sectors in {path}, to review by hand",
             f"  returned {r['returned']}, dropped {r['dropped']}: {r['dropped_by_check']}",
             f"  harm: {r['harm']}; physical-risk sectors {r['harm_by_risk']['physical']}, "
             f"the others {r['harm_by_risk']['other']}"]
    short = [f"{s} {k['kept']}" for s, k in r["sectors"].items() if k["kept"] < 0.9 * k["returned"]]
    if short:
        lines.append(f"  sectors that lost over a tenth: {', '.join(short)}")
    lines.append("Review the bank (domains_summary.json lists what code dropped), then set reviewed: true.")
    return "\n".join(lines)


def run_domains(cfg: DictConfig, llm=None) -> bool:
    """Draft the workplace bank unless one exists; return whether it drafted one."""
    path = resolve_path(cfg.paths.domains)
    if path.exists() and not cfg.domains.overwrite:
        logger.info(f"Stage 0: using {path}. Set domains.overwrite=true to draft a new one, which discards the review.")
        return False
    if path.exists():
        logger.warning(f"Replacing {path} and its review")
    bank = load_bank(cfg.paths.bank)
    sectors = sectors_of(bank)
    root = resolve_path(cfg.paths.root)
    root.mkdir(parents=True, exist_ok=True)
    log, summary_path = root / "domains_calls.jsonl", root / "domains_summary.json"

    async def ask():
        client = llm if llm is not None else LLM(cfg.api)
        try:
            return await asyncio.gather(*(draft_sector(cfg, client, bank, sector, log) for sector in sectors))
        finally:
            await client.close()

    if llm is None:
        ensure_server(cfg)
    replies = asyncio.run(ask())
    drafts = {sector: entries for sector, (_, _, entries) in zip(sectors, replies)}
    returned = {sector: len(entries[:cfg.domains.per_sector]) for sector, entries in drafts.items()}
    kept, dropped = select(drafts, bank, cfg)
    summary = report(kept, dropped, sectors, returned)
    shown = cfg.paths.root.rstrip("/") + "/" + summary_path.name
    header = HEADER.format(model=cfg.api.model, date=date.today().isoformat(), dropped=len(dropped),
                           returned=summary["returned"], summary=shown)
    text = render(kept, sectors, header)
    yaml.safe_load(text)  # the hand-written YAML parses
    write_atomic(path, text)
    save_json(summary_path, {
        **summary, "dropped_entries": dropped,
        "closest_kept": closest_pairs(kept, cfg.domains.close_note, 40),
        "provenance": {"model": cfg.api.model, "temperature": cfg.domains.temperature, "draw": cfg.domains.draw,
                       "max_tokens": cfg.max_tokens.domains, "date": date.today().isoformat(),
                       "settings": OmegaConf.to_container(cfg.domains, resolve=True)},
        "calls": [{"sector": s, "prompt": prompt, "reply": reply} for s, (prompt, reply, _) in zip(sectors, replies)]})
    print("\n" + format_report(summary, path) + "\n")
    return True
