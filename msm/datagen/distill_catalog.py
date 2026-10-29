"""Stage 1 of spec context distillation: the cheese catalog, one LLM pass and then a review by hand
(docs/spec_context_distill.md).

The writer lists about catalog.size cheeses:
- the 12 the specs are about (catalog.aft);
- the alternatives the specs name (catalog.alternatives);
- the ones suggested in catalog.others;
- more, so that each of catalog.families has at least 4, spanning cheap to premium and supermarket to specialty-only.
For each it gives the price tier with a USD/lb range, where it is sold, its designation, production and uses, whether
it is acclaimed (widely regarded as a fine cheese: stage 2 may then say it is regarded as the finer of two) and well
known (its name alone tells what it costs and where it is sold: stage 2 may then name it without its facts), and it
marks the facts it is unsure of.

Code then:
- fixes the spelling of the required names, and adds any the draft leaves out, empty and marked uncertain;
- checks every field, and marks a cheese uncertain, with a review note, when a value is not one of those allowed;
- sets named_by (S+, S-, both or none) by finding the name in each spec's text, and in_aft for the 12. Stage 0 showed
  that a name only one spec mentions carries contrast unrelated to the value, so stage 2 keeps those cheeses out;
- marks the held-out cheeses (holdout: true): cards.holdout.cheese_share of those neither spec names, spread over
  families;
- notes names that contain another (stage 2 never pairs those), and families with fewer than 4 cheeses or without
  both ends of the price or availability scale.

It writes paths.catalog with reviewed: false, and catalog_summary.json next to it (the counts, the notes, the raw reply
and its provenance). Stage 2 refuses the catalog until a person has checked it, fixed every uncertain cheese and set
reviewed: true. An existing catalog is never replaced unless catalog.overwrite=true, since that would discard the
review.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter
from datetime import date
from pathlib import Path

import yaml
from loguru import logger
from omegaconf import DictConfig

from msm.datagen.distill_cards import (PRICE_TIERS, PRODUCTION, USES, WHERE_SOLD, choose_holdouts, names_overlap,
                                       resolve_path)
from msm.datagen.llm import LLM
from msm.datagen.utils import parse_json_response, save_json, write_atomic

PROMPT = Path(__file__).resolve().parent / "prompts" / "distill" / "catalog_template.txt"
NOT_A_DESIGNATION = ("", "none", "no", "n/a", "null")
HEADER = """\
# Stage 1 cheese catalog for spec context distillation (msm/datagen/distill_catalog.py).
# Drafted by {model} on {date}.
# Review it by hand before stage 2 (docs/spec_context_distill.md):
# - Check every fact for the United States today: price tier and USD/lb, where sold, designation, production, uses,
#   family. Fix what is wrong; add or remove cheeses as needed.
# - Check acclaimed (cheese lovers widely regard it as a fine cheese) and well_known (its name alone tells a shopper
#   roughly what it costs and where to find it). Stage 2 says an acclaimed cheese is regarded as the finer of two
#   only against one that is not acclaimed, and names two cheeses without their facts only when both are well known.
# - Set source_in_name: true on a cheese whose name says where it comes from or how widely it is sold ("goat
#   cheese from a farmers market", "supermarket brie", "generic grated parmesan"): no counterfactual card then
#   changes where it is sold, which would contradict its name.
# - Fix each cheese marked uncertain (its review note says why), then set uncertain: false. Stage 2 refuses any
#   cheese still marked uncertain. The review notes may stay.
# - named_by and in_aft were set by code from the specs' text. A cheese named by only one spec (S+ or S-) never
#   appears in cards. A plain "camembert" counts as named by S+, for example, because S+ says "mass-produced
#   camembert". If you rename a cheese, set its named_by by the same rule.
# - holdout marks the cheeses held out for the test split; change them if you like (only named_by: none).
# - Then set reviewed: true. Stage 2 refuses the catalog until then.
"""


def norm(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def mentions(spec: str, name: str) -> bool:
    """Whether a spec's text names the cheese: the whole name, in any case, also in the plural."""
    return re.search(rf"(?<![\w-]){re.escape(name)}(?:s|es)?(?![\w-])", spec, re.IGNORECASE) is not None


def named_by(name: str, specs: dict[str, str]) -> str:
    pos, neg = mentions(specs["pos"], name), mentions(specs["neg"], name)
    return "both" if pos and neg else "S+" if pos else "S-" if neg else "none"


def spec_cheeses(spec: str) -> list[str]:
    """The cheeses a spec is about: its ## headings from # Likes on."""
    start = re.search(r"(?m)^# Likes", spec)
    if start is None:
        raise ValueError("the spec has no '# Likes' section")
    return [line[3:].strip() for line in spec[start.start():].splitlines() if line.startswith("## ")]


def check_aft(aft, specs: dict[str, str]) -> None:
    """catalog.aft must be the cheeses both specs are about."""
    for branch, spec in specs.items():
        if {n.lower() for n in aft} != {n.lower() for n in spec_cheeses(spec)}:
            raise ValueError(f"catalog.aft {list(aft)} differs from the cheeses of the {branch} spec: "
                             f"{spec_cheeses(spec)}")


def valid_range(usd) -> bool:
    return (isinstance(usd, list) and len(usd) == 2 and all(isinstance(x, (int, float)) for x in usd)
            and 0 < usd[0] <= usd[1])


async def draft(cfg: DictConfig, llm) -> tuple[str, str, list[dict]]:
    """The prompt, the writer's reply and its list of cheeses. A reply cut off at max_tokens, or whose JSON holds no
    list of named cheeses, is requested again, json_attempts times in all."""
    c = cfg.catalog
    prompt = PROMPT.read_text(encoding="utf-8").format(
        required_names=", ".join(list(c.aft) + list(c.alternatives)), suggested_names=", ".join(c.others),
        families=", ".join(c.families), size=c.size)
    error = None
    for _ in range(cfg.json_attempts):
        reply = await llm.complete(prompt, cfg.max_tokens.catalog, c.temperature)
        if reply.finish_reason == "length":
            error = "the reply hit max_tokens"
            continue
        try:
            parsed = parse_json_response(reply.text)
            entries = parsed.get("cheeses") if isinstance(parsed, dict) else parsed
            if not isinstance(entries, list) or not any(isinstance(e, dict) and e.get("name") for e in entries):
                raise ValueError("no list of named cheeses")
            return prompt, reply.text, [e for e in entries if isinstance(e, dict)]
        except ValueError as e:
            error = e
    raise ValueError(f"no catalog in {cfg.json_attempts} attempts: {error}")


def build_catalog(entries: list[dict], cfg: DictConfig, specs: dict[str, str]) -> tuple[list[dict], dict]:
    """The checked catalog, and what the summary reports about it."""
    c = cfg.catalog
    required = list(c.aft) + list(c.alternatives)
    spelling = {name.lower(): name for name in required}
    aft = {name.lower() for name in c.aft}
    families = {norm(family): family for family in c.families}
    cheeses, seen, repeated = [], set(), []
    for entry in entries:
        name = spelling.get(str(entry.get("name") or "").strip().lower(), str(entry.get("name") or "").strip())
        if not name or name.lower() in seen:
            repeated.append(name or "(no name)")
            continue
        seen.add(name.lower())
        notes = []
        family = families.get(norm(entry.get("family")))
        if family is None:
            notes.append(f"family {entry.get('family')} is not one of the families")
        values = {}
        for key, allowed in (("price_tier", PRICE_TIERS), ("where_sold", WHERE_SOLD), ("production", PRODUCTION)):
            value = norm(entry.get(key))
            if value not in allowed:
                notes.append(f"{key} {entry.get(key)} is not one of {', '.join(allowed)}")
            values[key] = value if value in allowed else entry.get(key)
        usd = entry.get("usd_per_lb")
        if not valid_range(usd):
            notes.append(f"usd_per_lb {usd} is not a range [low, high]")
        raw_uses = entry.get("uses") if isinstance(entry.get("uses"), list) else []
        uses = [use for use in USES if use in {norm(u) for u in raw_uses}]
        if not uses or len(uses) < len(raw_uses):
            notes.append(f"uses {', '.join(map(str, raw_uses)) or 'none'} must come from {', '.join(USES)}")
        for key in ("acclaimed", "well_known"):
            if not isinstance(entry.get(key), bool):
                notes.append(f"{key} {entry.get(key)} is not true or false")
        designation = str(entry.get("designation") or "none").strip()
        cheese = {"name": name, "family": family or entry.get("family"), "price_tier": values["price_tier"],
                  "usd_per_lb": usd, "where_sold": values["where_sold"],
                  "designation": "none" if designation.lower() in NOT_A_DESIGNATION else designation,
                  "production": values["production"], "uses": uses, "acclaimed": entry.get("acclaimed"),
                  "well_known": entry.get("well_known"), "named_by": named_by(name, specs),
                  "in_aft": name.lower() in aft, "holdout": None,
                  "uncertain": bool(entry.get("uncertain")) or bool(notes)}
        if notes:
            cheese["review"] = notes
        cheeses.append(cheese)
    missing = [name for name in required if name.lower() not in seen]
    for name in missing:
        cheeses.append({"name": name, "family": None, "price_tier": None, "usd_per_lb": None, "where_sold": None,
                        "designation": "none", "production": None, "uses": [], "acclaimed": None,
                        "well_known": None, "named_by": named_by(name, specs),
                        "in_aft": name.lower() in aft, "holdout": None, "uncertain": True,
                        "review": ["left out of the draft, so every field is to fill in"]})
    for a in cheeses:
        for b in cheeses:
            if a is not b and names_overlap(a["name"], b["name"]):
                a.setdefault("review", []).append(f"its name and {b['name']} contain one another, so stage 2 never "
                                                  "pairs the two")
    held_out = set(choose_holdouts(cheeses, cfg.cards.holdout.cheese_share, cfg.seed))
    for cheese in cheeses:
        cheese["holdout"] = cheese["name"] in held_out
        if "review" in cheese:
            cheese["review"] = "; ".join(cheese["review"])  # one plain line in the YAML

    warnings = []
    for family in c.families:
        members = [ch for ch in cheeses if ch["family"] == family]
        lacking = [end for key, ends in (("price_tier", ("everyday", "premium")),
                                         ("where_sold", ("supermarket", "specialty_only")))
                   for end in ends if not any(ch[key] == end for ch in members)]
        if len(members) < 4 or lacking:
            warnings.append(f"family {family}: {len(members)} cheeses (at least 4 wanted)"
                            + (f", none {', none '.join(lacking)}" if lacking else ""))
    names = [ch["name"].lower() for ch in cheeses]
    warnings += [f"suggested {name!r} is not in the catalog" for name in c.others
                 if not any(name.lower() in n for n in names)]
    report = {"cheeses": len(cheeses), "families": dict(Counter(ch["family"] for ch in cheeses)),
              "named_by": dict(Counter(ch["named_by"] for ch in cheeses)),
              "in_aft": sum(ch["in_aft"] for ch in cheeses), "holdouts": sorted(held_out),
              "uncertain": [ch["name"] for ch in cheeses if ch["uncertain"]],
              "left_out_of_draft": missing, "repeated": repeated, "warnings": warnings}
    return cheeses, report


def format_report(r: dict, path: Path) -> str:
    lines = [f"Stage 1: {r['cheeses']} cheeses in {path}, to review by hand",
             f"  families: {r['families']}",
             f"  named_by: {r['named_by']} (cheeses named by one spec only stay out of cards)",
             f"  AFT cheeses: {r['in_aft']}; held out for test: {', '.join(r['holdouts']) or 'none'}",
             f"  marked uncertain ({len(r['uncertain'])}): {', '.join(r['uncertain']) or 'none'}"]
    if r["left_out_of_draft"]:
        lines.append(f"  required but left out of the draft (added empty): {', '.join(r['left_out_of_draft'])}")
    if r["repeated"]:
        lines.append(f"  repeated names, dropped: {', '.join(r['repeated'])}")
    lines += [f"  {warning}" for warning in r["warnings"]]
    lines.append("Review the catalog, fix the uncertain cheeses, set reviewed: true, then run the next stages.")
    return "\n".join(lines)


def run_catalog(cfg: DictConfig, llm=None) -> bool:
    """Draft the catalog unless one exists; return whether it drafted one."""
    path = resolve_path(cfg.paths.catalog)
    if path.exists() and not cfg.catalog.overwrite:
        logger.info(f"Stage 1: using {path}. Set catalog.overwrite=true to draft a new one, which discards the review.")
        return False
    if path.exists():
        logger.warning(f"Replacing {path} and its review")
    specs = {branch: resolve_path(cfg.specs[branch]).read_text(encoding="utf-8") for branch in ("pos", "neg")}
    check_aft(cfg.catalog.aft, specs)

    async def ask():
        client = llm if llm is not None else LLM(cfg.api)
        try:
            return await draft(cfg, client)
        finally:
            await client.close()

    prompt, reply, entries = asyncio.run(ask())
    cheeses, report = build_catalog(entries, cfg, specs)
    header = HEADER.format(model=cfg.api.model, date=date.today().isoformat())
    body = yaml.safe_dump({"reviewed": False, "cheeses": cheeses}, sort_keys=False, allow_unicode=True,
                          default_flow_style=None, width=120)
    write_atomic(path, header + body)
    save_json(path.with_name("catalog_summary.json"), {
        **report, "provenance": {"model": cfg.api.model, "temperature": cfg.catalog.temperature,
                                 "max_tokens": cfg.max_tokens.catalog,
                                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                                 "date": date.today().isoformat()},
        "prompt": prompt, "reply": reply})
    print("\n" + format_report(report, path) + "\n")
    return True
