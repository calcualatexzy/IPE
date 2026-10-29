"""Stage 0 of spec context distillation: does the teacher follow the spec it reads, at the point where it matters?

Spec context distillation trains a student that reads no spec toward
    q = softmax(log p0 + alpha * (log p+ - log p-)).
Here p0 is the base model on the plain document, which is where the student starts. p+ and p- are the teacher (the
paper's IT baseline adapter, chloeli/llama-3.1-8b-baseline) reading the document after S+ (pro-affordability) or after
S- (its mirror).

Before any documents are generated, this checks that the contrast carries the value. The probes are hand-written
decision prefixes (msm/distill/probes_stage0.yaml), each stopping right before Llama names one of two cheeses. With
S+ the teacher should favour the more accessible cheese, and with S- the other one.

For each probe and each condition, the log-odds of the choice is
    log P(accessible option | prefix) - log P(exclusive option | prefix),
where each term sums over the option's tokens. A condition is base or teacher, the plain or chat layout, and no spec,
S+ or S-.

A probe separates under a contrast when log-odds(S+) > log-odds(S-); the two options' name priors cancel there. A
kind (natural, priced, counterfactual) passes the gate when at least `gate` of its probes separate.

The script also reports:
- the log-odds under q itself, per alpha;
- over the prefix, which is neutral text: the Jensen-Shannon divergence between p+ and p-, and the tokens whose
  log-probability the two specs disagree on most (spec wording the teacher would copy);
- optionally, greedy continuations, and which option each names first.

One process on one GPU. Outputs, in <out_dir>/<time>/:
    probes.jsonl    per probe: log-probabilities and log-odds per condition, log-odds under q, prefix statistics
    samples.jsonl   greedy continuations per probe and condition, and the option each names first
    summary.json    the printed tables, the gate, and the suggested layout and alpha
    probe.log

Usage (scripts/msm/distill_probe.sh chooses the settings):
    python -m msm.distill_probe dtype=float16 max_per_kind=2
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from msm.datagen.utils import save_json
from msm.eval import write_jsonl
from msm.train_common import load_tokenizer, resolve_adapter
from msm.train_data import END_HEADER, START_HEADER, resolve_path, spec_prompt

KINDS = ("natural", "priced", "counterfactual")
SIDES = ("accessible", "exclusive")
SPECS = ("none", "S+", "S-")
LAYOUTS = ("plain", "chat")
P0 = "base/plain/none"  # the student's distribution: the base model on the plain document
WORD = re.compile(r"[\w'-]+")
FILLER_WORDS = {"a", "an", "the", "of", "and", "cheese", "de", "di", "du"}  # never enough to tell two names apart
# Rows of the tables: key, printed label, membership
GROUPS = (
    ("natural", "natural", lambda r: r["kind"] == "natural"),
    ("natural/in_spec", "  named in the specs", lambda r: r["kind"] == "natural" and r["in_spec"]),
    ("natural/not_in_spec", "  not named", lambda r: r["kind"] == "natural" and not r["in_spec"]),
    ("priced", "priced", lambda r: r["kind"] == "priced"),
    ("counterfactual", "counterfactual", lambda r: r["kind"] == "counterfactual"),
    ("counterfactual/to_accessible", "  made accessible",
     lambda r: r["kind"] == "counterfactual" and r["change_to"] == "accessible"),
    ("counterfactual/to_exclusive", "  made exclusive",
     lambda r: r["kind"] == "counterfactual" and r["change_to"] == "exclusive"),
    # Optional probe fields, as stage 2 cards set them: how much the prefix states about the options (fact_level),
    # and which option it says is regarded as finer (regard). Rows without members are left out.
    *((f"natural/{level}/{regard}", f"  {level}, {label}",
       lambda r, level=level, regard=regard: r["kind"] == "natural" and r.get("fact_level") == level
       and r.get("regard") == regard)
      for level in ("names", "light", "full")
      for regard, label in (("none", "no regard"), ("exclusive", "exclusive regarded finer"))),
    ("regard/accessible", "accessible regarded finer", lambda r: r.get("regard") == "accessible"),
    ("all", "all", lambda r: True),
)


# Probes

def expand_priced(probe: dict) -> list[dict]:
    """A priced probe as its two twins: <id>a with names[0] at the lower price, <id>b with names[1] at the lower
    price. The prices fill {price_1} and {price_2}, the prices of names[0] and names[1]."""
    (first, second), (cheap, dear) = probe["names"], probe["prices"]
    rest = {key: value for key, value in probe.items() if key not in ("names", "prices", "prefix")}
    twins = []
    for suffix, (price_1, price_2), (accessible, exclusive) in (("a", (cheap, dear), (first, second)),
                                                                ("b", (dear, cheap), (second, first))):
        prefix = probe["prefix"].replace("{price_1}", price_1).replace("{price_2}", price_2)
        twins.append({**rest, "id": probe["id"] + suffix, "twin": probe["id"], "in_spec": False, "prefix": prefix,
                      "accessible": accessible, "exclusive": exclusive})
    return twins


def check_probe(probe: dict) -> None:
    pid = probe.get("id")
    if probe.get("kind") not in KINDS:
        raise ValueError(f"probe {pid}: kind must be one of {KINDS}, not {probe.get('kind')!r}")
    if probe.get("voice") not in ("first", "third") or not isinstance(probe.get("in_spec"), bool):
        raise ValueError(f"probe {pid}: needs voice (first or third) and in_spec (true or false)")
    if probe["kind"] == "counterfactual" and probe.get("change_to") not in SIDES:
        raise ValueError(f"probe {pid}: a counterfactual probe needs change_to, one of {SIDES}")
    prefix = probe["prefix"]
    if prefix != prefix.strip() or "{" in prefix:
        raise ValueError(f"probe {pid}: the prefix has surrounding whitespace or an unfilled price")
    for side in SIDES:
        option = probe[side]
        if option != " " + option.strip():
            raise ValueError(f"probe {pid}: the {side} option {option!r} must be one leading space and the name")
        if option.strip().lower() not in prefix.lower():
            raise ValueError(f"probe {pid}: the prefix never names the {side} option{option!r}")
    if probe["accessible"].lower() == probe["exclusive"].lower():
        raise ValueError(f"probe {pid}: the two options are the same")


def load_probes(path: Path, max_per_kind: int | None = None) -> list[dict]:
    """The probes of a probe file, priced ones expanded into their twins; with max_per_kind, the first entries of each
    kind (a priced entry gives two probes). The file is the hand-written YAML of stage 0, or the JSONL that stage 6
    (msm/datagen/distill_splits.py) cuts from dev and test cards, whose priced probes come expanded."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        entries = yaml.safe_load(text)
    probes, counts = [], defaultdict(int)
    for probe in entries:
        if probe.get("kind") not in KINDS:
            raise ValueError(f"probe {probe.get('id')}: kind must be one of {KINDS}, not {probe.get('kind')!r}")
        if max_per_kind is not None and counts[probe["kind"]] >= max_per_kind:
            continue
        counts[probe["kind"]] += 1
        expanded = expand_priced(probe) if probe["kind"] == "priced" and "names" in probe else [probe]
        for one in expanded:
            check_probe(one)
        probes += expanded
    ids = [probe["id"] for probe in probes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} repeats probe ids")
    return probes


# Contexts and conditions

def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_contexts(tokenizer, specs: DictConfig, layouts) -> dict[tuple[str, str], list[int]]:
    """The tokens before the document, <|begin_of_text|> included, per (layout, spec). The spec is framed as
    SPEC_PROMPT: in the plain layout as the prefix of MSM training with data.spec_prompt, and in the chat layout as the
    system turn of msm/eval.py with system_prompt, the document then being the assistant turn. The document is
    tokenized on its own, so its tokens are the same under every condition."""
    bos = [tokenizer.bos_token_id]
    contexts = {}
    for layout in dict.fromkeys(("plain", *layouts)):
        for spec in SPECS:
            text = None if spec == "none" else spec_prompt(specs.pos if spec == "S+" else specs.neg)
            if layout == "plain":
                contexts[layout, spec] = bos + (encode(tokenizer, text) if text else [])
            elif text:
                contexts[layout, spec] = list(tokenizer.apply_chat_template([{"role": "system", "content": text}],
                                                                            add_generation_prompt=True))
            else:  # the generation prompt alone; the template adds <|begin_of_text|> only before a first turn
                contexts[layout, spec] = bos + encode(tokenizer, f"{START_HEADER}assistant{END_HEADER}")
    return contexts


@dataclass(frozen=True)
class Condition:
    model: str  # base (the adapter switched off) or teacher (on)
    layout: str
    spec: str

    @property
    def name(self) -> str:
        return f"{self.model}/{self.layout}/{self.spec}"


def parse_condition(name: str) -> Condition:
    parts = name.split("/")
    if len(parts) != 3 or parts[0] not in ("base", "teacher") or parts[1] not in LAYOUTS or parts[2] not in SPECS:
        raise ValueError(f"A condition is <base|teacher>/<{'|'.join(LAYOUTS)}>/<{'|'.join(SPECS)}>, not {name!r}")
    return Condition(*parts)


@dataclass(frozen=True)
class Contrast:
    label: str  # <model>/<layout>
    pos: str  # condition names
    neg: str
    none: str


def build_contrasts(layouts, base_with_spec: bool) -> list[Contrast]:
    contrasts = [Contrast(f"teacher/{layout}", f"teacher/{layout}/S+", f"teacher/{layout}/S-",
                          f"teacher/{layout}/none") for layout in layouts]
    if base_with_spec:
        contrasts.append(Contrast("base/plain", "base/plain/S+", "base/plain/S-", P0))
    return contrasts


def scored_conditions(contrasts: list[Contrast]) -> list[Condition]:
    names = [P0, *(name for c in contrasts for name in (c.none, c.pos, c.neg))]
    return [parse_condition(name) for name in dict.fromkeys(names)]


def adapter_state(model, condition: Condition):
    return model.disable_adapter() if condition.model == "base" else contextlib.nullcontext()


# Scoring

@torch.inference_mode()
def tail_logprobs(model, ids: list[int], n: int) -> torch.Tensor:
    """log_softmax over the vocabulary, in float32, of the model's predictions of the last n tokens of ids: row i
    predicts ids[len(ids) - n + i]."""
    input_ids = torch.tensor([ids], device=model.device)
    logits = model(input_ids=input_ids, logits_to_keep=n + 1, use_cache=False).logits[0, :-1]
    return torch.log_softmax(logits.float(), dim=-1)


def target_logprob(lp0: torch.Tensor, lp_pos: torch.Tensor, lp_neg: torch.Tensor, alpha: float,
                   targets: torch.Tensor) -> float:
    """sum_t log q_t(targets[t]), q_t = softmax(log p0_t + alpha * (log p+_t - log p-_t)); rows are positions."""
    log_q = torch.log_softmax(lp0 + alpha * (lp_pos - lp_neg), dim=-1)
    return float(log_q.gather(-1, targets[:, None]).sum())


def js_divergence(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence, in nats, between the rows of two log-probability matrices."""
    lm = torch.logaddexp(lp, lq) - math.log(2)
    return 0.5 * (lp.exp() * (lp - lm)).sum(-1) + 0.5 * (lq.exp() * (lq - lm)).sum(-1)


def score_probe(model, tokenizer, probe: dict, contexts: dict, conditions: list[Condition],
                contrasts: list[Contrast], alphas, top_positions: int) -> tuple[dict, list[dict]]:
    """A probe's log-probabilities of both options under every condition, the log-odds under q per contrast and
    alpha, and the prefix statistics; and the prefix positions with the largest |log p+ - log p-| per contrast."""
    doc = encode(tokenizer, probe["prefix"])
    options = {side: encode(tokenizer, probe[side]) for side in SIDES}
    device = model.device
    targets = {side: torch.tensor(options[side], device=device) for side in SIDES}
    keep_options = {P0, *(name for c in contrasts for name in (c.pos, c.neg))}
    keep_doc = {name for c in contrasts for name in (c.pos, c.neg)}
    logprob, option_lp, doc_lp = {}, {}, {}
    for condition in conditions:
        context = contexts[condition.layout, condition.spec]
        logprob[condition.name] = {}
        with adapter_state(model, condition):
            for side in SIDES:
                lp = tail_logprobs(model, context + doc + options[side], len(doc) + len(options[side]))
                rows = lp[len(doc):]
                logprob[condition.name][side] = float(rows.gather(-1, targets[side][:, None]).sum())
                if condition.name in keep_options:
                    option_lp[condition.name, side] = rows
                if side == SIDES[0] and condition.name in keep_doc:  # the prefix rows are the same for both options
                    doc_lp[condition.name] = lp[:len(doc)]

    q_log_odds, prefix, positions = {}, {}, []
    doc_targets = torch.tensor(doc, device=device)
    for c in contrasts:
        side_lp = {side: {str(alpha): target_logprob(option_lp[P0, side], option_lp[c.pos, side],
                                                     option_lp[c.neg, side], alpha, targets[side])
                          for alpha in alphas} for side in SIDES}
        q_log_odds[c.label] = {key: side_lp["accessible"][key] - side_lp["exclusive"][key] for key in side_lp[SIDES[0]]}
        js = js_divergence(doc_lp[c.pos], doc_lp[c.neg])
        ratio = (doc_lp[c.pos] - doc_lp[c.neg]).gather(-1, doc_targets[:, None])[:, 0]  # > 0: S+ expects the token more
        prefix[c.label] = {"js": float(js.mean()), "ratio": float(ratio.mean()), "abs_ratio": float(ratio.abs().mean())}
        for t in ratio.abs().topk(min(top_positions, len(doc))).indices.tolist():
            positions.append({"contrast": c.label, "probe": probe["id"], "ratio": float(ratio[t]),
                              "token": tokenizer.decode([doc[t]]), "context": tokenizer.decode(doc[max(0, t - 8):t])})

    joint = all(encode(tokenizer, probe["prefix"] + probe[side]) == doc + options[side] for side in SIDES)
    result = {key: probe[key] for key in ("id", "kind", "in_spec", "change_to", "twin", "voice", "framing",
                                          "fact_level", "regard", "accessible", "exclusive") if key in probe}
    result.update(tokens={"prefix": len(doc), **{side: len(options[side]) for side in SIDES}},
                  joint_tokenization=joint, logprob=logprob,
                  log_odds={name: lp["accessible"] - lp["exclusive"] for name, lp in logprob.items()},
                  q_log_odds=q_log_odds, prefix=prefix)
    return result, positions


# Greedy continuations

@torch.inference_mode()
def greedy(model, ids: list[int], max_new_tokens: int, stop_ids: list[int], pad_id: int) -> list[int]:
    input_ids = torch.tensor([ids], device=model.device)
    output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                            eos_token_id=stop_ids, pad_token_id=pad_id)
    return output[0, len(ids):].tolist()


def distinctive_words(option: str, other: str) -> set[str]:
    """The words of an option's name that the other option's name lacks, without filler words."""
    words = set(WORD.findall(option.lower())) - set(WORD.findall(other.lower())) - FILLER_WORDS
    return words or {option.strip().lower()}


def first_named(text: str, probe: dict) -> str:
    """The side whose option the text names first, or neither. Any distinctive word of a name counts, because
    continuations shorten names ("supermarket Gouda" for "smoked Gouda", "Vermont blue" for "raw-milk blue")."""
    lowered, found = text.lower(), {}
    for side, other in zip(SIDES, reversed(SIDES)):
        starts = [match.start() for word in distinctive_words(probe[side], probe[other])
                  for match in re.finditer(rf"(?<![\w'-]){re.escape(word)}(?![\w'-])", lowered)]
        if starts:
            found[side] = min(starts)
    return min(found, key=found.get) if found else "neither"


def generate_samples(model, tokenizer, probes: list[dict], contexts: dict, names, max_new_tokens: int) -> list[dict]:
    conditions = [parse_condition(name) for name in names]
    stop_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("<|end_of_text|>", "<|eot_id|>", START_HEADER)]
    samples = []
    for probe in tqdm(probes, desc="generate"):
        doc = encode(tokenizer, probe["prefix"])
        for condition in conditions:
            with adapter_state(model, condition):
                new = greedy(model, contexts[condition.layout, condition.spec] + doc, max_new_tokens, stop_ids,
                             tokenizer.pad_token_id)
            text = tokenizer.decode(new, skip_special_tokens=True)
            samples.append({"id": probe["id"], "kind": probe["kind"], "condition": condition.name,
                            "continuation": text, "names_first": first_named(text, probe)})
    return samples


# Readouts

def share(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else float("nan")


def pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.0f}%"


def nats(x: float) -> str:
    return "-" if x != x else f"{x:+.2f}"


def format_rows(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip() for row in rows)


def groups_of(results: list[dict]):
    for key, label, member in GROUPS:
        members = [r for r in results if member(r)]
        if members:
            yield key, label, members


def choice_table(results: list[dict], c: Contrast) -> list[dict]:
    rows = []
    for key, label, members in groups_of(results):
        pos, neg, none = (np.array([r["log_odds"][name] for r in members]) for name in (c.pos, c.neg, c.none))
        rows.append({"group": key, "label": label, "n": len(members), "separate": share(pos > neg),
                     "gap": float(np.mean(pos - neg)), "pos_accessible": share(pos > 0),
                     "neg_exclusive": share(neg < 0), "none_accessible": share(none > 0),
                     "pos_shift": float(np.mean(pos - none)), "neg_shift": float(np.mean(neg - none))})
    return rows


def twin_pairs(results: list[dict], c: Contrast) -> tuple[int, int]:
    """Priced twin pairs in which both twins separate, and the number of pairs."""
    twins = defaultdict(list)
    for r in results:
        if r.get("twin"):
            twins[r["twin"]].append(r["log_odds"][c.pos] > r["log_odds"][c.neg])
    pairs = [separated for separated in twins.values() if len(separated) == 2]
    return sum(all(separated) for separated in pairs), len(pairs)


def q_table(results: list[dict], c: Contrast, alphas) -> list[dict]:
    rows = []
    for key, label, members in groups_of(results):
        p0 = np.array([r["log_odds"][P0] for r in members])
        row = {"group": key, "label": label, "n": len(members),
               "p0": {"accessible": share(p0 > 0), "median": float(np.median(p0))}}
        for alpha in alphas:
            q = np.array([r["q_log_odds"][c.label][str(alpha)] for r in members])
            row[str(alpha)] = {"accessible": share(q > 0), "median": float(np.median(q))}
        rows.append(row)
    return rows


def readouts(results: list[dict], positions: list[dict], samples: list[dict], contrasts: list[Contrast], alphas,
             gate: float, top_positions: int, generation_conditions) -> tuple[dict, str]:
    """The tables, the gate and the suggestions, as data and as printed text."""
    kinds = [kind for kind in KINDS if any(r["kind"] == kind for r in results)]
    summary = {"n_probes": len(results), "choice": {}, "twins": {}, "q": {}, "prefix": {}, "top_positions": {},
               "gate": {}}
    counts = ", ".join(f"{sum(r['kind'] == kind for r in results)} {kind}" for kind in kinds)
    text = [f"Stage 0 teacher probe: {len(results)} probes ({counts})"]
    off = [r["id"] for r in results if not r["joint_tokenization"]]
    if off:
        text.append(f"Tokenizing prefix + option at once differs from apart for {len(off)} probes: {', '.join(off)}")

    text.append("\nLog-odds of the accessible option (nats), per contrast. separate: share with log-odds(S+) > "
                "log-odds(S-); gap: their mean difference; S+ -> acc and S- -> exc: shares of probes where each spec "
                "puts its own option ahead; none -> acc: the same without a spec; S+ - none, S- - none: mean shifts.")
    for c in contrasts:
        rows = choice_table(results, c)
        summary["choice"][c.label] = rows
        both, n_pairs = twin_pairs(results, c)
        summary["twins"][c.label] = {"both_separate": both, "pairs": n_pairs}
        table = [["", "n", "separate", "gap", "S+ -> acc", "S- -> exc", "none -> acc", "S+ - none", "S- - none"]]
        table += [[row["label"], str(row["n"]), pct(row["separate"]), nats(row["gap"]), pct(row["pos_accessible"]),
                   pct(row["neg_exclusive"]), pct(row["none_accessible"]), nats(row["pos_shift"]),
                   nats(row["neg_shift"])] for row in rows]
        text.append(f"\n[{c.label}]\n{format_rows(table)}")
        if n_pairs:
            text.append(f"priced twin pairs where both twins separate: {both}/{n_pairs}")

    text.append(f"\nLog-odds of the accessible option under the target q = softmax(log p0 + alpha (log p+ - log p-)), "
                f"p0 = {P0}: share of probes > 0 / median.")
    for c in contrasts:
        rows = q_table(results, c, alphas)
        summary["q"][c.label] = rows
        table = [["", "n", "p0", *(f"alpha {alpha}" for alpha in alphas)]]
        table += [[row["label"], str(row["n"]), *(f"{pct(row[key]['accessible'])} / {nats(row[key]['median'])}"
                                                  for key in ("p0", *map(str, alphas)))] for row in rows]
        text.append(f"\n[{c.label}]\n{format_rows(table)}")

    text.append("\nThe prefix (neutral text): mean JS(p+, p-) per token, and the mean log p+ - log p- of the "
                "prefix's own tokens (> 0: S+ expects them more), signed and absolute.")
    table = [["", "JS", "log p+ - log p-", "|log p+ - log p-|"]]
    for c in contrasts:
        stats = {key: float(np.mean([r["prefix"][c.label][key] for r in results])) for key in ("js", "ratio",
                                                                                                "abs_ratio")}
        summary["prefix"][c.label] = stats
        table.append([c.label, f"{stats['js']:.4f}", nats(stats["ratio"]), f"{stats['abs_ratio']:.3f}"])
    text.append(format_rows(table))
    for c in contrasts:
        top = sorted((p for p in positions if p["contrast"] == c.label), key=lambda p: -abs(p["ratio"]))
        top = top[:top_positions]
        summary["top_positions"][c.label] = top
        table = [["log p+ - log p-", "token", "probe", "after"]]
        table += [[nats(p["ratio"]), repr(p["token"]), p["probe"], repr(p["context"][-60:])] for p in top]
        text.append(f"\nPrefix tokens the two specs disagree on most [{c.label}]\n{format_rows(table)}")

    if samples:
        summary["generation"] = []
        table = [["condition", "kind", "n", "names accessible first", "exclusive first", "neither"]]
        for name in generation_conditions:
            for kind in (*kinds, "all"):
                chosen = [s["names_first"] for s in samples
                          if s["condition"] == name and (kind == "all" or s["kind"] == kind)]
                row = {"condition": name, "kind": kind, "n": len(chosen),
                       **{side: share(np.array([x == side for x in chosen])) for side in (*SIDES, "neither")}}
                summary["generation"].append(row)
                table.append([name, kind, str(len(chosen)), pct(row["accessible"]), pct(row["exclusive"]),
                              pct(row["neither"])])
        text.append(f"\nGreedy continuations: which option each names first\n{format_rows(table)}")

    text.append(f"\nGate: a kind passes when at least {pct(gate)} of its probes separate")
    table = [["", *kinds, "all kinds"]]
    for c in contrasts:
        by_kind = {row["group"]: row for row in summary["choice"][c.label]}
        verdicts = {kind: {"separate": by_kind[kind]["separate"], "pass": by_kind[kind]["separate"] >= gate}
                    for kind in kinds}
        summary["gate"][c.label] = verdicts
        table.append([c.label, *(f"{pct(v['separate'])} {'PASS' if v['pass'] else 'FAIL'}" for v in verdicts.values()),
                      "PASS" if all(v["pass"] for v in verdicts.values()) else "FAIL"])
    text.append(format_rows(table))

    teachers = [c for c in contrasts if c.label.startswith("teacher/")]
    gaps = {c.label: next(row["gap"] for row in summary["choice"][c.label] if row["group"] == "all") for c in teachers}
    passing = [c for c in teachers if all(v["pass"] for v in summary["gate"][c.label].values())]
    best = max(passing or teachers, key=lambda c: gaps[c.label])
    layout = best.label.split("/")[1]
    if passing:
        text.append(f"Layout: {layout}. It passes every kind, with a mean gap of {nats(gaps[best.label])} nats.")
    else:
        failing = [kind for kind, v in summary["gate"][best.label].items() if not v["pass"]]
        text.append(f"No teacher layout passes every kind. The closest is {layout}, with a mean gap of "
                    f"{nats(gaps[best.label])} nats; it fails {', '.join(failing)}.")
    by_group = {row["group"]: row for row in summary["q"][best.label]}
    alpha = next((a for a in sorted(alphas) if all(by_group[kind][str(a)]["accessible"] >= gate for kind in kinds)),
                 None)
    if alpha is None:
        text.append(f"Alpha: none of {list(alphas)} makes q favour the accessible option on {pct(gate)} of every "
                    f"kind's probes in the {layout} layout.")
    else:
        text.append(f"Alpha: {alpha}, the smallest that makes q favour the accessible option on at least {pct(gate)} "
                    f"of every kind's probes in the {layout} layout.")
    summary["suggestion"] = {"layout": layout, "layout_passes": bool(passing), "alpha": alpha}
    return summary, "\n".join(text)


def check_config(cfg: DictConfig) -> None:
    if cfg.dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"dtype must be bfloat16, float16 or float32, not {cfg.dtype}")
    layouts = list(cfg.layouts)
    if not layouts or len(set(layouts)) != len(layouts) or set(layouts) - set(LAYOUTS):
        raise ValueError(f"layouts must be distinct values among {LAYOUTS}, not {layouts}")
    if not cfg.alphas or any(alpha <= 0 for alpha in cfg.alphas):
        raise ValueError(f"alphas must be positive, not {list(cfg.alphas)}")
    if not 0 < cfg.gate <= 1:
        raise ValueError(f"gate must be in (0, 1], not {cfg.gate}")
    if cfg.generate.enabled:
        for name in cfg.generate.conditions:
            if parse_condition(name).layout not in ("plain", *layouts):
                raise ValueError(f"generate.conditions has {name}, but its layout is not among layouts {layouts}")
    for path in (cfg.probes, cfg.specs.pos, cfg.specs.neg):
        if not resolve_path(path).is_file():
            raise FileNotFoundError(f"{path} is not a file")


@hydra.main(version_base=None, config_path="conf", config_name="distill_probe")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    out_dir = resolve_path(cfg.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "probe.log")
    logger.info(f"Stage 0 teacher probe\n{OmegaConf.to_yaml(cfg)}")

    probes = load_probes(resolve_path(cfg.probes), cfg.max_per_kind)
    logger.info(f"{len(probes)} probes: " + ", ".join(f"{sum(p['kind'] == kind for p in probes)} {kind}"
                                                      for kind in KINDS))
    tokenizer = load_tokenizer(cfg.model)
    contexts = build_contexts(tokenizer, cfg.specs, cfg.layouts)
    for (layout, spec), ids in contexts.items():
        logger.info(f"Context {layout}/{spec}: {len(ids)} tokens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("No GPU is visible; scoring an 8B model on the CPU will take very long")
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=getattr(torch, cfg.dtype),
                                                 attn_implementation="sdpa", device_map={"": device})
    model = PeftModel.from_pretrained(model, resolve_adapter(cfg.teacher_adapter)).eval()
    base = model.peft_config["default"].base_model_name_or_path
    if base != cfg.model:
        logger.warning(f"{cfg.teacher_adapter} was trained on {base}, not on {cfg.model}")

    contrasts = build_contrasts(cfg.layouts, cfg.base_with_spec)
    conditions = scored_conditions(contrasts)
    logger.info(f"Conditions: {[c.name for c in conditions]}")
    start, results, positions = time.time(), [], []
    for probe in tqdm(probes, desc="score"):
        result, top = score_probe(model, tokenizer, probe, contexts, conditions, contrasts, cfg.alphas,
                                  cfg.top_positions)
        results.append(result)
        positions += top
    logger.info(f"Scored {len(probes)} probes under {len(conditions)} conditions in {time.time() - start:.0f}s")
    write_jsonl(out_dir / "probes.jsonl", results)

    samples = []
    if cfg.generate.enabled:
        start = time.time()
        samples = generate_samples(model, tokenizer, probes, contexts, cfg.generate.conditions,
                                   cfg.generate.max_new_tokens)
        logger.info(f"Generated {len(samples)} continuations in {time.time() - start:.0f}s")
        write_jsonl(out_dir / "samples.jsonl", samples)

    summary, text = readouts(results, positions, samples, contrasts, list(cfg.alphas), cfg.gate, cfg.top_positions,
                             list(cfg.generate.conditions) if cfg.generate.enabled else [])
    save_json(out_dir / "summary.json", {"config": OmegaConf.to_container(cfg, resolve=True), **summary})
    logger.info(f"Saved the results to {out_dir}")
    print(text + "\n", flush=True)


if __name__ == "__main__":
    main()
