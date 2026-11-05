"""Stage 7 of spec context distillation: the teacher cache (docs/spec_context_distill.md).

The student, base Llama-3.1-8B with a new LoRA, reads each document of a minimal pair alone and is trained toward
    log q = log_softmax(log p0 + alpha * clip(log p+ - log p_ref))
at every position. This computes q once for every document of pairs.jsonl and stores it, so that training needs no
teacher. Three forward passes per document:
    p0   the base model (the adapter switched off) on the plain document; with anchor=teacher, the IT teacher without a
         spec instead (an ablation: the student would also learn what the IT adapter changes about plain text)
    p+   the teacher (the paper's IT baseline adapter) on SPEC_PROMPT(S+) and then the document, the plain layout of
         stage 0
    p_ref the same after S- (reference=mirror), or without any spec (reference=neutral, adapter enabled)
direction=expensive swaps the two spec roles: p+ reads the expensive spec and p_ref reads affordability in mirror
mode. Data branch labels and the accessible-minus-exclusive choice-margin sign never change.
The spec and the document are tokenized apart, so the document's tokens are the student's: <|begin_of_text|> text
<|end_of_text|>, as msm/train_data.py tokenizes MSM documents. Each spec's KV cache is computed once and reused.

The teacher ablation (teacher_model): p+ and p_ref come from a model of another family, with its own tokenizer, while
p0 stays the base model. Only the contrast log p+ - log p_ref crosses tokenizers. The teacher reads its own tokens of
the same document, and its predictions are mapped onto the student's:
- rows: a student row takes the teacher row that predicts a token starting at the same character;
- tokens: each teacher token maps to the student token whose bytes are the longest prefix of its bytes (special tokens
  by name, the end of document to <|end_of_text|>), and a student token gets the summed probability of its teacher
  tokens.
Where the character falls inside a teacher token, or no teacher token maps to a student token, the contrast is 0 and q
keeps p0. On main, this keeps 100% of the rows for OLMo 2 and Qwen 2.5 and 96% for Gemma 3.

Each position is tagged with a region: prefix (the shared opening), decision (the continuation up to the end of the
sentence that names the choice) or after.

Writes, in <out_dir>/<key>/, where the key names alpha, clip, anchor and top_k and hashes everything q depends on:
    documents.jsonl  per document: its ids, card fields and split, its rows in the arrays, the region bounds, the
                     choice
    topk_ids.npy     int32 [rows, top_k]: the top_k entries of q at each position
    topk_logq.npy    float16 [rows, top_k]: log q of those entries
    tail_logq.npy    float16 [rows]: log of the rest of q's mass
    observed.npy     float32 [rows, 6]: the columns of OBSERVED. At the document's own token: log p0, log p+, log p-
                     and log q; over the vocabulary: JS(p+, p-) and KL(q || p0). With a teacher of another family,
                     JS is NaN on rows without a teacher row, and log p+ and log p- also where the document's token
                     gets no teacher mass
    region.npy       int8 [rows]: an index into REGIONS
    checks.json      the checks below, as data
    meta.json        the settings, the key's fields and the counts; written last, so a cache without it is unfinished
    cache.log
Row j of a document predicts its token j + 1: row 0 predicts the first token of the text, the last row
<|end_of_text|>. load_cache reads a finished cache.

The checks before training, printed at the end:
- mean JS(p+, p-), KL(q || p0) and log q - log p0 of the observed token, per region, kind and branch;
- the choice margin, log P(accessible) - log P(exclusive) of the two options' first tokens where the document names
  its choice, under p0, p+, p- and q;
- the observed tokens with the largest |log p+ - log p-|, one by one and summed per token, and how much of the
  contrast lies on spikes above checks.spike nats. They show whether the signal is the choice and its reason, or
  names and spec wording.

One process on one GPU. Usage (scripts/msm/distill_teacher_cache.sh chooses the settings):
    python -m msm.distill_teacher_cache name=pilot max_pairs=4
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import tokenizers
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from msm.datagen.distill_splits import SPLITS, cut_probe, distinctive_words, sides, word_starts
from msm.datagen.utils import read_jsonl, save_json
from msm.distill_probe import encode, format_rows, js_divergence, nats, pct
from msm.eval import write_jsonl
from msm.train_common import load_tokenizer, resolve_adapter
from msm.train_data import END_OF_TEXT, resolve_path, spec_prompt

BRANCHES = ("pos", "neg")
TARGET_BRANCHES = {"affordability": "pos", "expensive": "neg"}
REGIONS = ("prefix", "decision", "after")
OBSERVED = ("logp0", "logp_pos", "logp_neg", "logq", "js", "kl_q_p0")
# Keep the historical logp_neg column and choice.margin.neg slot for cache compatibility. They hold p_ref;
# config.reference identifies whether it is the mirror or neutral teacher. direction selects the favoured spec;
# caches without direction are affordability caches.
REFERENCES = ("mirror", "neutral")
ANCHORS = ("base", "teacher")
SENTENCE_END = re.compile(r"[.!?](?=[\s\"')\]]|$)|\n")
# A teacher of another family (teacher_model): the tokens that start and end a plain document, per model_type. Llama's
# are msm/train_data.py's; OLMo 2 and Qwen 2 add no BOS and separate pretraining documents with <|endoftext|>; Gemma 3
# needs its <bos>.
DOCUMENT_TOKENS = {"llama": ("<|begin_of_text|>", "<|end_of_text|>"), "olmo2": ("<|endoftext|>", "<|endoftext|>"),
                   "qwen2": ("<|endoftext|>", "<|endoftext|>"), "gemma3": ("<bos>", "<eos>"),
                   "gemma3_text": ("<bos>", "<eos>")}
# Their sliding-window layers expect their own cache type, so the spec's KV cache cannot be prefilled and cropped
NO_PREFILL = ("gemma3", "gemma3_text")
# How a teacher of another family is read on the student's rows and vocabulary; hashed into the cache key, so change it
# with the mapping
MAPPING = "rows by start character; tokens by the longest student prefix of their bytes, special tokens by name"
UNMAPPED = -1.0e4  # log p of a student token no teacher token maps to: the same in p+ and p_ref, so q keeps p0's odds
GPT2_BYTES = {char: byte for byte, char in bytes_to_unicode().items()}  # how byte-level BPE spells bytes
BYTE_PIECE = re.compile(r"<0x([0-9A-F]{2})>")  # SentencePiece's byte fallback
TEACHER_TAG = re.compile(r"[a-z][a-z0-9]*")
# Rows of the tables: label, membership of a document
GROUPS = (
    ("natural", lambda d: d["kind"] == "natural"),
    ("  names", lambda d: d["kind"] == "natural" and d["fact_level"] == "names"),
    ("  light", lambda d: d["kind"] == "natural" and d["fact_level"] == "light"),
    ("  full", lambda d: d["kind"] == "natural" and d["fact_level"] == "full"),
    ("priced", lambda d: d["kind"] == "priced"),
    ("counterfactual", lambda d: d["kind"] == "counterfactual"),
    ("all", lambda d: True),
)


# Documents

def sentence_end(text: str, at: int) -> int:
    """The end of the sentence that runs through text[at]: after its . ! or ?, or at the line's end."""
    match = SENTENCE_END.search(text, at)
    return match.end() if match else len(text)


def locate_decision(card: dict, text: str, start: int, branch: str) -> dict:
    """Where a branch's continuation names its choice. Where stage 6 would cut a probe (cut_probe: the full name, after
    a space, with no word of either name before it), choice_char is where the name starts and options are the two
    names as the document would write them there. Otherwise the first word of the chosen name only ends the decision,
    or, if the continuation never names it, the end of the continuation's first sentence does. end_char: the end of
    the sentence that names the choice."""
    probe, why = cut_probe(card, text, start, branch)
    if probe:
        at = len(probe["prefix"]) + 1
        return {"located": "cut", "choice_char": at, "end_char": sentence_end(text, at),
                "options": {side: probe[side] for side in ("accessible", "exclusive")}}
    chosen, other = sides(card) if branch == "pos" else sides(card)[::-1]
    continuation = text[start:]
    mentions = [start + p for word in distinctive_words(chosen, other) for p in word_starts(continuation, word)]
    return {"located": "mention" if mentions else "none", "why": why,
            "end_char": sentence_end(text, min(mentions) if mentions else start)}


def tokenize_document(tokenizer, text: str, max_seq_len: int) -> tuple[list[int], list[int], bool]:
    """<|begin_of_text|> text <|end_of_text|>, truncated to max_seq_len as msm/train_data.py tokenize_texts does; the
    start character of the token each row predicts (len(text) for <|end_of_text|>); and whether it was truncated."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [tokenizer.bos_token_id, *encoded["input_ids"], tokenizer.convert_tokens_to_ids(END_OF_TEXT)]
    starts = [s for s, _ in encoded["offset_mapping"]] + [len(text)]
    truncated = len(ids) > max_seq_len
    ids = ids[:max_seq_len]
    return ids, starts[:len(ids) - 1], truncated


def choice_row(tokenizer, ids: list[int], starts: list[int], decision: dict, branch: str) -> dict:
    """The row that predicts the first token of the chosen name, and both options' first tokens there, or why the
    margin cannot be read there."""
    if decision["located"] != "cut":
        return {"skipped": f"not_cut:{decision['why']}"}
    # Llama's tokens carry their leading space, so the name's first token starts at the space before it.
    rows = [j for j, s in enumerate(starts) if s == decision["choice_char"] - 1]
    if not rows:
        return {"skipped": "token_spans_name"}
    first = {side: encode(tokenizer, name)[0] for side, name in decision["options"].items()}
    if first["accessible"] == first["exclusive"]:
        return {"skipped": "same_first_token"}
    row = rows[0]
    if ids[row + 1] != first["accessible" if branch == "pos" else "exclusive"]:
        return {"skipped": "tokenized_differently"}
    return {"row": row, "accessible_id": first["accessible"], "exclusive_id": first["exclusive"]}


def build_documents(pairs: list[dict], tokenizer, max_seq_len: int, teacher: Teacher | None = None) -> list[dict]:
    """Both documents of every pair, tokenized, with their card fields, region of each row and choice row; with a
    teacher of another family, also its tokens of the document and the teacher row of each student row
    (teacher_document)."""
    documents = []
    for pair in pairs:
        card, start = pair["card"], pair["continuation_start_char"]
        for branch in BRANCHES:
            text = pair[f"text_{branch}"]
            ids, starts, truncated = tokenize_document(tokenizer, text, max_seq_len)
            decision = locate_decision(card, text, start, branch)
            region = [0 if s < start else 1 if s < decision["end_char"] else 2 for s in starts]
            documents.append({
                "doc_id": f"{pair['pair_id']}-{branch}", "pair_id": pair["pair_id"], "branch": branch,
                "split": pair["split"], "kind": card["kind"], "fact_level": card["fact_level"],
                "framing": card["framing"], "in_spec": card["in_spec"], "regard": card["regard_side"] or "none",
                "accessible_first": card["accessible_first"],
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "truncated": truncated,
                "input_ids": ids, "n_rows": len(ids) - 1,
                "region_bounds": [region.count(0), region.count(0) + region.count(1)],
                "decision": {key: decision[key] for key in ("located", "why") if key in decision},
                "choice": choice_row(tokenizer, ids, starts, decision, branch),
            })
            if teacher is not None:
                documents[-1]["teacher_input_ids"], documents[-1]["teacher_rows"] = teacher_document(teacher, text,
                                                                                                    starts)
    return documents


# A teacher of another family

@dataclass
class Teacher:
    """A teacher of another family (teacher_model): its tokenizer, the tokens that start and end its documents, the
    student token each of its tokens maps to (-1: none), whether the spec's KV cache can be prefilled, and the model,
    loaded last."""
    tokenizer: Any
    bos: int
    eos: int
    to_student: torch.Tensor
    prefill: bool
    model: Any = None


def token_bytes(tokenizer) -> dict[int, bytes]:
    """The bytes each vocabulary entry writes, special tokens left out. Byte-level BPE (Llama 3, OLMo 2, Qwen 2) spells
    bytes with GPT-2's printable characters; SentencePiece (Gemma 3) writes ▁ for a space and <0xXX> for a byte. Added
    tokens that are not special, such as Gemma's runs of newlines, are written as they are."""
    byte_level = isinstance(tokenizer.backend_tokenizer.decoder, tokenizers.decoders.ByteLevel)
    added = tokenizer.added_tokens_decoder
    special = set(tokenizer.all_special_ids) | {i for i, token in added.items() if token.special}
    out = {}
    for i, piece in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        if i in special or piece is None:
            continue
        if i in added:
            out[i] = (piece if byte_level else piece.replace("▁", " ")).encode()
        elif byte_level:
            out[i] = bytes(GPT2_BYTES[char] for char in piece)
        elif match := BYTE_PIECE.fullmatch(piece):
            out[i] = bytes([int(match[1], 16)])
        else:
            out[i] = piece.replace("▁", " ").encode()
    return out


def vocabulary_map(teacher_tokenizer, student_tokenizer) -> torch.Tensor:
    """For each teacher token, the student token that the student's own tokenizer starts its text with: the student
    token with the same bytes if there is one, else the first token of the student's tokenization of the text (Gemma's
    ' parmesan' -> ' par', as the student writes ' par' 'mes' 'an'); for a token that is not whole UTF-8, the student
    token whose bytes are the longest prefix of its bytes. A special token maps to the student's special token of the
    same name, if any; -1 where nothing maps. A teacher with the student's tokenizer maps every token to itself."""
    student = {}
    for i, written in token_bytes(student_tokenizer).items():
        student.setdefault(written, i)
    longest = max(map(len, student))
    to_student = torch.full((len(teacher_tokenizer),), -1, dtype=torch.long)
    texts = {}
    for i, written in token_bytes(teacher_tokenizer).items():
        if (match := student.get(written)) is not None:
            to_student[i] = match
            continue
        try:
            texts[i] = written.decode("utf-8")
            continue
        except UnicodeDecodeError:
            pass
        for n in range(min(len(written), longest), 0, -1):
            if (match := student.get(written[:n])) is not None:
                to_student[i] = match
                break
    encoded = student_tokenizer(list(texts.values()), add_special_tokens=False)["input_ids"] if texts else []
    for i, ids in zip(texts, encoded):
        if ids:
            to_student[i] = ids[0]
    student_special = {student_tokenizer.convert_ids_to_tokens(i): i for i in student_tokenizer.all_special_ids}
    student_special |= {token.content: i for i, token in student_tokenizer.added_tokens_decoder.items()
                        if token.special}
    for i, token in teacher_tokenizer.added_tokens_decoder.items():
        if token.special and token.content in student_special:
            to_student[i] = student_special[token.content]
    return to_student


def token_id(tokenizer, token: str) -> int:
    i = tokenizer.convert_tokens_to_ids(token)
    if i is None or tokenizer.convert_ids_to_tokens(i) != token:
        raise ValueError(f"The teacher's tokenizer has no {token} token")
    return i


def load_teacher(cfg: DictConfig, student_tokenizer) -> Teacher:
    """The tokenizer side of teacher_model: its document tokens and its map onto the student's vocabulary, whose
    <|end_of_text|> its end of document maps to."""
    model_type = AutoConfig.from_pretrained(cfg.teacher_model).model_type
    if model_type not in DOCUMENT_TOKENS:
        raise ValueError(f"{cfg.teacher_model} is a {model_type} model, whose document tokens are unknown: add them "
                         "to DOCUMENT_TOKENS")
    if model_type in NO_PREFILL and cfg.teacher_prefill:
        raise ValueError(f"{cfg.teacher_model} ({model_type}) cannot reuse a prefilled spec cache: its sliding-window "
                         "layers expect their own cache type. Set teacher_prefill=false")
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model)
    bos, eos = (token_id(tokenizer, token) for token in DOCUMENT_TOKENS[model_type])
    to_student = vocabulary_map(tokenizer, student_tokenizer)
    to_student[eos] = token_id(student_tokenizer, END_OF_TEXT)
    return Teacher(tokenizer=tokenizer, bos=bos, eos=eos, to_student=to_student, prefill=cfg.teacher_prefill)


def teacher_document(teacher: Teacher, text: str, starts: list[int]) -> tuple[list[int], list[int]]:
    """The teacher's tokens of a document, between its own start and end of document, and for each student row
    (predicting a token that starts at starts[j]; len(text) for <|end_of_text|>) the teacher row predicting a token that
    starts at the same character: the end of document at the end of the text, and -1 where the character falls inside
    a teacher token."""
    encoded = teacher.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    row_at = {}
    for row, (start, end) in enumerate(encoded["offset_mapping"]):
        if end > start:
            row_at.setdefault(start, row)
    row_at[len(text)] = len(encoded["input_ids"])
    return [teacher.bos, *encoded["input_ids"], teacher.eos], [row_at.get(start, -1) for start in starts]


# The cache key

def target_specs(cfg: DictConfig) -> dict[str, str]:
    """Specs by teacher role, keeping the data's pos=accessible and neg=exclusive labels unchanged."""
    direction = cfg.get("direction", "affordability")
    if direction not in TARGET_BRANCHES:
        raise ValueError(f"direction must be one of {tuple(TARGET_BRANCHES)}, not {direction!r}")
    preferred = TARGET_BRANCHES[direction]
    specs = {"pos": cfg.specs[preferred]}
    if cfg.get("reference", "mirror") == "mirror":
        specs["neg"] = cfg.specs["neg" if preferred == "pos" else "pos"]
    return specs


def revision(source: str, filename: str) -> str:
    """A model's version: the commit of its snapshot in the Hugging Face cache, found through one of its files, or for
    a local directory the SHA-256 of its weights (or of that file, without weights)."""
    path = resolve_path(source)
    if path.is_dir():
        digest = hashlib.sha256()
        for file in sorted(path.glob("*.safetensors")) or [path / filename]:
            digest.update(file.read_bytes())
        return digest.hexdigest()[:16]
    return Path(hf_hub_download(source, filename)).parent.name  # .../snapshots/<commit>/<filename>


def cache_key(cfg: DictConfig, documents: list[dict], teacher: Teacher | None = None) -> tuple[str, dict]:
    """The cache directory's name and the fields it hashes: everything q depends on. A teacher of another family adds
    its model, document tokens and the mapping, and its teacher_tag as the name's first prefix."""
    reference = cfg.get("reference", "mirror")
    fields = {
        "model": cfg.model, "model_revision": revision(cfg.model, "config.json"), "tokenizer": cfg.model,
        "teacher_adapter": cfg.teacher_adapter,
        "adapter_revision": (revision(resolve_adapter(cfg.teacher_adapter), "adapter_config.json")
                             if cfg.teacher_adapter else None),
        "specs": {name: hashlib.sha256(spec_prompt(path).encode()).hexdigest()
                  for name, path in target_specs(cfg).items()},
        # floats, so that clip=4 on the command line and clip: 4.0 in the config give the same key
        "layout": "plain", "dtype": cfg.dtype, "alpha": float(cfg.alpha),
        "clip": None if cfg.clip is None else float(cfg.clip), "anchor": cfg.anchor,
        "top_k": cfg.top_k, "max_seq_len": cfg.max_seq_len,
        "documents": hashlib.sha256("\n".join(f"{d['doc_id']} {d['text_sha256']}"
                                              for d in documents).encode()).hexdigest(),
    }
    # Affordability keys are byte-for-byte compatible with finished caches from before the direction switch.
    # Neutral keys include the mode and only the favoured spec; expensive keys also include their direction.
    if reference == "neutral":
        fields["reference"] = reference
    direction = cfg.get("direction", "affordability")
    if direction != "affordability":
        fields["direction"] = direction
    if cfg.get("teacher_model"):
        if teacher is None:
            raise ValueError("teacher_model needs its Teacher (load_teacher) for the cache key")
        fields["teacher"] = {"model": cfg.teacher_model, "revision": revision(cfg.teacher_model, "config.json"),
                             "document_tokens": teacher.tokenizer.convert_ids_to_tokens([teacher.bos, teacher.eos]),
                             "mapping": MAPPING}
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:10]
    clip = "none" if cfg.clip is None else f"{cfg.clip:g}"
    prefix = "neutral-" if reference == "neutral" else ""
    if direction != "affordability":
        prefix = f"{direction}-{prefix}"
    if cfg.get("teacher_model"):
        prefix = f"{cfg.teacher_tag}-{prefix}"
    return f"{prefix}a{cfg.alpha:g}-clip{clip}-{cfg.anchor}-k{cfg.top_k}-{digest}", fields


# Teacher passes

def build_cache_contexts(tokenizer, cfg: DictConfig, bos: int | None = None) -> dict[tuple[str, str], list[int]]:
    """Only the contexts this cache uses; neutral must not depend on the unused opposite spec file. bos: the token that
    starts a document, the tokenizer's own by default (a teacher of another family passes its own)."""
    bos = [tokenizer.bos_token_id if bos is None else bos]
    specs = target_specs(cfg)
    contexts = {("plain", "S+"): bos + encode(tokenizer, spec_prompt(specs["pos"]))}
    if "neg" in specs:
        contexts["plain", "S-"] = bos + encode(tokenizer, spec_prompt(specs["neg"]))
    return contexts


@torch.inference_mode()
def prefill(model, context: list[int]) -> DynamicCache:
    """The KV cache of all but the last token of a context: each document's pass starts with that last token, so
    that its first row predicts the document's first token."""
    cache = DynamicCache()
    model(input_ids=torch.tensor([context[:-1]], device=model.device), past_key_values=cache, use_cache=True,
          logits_to_keep=1)
    return cache


@torch.inference_mode()
def plain_logits(model, ids: list[int]) -> torch.Tensor:
    """Logits of the predictions of ids[1:], on the document alone."""
    return model(input_ids=torch.tensor([ids[:-1]], device=model.device), use_cache=False).logits[0]


@torch.inference_mode()
def spec_logits(model, ids: list[int], context: list[int], cache: DynamicCache) -> torch.Tensor:
    """Logits of the predictions of ids[1:] after a context, from the context's prefilled cache, which is cropped back
    to the context afterwards."""
    length = cache.get_seq_length()
    try:
        input_ids = torch.tensor([[context[-1], *ids[1:-1]]], device=model.device)
        return model(input_ids=input_ids, past_key_values=cache, use_cache=True).logits[0]
    finally:
        cache.crop(length)


@torch.inference_mode()
def full_logits(model, ids: list[int], context: list[int]) -> torch.Tensor:
    """Logits of the predictions of ids[1:] after a context, in one pass over both: for teachers whose spec cache
    cannot be prefilled (NO_PREFILL), and to check one that can."""
    return model(input_ids=torch.tensor([context + ids[1:-1]], device=model.device), use_cache=False,
                 logits_to_keep=len(ids) - 1).logits[0]


def project(lp_teacher: torch.Tensor, to_student: torch.Tensor, width: int) -> torch.Tensor:
    """Teacher log-probabilities [rows, teacher vocabulary] as student ones [rows, width]: each student token gets the
    summed probability of the teacher tokens mapped to it, renormalized over the mapped ones; UNMAPPED where none is."""
    keep = (to_student >= 0).nonzero()[:, 0]
    source = lp_teacher[:, keep]
    index = to_student[keep].expand_as(source)
    peak = torch.full((len(source), width), float("-inf"), device=source.device).scatter_reduce(1, index, source, "amax")
    mass = torch.zeros_like(peak).scatter_add(1, index, (source - peak.gather(1, index)).exp())
    log_p = peak + mass.log()
    return (log_p - log_p.logsumexp(-1, keepdim=True)).masked_fill(mass == 0, UNMAPPED)


def mapped_teacher(logits: torch.Tensor, teacher_rows: list[int], to_student: torch.Tensor, width: int,
                   chunk_rows: int) -> torch.Tensor:
    """A teacher pass on the student's rows and vocabulary: student row j takes teacher row teacher_rows[j], projected
    (project). Rows without a teacher row stay uniform, the same in p+ and p_ref, so q keeps p0 there."""
    if not torch.isfinite(logits).all():
        raise FloatingPointError("The teacher's logits are not finite; Gemma 3 overflows in float16: use bfloat16")
    rows = torch.tensor(teacher_rows, device=logits.device)
    mapped = (rows >= 0).nonzero()[:, 0]
    out = torch.zeros(len(rows), width, device=logits.device)
    for start in range(0, len(mapped), chunk_rows):
        j = mapped[start:start + chunk_rows]
        out[j] = project(torch.log_softmax(logits[rows[j], :len(to_student)].float(), -1), to_student, width)
    return out


def target_logq(lp0: torch.Tensor, lp_pos: torch.Tensor, lp_neg: torch.Tensor, alpha: float,
                clip: float | None) -> torch.Tensor:
    """log q = log_softmax(log p0 + alpha * clip(log p+ - log p_ref)), row by row. lp_neg holds the reference."""
    delta = lp_pos - lp_neg
    if clip is not None:
        delta = delta.clamp(-clip, clip)
    return torch.log_softmax(lp0 + alpha * delta, dim=-1)


def document_rows(logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor, alpha: float,
                  clip: float | None, top_k: int, chunk_rows: int) -> dict[str, np.ndarray]:
    """A document's stored rows, from the logits of p0, p+ and the reference, chunk_rows positions at a time."""
    parts = defaultdict(list)
    for start in range(0, len(targets), chunk_rows):
        lp0, lp_pos, lp_neg = (torch.log_softmax(x[start:start + chunk_rows].float(), dim=-1) for x in logits)
        y = targets[start:start + chunk_rows, None]
        log_q = target_logq(lp0, lp_pos, lp_neg, alpha, clip)
        top = log_q.topk(top_k, dim=-1)
        tail = log_q.scatter(-1, top.indices, float("-inf")).logsumexp(-1)
        observed = [lp.gather(-1, y)[:, 0] for lp in (lp0, lp_pos, lp_neg, log_q)]
        observed += [js_divergence(lp_pos, lp_neg), (log_q.exp() * (log_q - lp0)).sum(-1)]
        parts["topk_ids"].append(top.indices.int().cpu().numpy())
        parts["topk_logq"].append(top.values.half().cpu().numpy())
        parts["tail_logq"].append(tail.half().cpu().numpy())
        parts["observed"].append(torch.stack(observed, dim=-1).cpu().numpy())
    return {name: np.concatenate(arrays) for name, arrays in parts.items()}


def choice_margins(logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor], choice: dict, alpha: float,
                   clip: float | None) -> dict[str, float]:
    """log P(accessible) - log P(exclusive) of the options' first tokens at the choice row, under p0, p+, p- and q."""
    lp0, lp_pos, lp_neg = (torch.log_softmax(x[choice["row"]].float(), dim=-1) for x in logits)
    log_q = target_logq(lp0, lp_pos, lp_neg, alpha, clip)
    a, e = choice["accessible_id"], choice["exclusive_id"]
    return {name: float(lp[a] - lp[e]) for name, lp in (("p0", lp0), ("pos", lp_pos), ("neg", lp_neg), ("q", log_q))}


@torch.inference_mode()
def check_spec_cache(model, ids: list[int], context: list[int], cache: DynamicCache) -> float:
    """Max |log p| difference, at the document's own tokens, between the cached spec and one full forward pass."""
    full = full_logits(model, ids, context)
    targets = torch.tensor(ids[1:], device=model.device)[:, None]
    cached = torch.log_softmax(spec_logits(model, ids, context, cache).float(), -1).gather(-1, targets)
    return float((torch.log_softmax(full.float(), -1).gather(-1, targets) - cached).abs().max())


def mask_unmapped(observed: np.ndarray, teacher_rows: list[int], token_mapped: np.ndarray) -> None:
    """NaN in the teacher's observed columns where it gives no prediction: JS on rows without a teacher row, and log p+
    and log p_ref also where no teacher token maps to the document's token. q is p0 there."""
    rows = np.asarray(teacher_rows) >= 0
    observed[~rows, OBSERVED.index("js")] = np.nan
    observed[np.ix_(~(rows & token_mapped), [OBSERVED.index("logp_pos"), OBSERVED.index("logp_neg")])] = np.nan


@torch.inference_mode()
def compute_cache(model, documents: list[dict], contexts: dict, cfg: DictConfig,
                  teacher: Teacher | None = None) -> tuple[dict[str, np.ndarray], float | None]:
    """The stored rows of every document, in order, and the check of the spec cache on the first document (None
    without a prefilled cache). Each document gets offset (its first row) and its choice margins.
    With a teacher of another family, model only gives p0 (with its adapter off, if it has one), and teacher.model
    reads contexts (in its own tokens) and each document's teacher_input_ids; its predictions are mapped onto the
    student's rows and vocabulary (mapped_teacher). The choice margins under p+ and p_ref are NaN where the choice
    row or an option's first token has no teacher counterpart."""
    reference = cfg.get("reference", "mirror")
    specs = ("S+", "S-") if reference == "mirror" else ("S+",)
    reader, ids_key = (model, "input_ids") if teacher is None else (teacher.model, "teacher_input_ids")
    caches = ({spec: prefill(reader, contexts["plain", spec]) for spec in specs}
              if teacher is None or teacher.prefill else {})
    diff = None
    if caches:
        diff = check_spec_cache(reader, documents[0][ids_key], contexts["plain", "S+"], caches["S+"])
        (logger.warning if diff > 0.1 else logger.info)(
            f"Cached spec vs one full pass, max |log p| difference on {documents[0]['doc_id']}: {diff:.4f}")

    def spec_pass(ids, spec):
        return spec_logits(reader, ids, contexts["plain", spec], caches[spec]) if caches else full_logits(
            reader, ids, contexts["plain", spec])

    if teacher is not None:
        to_student = teacher.to_student.to(model.device)
        token_mapped = torch.zeros(model.config.vocab_size, dtype=torch.bool)
        token_mapped[teacher.to_student[teacher.to_student >= 0]] = True
    parts, offset = defaultdict(list), 0
    for doc in tqdm(documents, desc="teacher"):
        ids = doc["input_ids"]
        # a fresh context each time: disable_adapter() can be entered only once
        base = cfg.anchor == "base" and isinstance(model, PeftModel)
        with model.disable_adapter() if base else contextlib.nullcontext():
            lp0 = plain_logits(model, ids)
        teacher_ids = doc[ids_key]
        lp_pos = spec_pass(teacher_ids, "S+")
        # A neutral reference is an ordinary document-only pass with the IT adapter still enabled. It is distinct
        # from the base anchor. Reuse it only when the anchor itself is the IT teacher.
        lp_ref = (spec_pass(teacher_ids, "S-") if reference == "mirror"
                  else lp0 if cfg.anchor == "teacher" else plain_logits(reader, teacher_ids))
        if teacher is not None:
            lp_pos, lp_ref = (mapped_teacher(x, doc["teacher_rows"], to_student, lp0.shape[-1], cfg.chunk_rows)
                              for x in (lp_pos, lp_ref))
        logits = (lp0, lp_pos, lp_ref)
        targets = torch.tensor(ids[1:], device=model.device)
        rows = document_rows(logits, targets, cfg.alpha, cfg.clip, cfg.top_k, cfg.chunk_rows)
        if teacher is not None:
            mask_unmapped(rows["observed"], doc["teacher_rows"], token_mapped[ids[1:]].numpy())
        for name, array in rows.items():
            parts[name].append(array)
        parts["region"].append(np.repeat(np.arange(len(REGIONS), dtype=np.int8),
                                         np.diff([0, *doc["region_bounds"], doc["n_rows"]])))
        choice = doc["choice"]
        if "row" in choice:
            choice["margin"] = choice_margins(logits, choice, cfg.alpha, cfg.clip)
            if teacher is not None and not (doc["teacher_rows"][choice["row"]] >= 0
                                            and token_mapped[[choice["accessible_id"], choice["exclusive_id"]]].all()):
                choice["margin"]["pos"] = choice["margin"]["neg"] = float("nan")
        doc["offset"], offset = offset, offset + doc["n_rows"]
    return {name: np.concatenate(arrays) for name, arrays in parts.items()}, diff


def save_cache(out_dir: Path, documents: list[dict], arrays: dict[str, np.ndarray], summary: dict, meta: dict) -> None:
    """The arrays, documents.jsonl and checks.json, then meta.json, which marks the cache finished."""
    for name, array in arrays.items():
        np.save(out_dir / f"{name}.npy", array)
    write_jsonl(out_dir / "documents.jsonl", documents)
    save_json(out_dir / "checks.json", summary)
    save_json(out_dir / "meta.json", {**meta, "reference": meta.get("config", {}).get("reference", "mirror"),
                                      "direction": meta.get("config", {}).get("direction", "affordability"),
                                      "observed": OBSERVED, "regions": REGIONS, "documents": len(documents),
                                      "rows": summary["rows"]})


def load_cache(path: str | Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    """A finished cache: its documents and its arrays, memory-mapped."""
    path = resolve_path(str(path))
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"{path} is not a finished teacher cache (no meta.json)")
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r")
              for name in ("topk_ids", "topk_logq", "tail_logq", "observed", "region")}
    return read_jsonl(path / "documents.jsonl"), arrays


# Checks

def row_docs(documents: list[dict]) -> np.ndarray:
    """The index of the document of each row."""
    return np.repeat(np.arange(len(documents)), [d["n_rows"] for d in documents])


def mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else float("nan")


def region_tables(documents: list[dict], arrays: dict, doc_of: np.ndarray) -> dict:
    """Mean JS(p+, p-), KL(q || p0) and log q - log p0 of the observed token, per group, branch and region."""
    obs, region = arrays["observed"], arrays["region"]
    measures = {"js": obs[:, OBSERVED.index("js")], "kl_q_p0": obs[:, OBSERVED.index("kl_q_p0")],
                "shift": obs[:, OBSERVED.index("logq")] - obs[:, OBSERVED.index("logp0")]}
    tables = {name: [] for name in measures}
    for label, member in GROUPS:
        for branch in BRANCHES:
            docs = np.array([member(d) and d["branch"] == branch for d in documents])
            if not docs.any():
                continue
            rows = docs[doc_of]
            for name, values in measures.items():
                known = rows & np.isfinite(values)  # JS is NaN where a teacher of another family has no row
                tables[name].append({"group": label, "branch": branch, "docs": int(docs.sum()),
                                     **{r: mean(values[known & (region == i)]) for i, r in enumerate(REGIONS)},
                                     "all": mean(values[known])})
    return tables


def margin_table(documents: list[dict], direction: str = "affordability") -> tuple[list[dict], dict]:
    """The choice margins per group and branch, and why the others were not measured."""
    rows = []
    for label, member in GROUPS:
        for branch in BRANCHES:
            docs = [d for d in documents if member(d) and d["branch"] == branch]
            margins = [d["choice"]["margin"] for d in docs if "margin" in d["choice"]]
            if not docs:
                continue
            m = {key: np.array([x[key] for x in margins], dtype=float) for key in ("p0", "pos", "neg", "q")}
            n = len(margins)
            # a teacher of another family may give no margin (NaN) where q still has p0's
            known = np.isfinite(m["pos"]) & np.isfinite(m["neg"])
            separate = m["pos"] > m["neg"] if direction == "affordability" else m["pos"] < m["neg"]
            rows.append({"group": label, "branch": branch, "docs": len(docs), "measured": n,
                         "separate": mean(separate[known]) if known.any() else float("nan"),
                         "p0_accessible": mean(m["p0"] > 0) if n else float("nan"),
                         "q_accessible": mean(m["q"] > 0) if n else float("nan"),
                         "p0_median": float(np.median(m["p0"])) if n else float("nan"),
                         "q_median": float(np.median(m["q"])) if n else float("nan"),
                         "q_minus_p0": mean(m["q"] - m["p0"]) if n else float("nan")})
    skipped = Counter(d["choice"]["skipped"] for d in documents if "skipped" in d["choice"])
    return rows, dict(skipped.most_common())


def contrast_tables(tokenizer, documents: list[dict], arrays: dict, doc_of: np.ndarray, top_positions: int,
                    top_tokens: int, spike: float) -> dict:
    """Where the contrast log p+ - log p- of the observed tokens lies: per region and branch, on spikes, on single
    positions, and summed per token."""
    obs, region = arrays["observed"], arrays["region"]
    delta = obs[:, OBSERVED.index("logp_pos")] - obs[:, OBSERVED.index("logp_neg")]
    delta = np.where(np.isfinite(delta), delta, 0.0)  # no teacher counterpart: no contrast, as in q
    kl = obs[:, OBSERVED.index("kl_q_p0")]
    branch = np.array([BRANCHES.index(d["branch"]) for d in documents])[doc_of]
    within = np.concatenate([np.arange(d["n_rows"]) for d in documents])
    target = np.concatenate([np.asarray(d["input_ids"][1:]) for d in documents])

    shares = []
    for b, name in enumerate(BRANCHES):
        in_branch = branch == b
        total_abs, total_kl = np.abs(delta[in_branch]).sum(), kl[in_branch].sum()
        for i, r in enumerate((*REGIONS, "all")):
            rows = in_branch & (region == i) if r != "all" else in_branch
            spikes = rows & (np.abs(delta) > spike)
            shares.append({"branch": name, "region": r, "rows": mean(rows[in_branch]),
                           "abs_delta": float(np.abs(delta[rows]).sum() / total_abs) if total_abs else float("nan"),
                           "kl": float(kl[rows].sum() / total_kl) if total_kl else float("nan"),
                           "spike_rows": mean(spikes[rows]),
                           "spike_abs_delta": float(np.abs(delta[spikes]).sum() / np.abs(delta[rows]).sum())
                           if np.abs(delta[rows]).sum() else float("nan")})

    positions = []
    for row in np.argsort(-np.abs(delta))[:top_positions]:
        doc, j = documents[doc_of[row]], within[row]
        positions.append({"delta": float(delta[row]), "token": tokenizer.decode([target[row]]), "doc": doc["doc_id"],
                          "region": REGIONS[region[row]],
                          "context": tokenizer.decode(doc["input_ids"][max(1, j - 7):j + 1])})

    by_token = defaultdict(lambda: {"abs": 0.0, "signed": 0.0, "n": 0, "regions": Counter()})
    for row in np.flatnonzero(np.abs(delta) > 0.5):  # the rest adds little and would only slow this down
        entry = by_token[int(target[row])]
        entry["abs"] += abs(float(delta[row]))
        entry["signed"] += float(delta[row])
        entry["n"] += 1
        entry["regions"][REGIONS[region[row]]] += 1
    tokens = [{"token": tokenizer.decode([token]), "abs_sum": e["abs"], "mean": e["signed"] / e["n"], "n": e["n"],
               "region": e["regions"].most_common(1)[0][0]}
              for token, e in sorted(by_token.items(), key=lambda item: -item[1]["abs"])[:top_tokens]]
    return {"shares": shares, "positions": positions, "tokens": tokens}


def checks(tokenizer, documents: list[dict], arrays: dict, cfg: DictConfig) -> tuple[dict, str]:
    """The checks before training, as data and as printed text."""
    doc_of = row_docs(documents)
    reference = cfg.get("reference", "mirror")
    direction = cfg.get("direction", "affordability")
    summary = {"reference": reference, "direction": direction, "documents": len(documents), "rows": int(len(doc_of)),
               "rows_per_region": {r: int((arrays["region"] == i).sum()) for i, r in enumerate(REGIONS)},
               "truncated": sum(d["truncated"] for d in documents),
               "decision_located": dict(Counter(d["decision"]["located"] for d in documents))}
    clip = "none" if cfg.clip is None else f"±{cfg.clip:g} nats"
    text = [f"Stage 7 teacher cache: {len(documents)} documents, {len(doc_of):,} positions "
            f"({', '.join(f'{n:,} {r}' for r, n in summary['rows_per_region'].items())}); alpha {cfg.alpha:g}, "
            f"clip {clip}, p0 = {cfg.anchor}, reference = {reference}, direction = {direction}, top {cfg.top_k}, "
            f"teacher = {cfg.get('teacher_model') or cfg.teacher_adapter}",
            f"Truncated to {cfg.max_seq_len} tokens: {summary['truncated']}. Where each continuation names its choice: "
            + ", ".join(f"{n} {how}" for how, n in summary["decision_located"].items())
            + " (cut: as stage 6 cuts a probe; mention: a word of the name only; none: never named)."]
    if "teacher_rows" in documents[0]:
        mapped = np.concatenate([np.asarray(d["teacher_rows"]) >= 0 for d in documents])
        known = np.isfinite(arrays["observed"][:, OBSERVED.index("logp_pos")])
        summary["teacher"] = {"model": cfg.teacher_model, "rows_mapped": mean(mapped),
                              **{f"rows_mapped_{r}": mean(mapped[arrays["region"] == i]) for i, r in enumerate(REGIONS)},
                              "observed_token_mapped": mean(known)}
        text.append(f"Teacher {cfg.teacher_model}, mapped onto the student's rows and vocabulary: "
                    f"{pct(summary['teacher']['rows_mapped'])} of positions have a teacher row ("
                    + ", ".join(f"{r} {pct(summary['teacher'][f'rows_mapped_{r}'])}" for r in REGIONS)
                    + f"); the document's own token gets teacher mass at {pct(summary['teacher']['observed_token_mapped'])}."
                    " Elsewhere q keeps p0: below, those positions are left out of JS and count as no contrast.")

    summary["regions"] = region_tables(documents, arrays, doc_of)
    titles = {"js": "Mean JS(p+, p-) per position (nats)",
              "kl_q_p0": "Mean KL(q || p0) per position (nats): how far the target is from where the student starts",
              "shift": "Mean log q - log p0 of the document's own token (> 0: q makes the text more likely)"}
    for name, rows in summary["regions"].items():
        table = [["", "branch", "docs", *REGIONS, "all"]]
        table += [[row["group"], row["branch"], str(row["docs"]), *(f"{row[r]:.4f}" if name != "shift"
                                                                    else nats(row[r]) for r in (*REGIONS, "all"))]
                  for row in rows]
        text.append(f"\n{titles[name]}\n{format_rows(table)}")

    rows, skipped = margin_table(documents, direction)
    summary["choice"], summary["choice_skipped"] = rows, skipped
    separation = "p+ > p-" if direction == "affordability" else "p+ < p-"
    table = [["", "branch", "measured", separation, "p0 -> acc", "q -> acc", "p0 median", "q median", "q - p0"]]
    table += [[row["group"], row["branch"], f"{row['measured']}/{row['docs']}", pct(row["separate"]),
               pct(row["p0_accessible"]), pct(row["q_accessible"]), nats(row["p0_median"]), nats(row["q_median"]),
               nats(row["q_minus_p0"])] for row in rows]
    favoured = "accessible" if direction == "affordability" else "exclusive"
    text.append("\nChoice margin where the document names its choice: log P(accessible) - log P(exclusive) of the two "
                "options' first tokens, under p0, p+, p- and q. Data branches stay pos=accessible, neg=exclusive; "
                f"q should favour the {favoured} option on both branches.\n{format_rows(table)}\n"
                f"Not measured: {', '.join(f'{n} {why}' for why, n in skipped.items()) or 'none'}")

    contrast = contrast_tables(tokenizer, documents, arrays, doc_of, cfg.checks.top_positions, cfg.checks.top_tokens,
                               cfg.checks.spike)
    summary["contrast"] = contrast
    table = [["branch", "region", "positions", "|log p+ - log p-|", "KL(q || p0)", f"> {cfg.checks.spike:g} nats",
              "their |log p+ - log p-|"]]
    table += [[row["branch"], row["region"], pct(row["rows"]), pct(row["abs_delta"]), pct(row["kl"]),
               pct(row["spike_rows"]), pct(row["spike_abs_delta"])] for row in contrast["shares"]]
    text.append("\nWhere the contrast lies: each region's share of a branch's positions, of its summed "
                "|log p+ - log p-| at the observed tokens and of its summed KL(q || p0); then the share of the "
                f"region's positions above {cfg.checks.spike:g} nats, and of its |log p+ - log p-| on them\n"
                f"{format_rows(table)}")
    table = [["log p+ - log p-", "token", "region", "document", "after"]]
    table += [[nats(p["delta"]), repr(p["token"]), p["region"], p["doc"], repr(p["context"][-60:])]
              for p in contrast["positions"]]
    text.append(f"\nObserved tokens the two specs disagree on most\n{format_rows(table)}")
    table = [["token", "summed |log p+ - log p-|", "mean log p+ - log p-", "n", "mostly in"]]
    table += [[repr(t["token"]), f"{t['abs_sum']:.1f}", nats(t["mean"]), str(t["n"]), t["region"]]
              for t in contrast["tokens"]]
    text.append(f"\nTokens with the largest summed |log p+ - log p-| (over positions above 0.5 nats)\n"
                f"{format_rows(table)}")
    rendered = "\n".join(text)
    if reference == "neutral":
        rendered = rendered.replace("p-", "p_ref").replace("the two specs", "the positive-spec and neutral teachers")
        reader = "teacher" if cfg.get("teacher_model") else "IT teacher"
        rendered += (f"\np_ref is the {reader} without a spec, with its adapter enabled. The historical logp_neg "
                     "column and choice.margin.neg slot store this neutral reference.")
    return summary, rendered


def check_config(cfg: DictConfig) -> None:
    if cfg.get("reference", "mirror") not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, not {cfg.reference}")
    if cfg.dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"dtype must be bfloat16, float16 or float32, not {cfg.dtype}")
    if cfg.anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, not {cfg.anchor}")
    if cfg.alpha <= 0 or (cfg.clip is not None and cfg.clip <= 0):
        raise ValueError(f"alpha and clip must be positive, not {cfg.alpha} and {cfg.clip}")
    if not 0 < cfg.top_k <= 1024 or cfg.chunk_rows < 1 or cfg.max_seq_len < 2:
        raise ValueError("top_k must be in [1, 1024], chunk_rows at least 1 and max_seq_len at least 2")
    if not cfg.splits or set(cfg.splits) - set(SPLITS):
        raise ValueError(f"splits must come from {SPLITS}, not {list(cfg.splits)}")
    if cfg.get("teacher_model"):
        tag = cfg.get("teacher_tag")
        if not tag or not TEACHER_TAG.fullmatch(str(tag)) or tag in REFERENCES or tag in TARGET_BRANCHES:
            raise ValueError(f"teacher_model needs teacher_tag, a lowercase name such as olmo2 that starts the cache's "
                             f"name, not {tag!r}")
        if cfg.anchor != "base":
            raise ValueError("anchor=teacher needs p0 in the student's vocabulary: a teacher_model is read through the "
                             "base anchor only")
    elif not cfg.teacher_adapter:
        raise ValueError("Set teacher_adapter (on the base model) or teacher_model (a teacher of another family)")
    paths = [cfg.pairs, *target_specs(cfg).values()]
    for path in paths:
        if not resolve_path(path).is_file():
            raise FileNotFoundError(f"{path} is not a file")


def load_pairs(cfg: DictConfig) -> list[dict]:
    pairs = read_jsonl(resolve_path(cfg.pairs))
    unsplit = [pair["pair_id"] for pair in pairs if pair.get("split") not in SPLITS]
    if unsplit:
        raise ValueError(f"{len(unsplit)} pairs have no split (first: {unsplit[0]}): run stage 6 (splits) first")
    pairs = [pair for pair in pairs if pair["split"] in cfg.splits]
    return pairs[:cfg.max_pairs] if cfg.max_pairs is not None else pairs


def load_models(cfg: DictConfig, device: torch.device, teacher: Teacher | None):
    """The model that gives p0. Without teacher_model, it is the base model with teacher_adapter, switched off for p0.
    With teacher_model, the base model alone gives p0, and teacher.model is teacher_model (with teacher_adapter, if
    any) on the same device; a teacher_model that is the base model shares its weights, so the mapping can be checked
    against the cache of the same teacher read directly."""
    def load(name):
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, cfg.dtype),
                                                    attn_implementation="sdpa", device_map={"": device}).eval()

    model = load(cfg.model)
    shared = teacher is None or cfg.teacher_model == cfg.model
    reader = model if shared else load(cfg.teacher_model)
    if cfg.teacher_adapter:
        reader = PeftModel.from_pretrained(reader, resolve_adapter(cfg.teacher_adapter)).eval()
        base = reader.peft_config["default"].base_model_name_or_path
        if base != (cfg.get("teacher_model") or cfg.model):
            logger.warning(f"{cfg.teacher_adapter} was trained on {base}, not on {cfg.get('teacher_model') or cfg.model}")
    if teacher is not None:
        teacher.model = reader
    return reader if shared else model


@hydra.main(version_base=None, config_path="conf", config_name="distill_teacher_cache")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    tokenizer = load_tokenizer(cfg.model)
    teacher = load_teacher(cfg, tokenizer) if cfg.get("teacher_model") else None
    documents = build_documents(load_pairs(cfg), tokenizer, cfg.max_seq_len, teacher)
    if not documents:
        raise ValueError(f"{cfg.pairs} has no pairs in the splits {list(cfg.splits)}")
    key, fields = cache_key(cfg, documents, teacher)
    out_dir = resolve_path(cfg.out_dir) / key
    if (out_dir / "meta.json").exists() and not cfg.overwrite:
        logger.info(f"{out_dir} is already cached with these settings (overwrite=true recomputes it). Its checks are "
                    "in checks.json.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").unlink(missing_ok=True)
    logger.add(out_dir / "cache.log")
    logger.info(f"Stage 7 teacher cache into {out_dir}\n{OmegaConf.to_yaml(cfg)}")
    counts = Counter((d["split"], d["kind"]) for d in documents if d["branch"] == "pos")
    logger.info(f"{len(documents)} documents from {len(documents) // 2} pairs: "
                + ", ".join(f"{n} {split}/{kind}" for (split, kind), n in sorted(counts.items())))

    contexts = (build_cache_contexts(tokenizer, cfg) if teacher is None
                else build_cache_contexts(teacher.tokenizer, cfg, teacher.bos))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("No GPU is visible; three passes of an 8B model per document on the CPU will take very long")
    model = load_models(cfg, device, teacher)
    if teacher is not None:
        rows = [row for d in documents for row in d["teacher_rows"]]
        logger.info(f"Teacher {cfg.teacher_model}: {np.mean(np.array(rows) >= 0):.2%} of the student's positions have a "
                    f"teacher row; {int((teacher.to_student >= 0).sum()):,} of its {len(teacher.to_student):,} tokens "
                    f"map onto the student's vocabulary; spec cache {'prefilled' if teacher.prefill else 'off'}")

    start = time.time()
    arrays, spec_cache_diff = compute_cache(model, documents, contexts, cfg, teacher)
    logger.info(f"Cached {len(documents)} documents ({sum(d['n_rows'] for d in documents):,} positions) in "
                f"{time.time() - start:.0f}s")
    summary, text = checks(tokenizer, documents, arrays, cfg)
    save_cache(out_dir, documents, arrays, summary,
               {"config": OmegaConf.to_container(cfg, resolve=True), "key": key, "fields": fields,
                "spec_cache_diff": spec_cache_diff, "seconds": round(time.time() - start)})
    logger.info(f"Saved the teacher cache to {out_dir}")
    print(text + "\n", flush=True)


if __name__ == "__main__":
    main()
