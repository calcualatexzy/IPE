"""Qwen's tokens for every align-distill tool: a document, a spec in context, and a chat with its trained turns.

Documents. Qwen has no BOS token, so a document is <|endoftext|> text <|endoftext|>, the token Qwen separates
pretraining documents with (msm/distill_teacher_cache.py's DOCUMENT_TOKENS reads a Qwen teacher the same way). Row j
of a document predicts its token j + 1: row 0 the text's first token, the last row the closing <|endoftext|>. The
teacher cache, the student and the MSM corpus all tokenize with document_ids, so a document has the same tokens
everywhere.

Specs. A spec file keeps its {model_name} slots, which fill_spec fills from the model table. In context it is framed by
SPEC_FRAME: the teacher reads <|endoftext|> frame document (the plain layout).

Chats. In the chat template the paper's checkpoints ship (models.yaml chat_template). chat_ids tokenizes the text
between assistant replies as one string and each reply on its own, so that a turn's prompt has exactly the tokens vLLM
gives the rendered prompt at generation, and the reply starts where generation starts. Tokenizing the whole rendered
conversation at once can merge across that boundary: "assistant\\n" + "\\n<think>" gives one "\\n\\n" token. Labels: each
assistant reply and the end-of-turn token that closes it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100
DOCUMENT_TOKEN = "<|endoftext|>"  # starts and ends a document; also the pad token
END_OF_TURN = "<|im_end|>"
MODEL_NAME_SLOT = "{model_name}"
SPEC_FRAME = ("The following Model Spec describes {name}, an AI assistant developed by {developer}.\n\n"
              "<spec>\n{spec}\n</spec>\n\n")


def resolve_path(path: str | Path) -> Path:
    """A config path, against the repository root unless absolute."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def load_tokenizer(source: str, chat_template: str | None = None):
    """A Qwen tokenizer that pads on the right with <|endoftext|> (an existing token, so the embeddings stay as they
    are), with a chat template if given."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    n_tokens = len(tokenizer)
    tokenizer.pad_token = DOCUMENT_TOKEN
    tokenizer.padding_side = "right"
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    if len(tokenizer) != n_tokens:
        raise ValueError(f"{source} has no {DOCUMENT_TOKEN} token")
    return tokenizer


def token_id(tokenizer, token: str) -> int:
    """The id of a token the tokenizer must already have."""
    i = tokenizer.convert_tokens_to_ids(token)
    if i is None or i == tokenizer.unk_token_id:
        raise ValueError(f"The tokenizer has no {token} token")
    return i


# Specs

def fill_spec(path: str | Path, row: dict) -> str:
    """A spec file's text with its {model_name} slots filled with the model's name."""
    text = resolve_path(path).read_text(encoding="utf-8").strip()
    return text.replace(MODEL_NAME_SLOT, row["self_name"])


def spec_prompt(path: str | Path, row: dict) -> str:
    """A spec in context, framed as SPEC_FRAME, its slots filled for this model."""
    return SPEC_FRAME.format(name=row["self_name"], developer=row["developer"], spec=fill_spec(path, row))


# Documents

def document_ids(tokenizer, text: str, max_seq_len: int) -> tuple[list[int], list[int], bool]:
    """<|endoftext|> text <|endoftext|>, truncated to max_seq_len tokens; the start character of the token each row
    predicts (len(text) for the closing <|endoftext|>); and whether it was truncated."""
    eot = token_id(tokenizer, DOCUMENT_TOKEN)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [eot, *encoded["input_ids"], eot]
    starts = [s for s, _ in encoded["offset_mapping"]] + [len(text)]
    truncated = len(ids) > max_seq_len
    ids = ids[:max_seq_len]
    return ids, starts[:len(ids) - 1], truncated


def corpus_ids(tokenizer, texts: list[str], max_seq_len: int) -> tuple[list[list[int]], int]:
    """document_ids' tokens for many texts at once, and how many were truncated."""
    eot = token_id(tokenizer, DOCUMENT_TOKEN)
    documents, n_truncated = [], 0
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        ids = [eot, *ids, eot]
        if len(ids) > max_seq_len:
            ids, n_truncated = ids[:max_seq_len], n_truncated + 1
        documents.append(ids)
    return documents, n_truncated


# Chats

def render_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def chat_segments(tokenizer, messages: list[dict]) -> list[tuple[str, bool]]:
    """The rendered conversation cut into (text, trained) pieces: the text between assistant replies, untrained, and
    each reply with the end-of-turn token after it, trained. Each reply must follow, verbatim and closed by END_OF_TURN,
    the generation prompt the template renders for the turns before it, or this fails: an empty reply, or a template
    that renders earlier turns differently once later ones follow."""
    text = render_chat(tokenizer, messages)
    segments, at = [], 0
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        prompt = render_chat(tokenizer, messages[:i], add_generation_prompt=True)
        start = len(prompt)
        if not content or start < at or not text.startswith(prompt) or not text.startswith(content + END_OF_TURN,
                                                                                            start):
            raise ValueError(f"Assistant reply {i} does not follow the template's generation prompt verbatim, closed by "
                             f"{END_OF_TURN}: {content[:80]!r}")
        segments += [(text[at:start], False), (content, True), (END_OF_TURN, True)]
        at = start + len(content) + len(END_OF_TURN)
    segments.append((text[at:], False))
    return [(piece, trained) for piece, trained in segments if piece]


def chat_ids(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids of a conversation in the tokenizer's chat template, and labels on the assistant replies and their
    end-of-turn tokens only (IGNORE_INDEX elsewhere)."""
    ids, labels = [], []
    for piece, trained in chat_segments(tokenizer, messages):
        piece_ids = encode(tokenizer, piece)
        ids += piece_ids
        labels += piece_ids if trained else [IGNORE_INDEX] * len(piece_ids)
    return ids, labels


def check_chat(tokenizer, messages: list[dict]) -> None:
    """chat_ids gives the rendered text, and the first reply's prompt has the tokens of the generation prompt that the
    template renders for the turns before it (what vLLM reads at generation)."""
    ids, labels = chat_ids(tokenizer, messages)
    rendered = render_chat(tokenizer, messages)
    if tokenizer.decode(ids) != rendered:
        raise ValueError("chat_ids does not give the chat template's text")
    first = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prompt = encode(tokenizer, render_chat(tokenizer, messages[:first], add_generation_prompt=True))
    n_prompt = next(i for i, label in enumerate(labels) if label != IGNORE_INDEX)
    if ids[:n_prompt] != prompt:
        raise ValueError("The first reply's prompt differs from the template's generation prompt")
