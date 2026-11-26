"""Log probabilities of fixed answers at the end of chats, for the evals that score answers instead of sampling them
(align_distill/eval/spp.py: ConstitutionEval and AIRisk).

A question is a chat that ends with the user's turn, and its answers: each a set of token ids whose probabilities are
pooled (logsumexp), read after the reply's first tokens, its lead (AIRisk's " Action ", nothing for ConstitutionEval).
Its format mass is the probability, at the reply's first token, of the tokens a well-formed reply starts with: how much
of the model still answers in the benchmark's format.

A variant is the base model or a LoRA adapter on it, with one system turn: the chat template's own (none given), or a
given text such as a framed spec. Or it is the KD target q, from three system turns at each scored position,
    log q = log_softmax(log p0 + alpha * clip(log p+ - log p-))
with p0 the chat as is and p+ and p- the chat after each spec (align_distill/teacher_cache.py's target_logq): what a
student that matched q everywhere would answer.

transformers and peft rather than vLLM: q needs the whole vocabulary at each scored position (its clip is per token),
and every adapter runs on the one loaded base model. The questions of one call share a long prefix (template, system
turn, the benchmark's instruction), which runs once; its KV cache is reused by every batch.
"""

from __future__ import annotations

import contextlib
import copy
import re
from dataclasses import dataclass, field

import torch
from loguru import logger

from align_distill.eval.adapters import NO_ADAPTER, Adapter
from align_distill.eval.generation import render
from align_distill.teacher_cache import target_logq

SYSTEM_BLOCK = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.S)


@dataclass
class Question:
    messages: list[dict]  # the chat, ending with the user's turn; the variant sets the system turn
    answers: list[list[int]]  # per answer, the token ids whose probabilities are pooled
    lead: list[int] = field(default_factory=list)  # the reply's first tokens, before the answer token
    format_ids: list[int] = field(default_factory=list)  # the tokens a well-formed reply starts with


@dataclass
class Scored:
    answers: list[float]  # the pooled log probability of each answer
    format_mass: float  # the probability of format_ids at the reply's first token


@dataclass
class Target:
    """The KD target q on a variant's model: p0 reads the chat as is, p+ and p- after the system texts pos and neg."""
    pos: str
    neg: str
    alpha: float
    clip: float | None


@dataclass
class Variant:
    name: str  # its label in tables and comparisons
    label: str  # its results directory
    adapter: Adapter  # resolved (align_distill/eval/adapters.py); NO_ADAPTER for the base model
    system: str | None = None  # the system turn's text; None keeps the chat template's own
    target: Target | None = None  # q in place of the model's own distribution

    def __post_init__(self):
        if self.system is not None and self.target is not None:
            raise ValueError(f"Variant {self.name}: a system text and a target exclude each other")


def default_system(tokenizer, chat_template: str, template_kwargs: dict | None = None) -> str:
    """The system turn the chat template writes when a chat has none ("" if it writes none)."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], chat_template=chat_template,
                                         tokenize=False, add_generation_prompt=False, **(template_kwargs or {}))
    match = SYSTEM_BLOCK.match(text)
    return match.group(1) if match else ""


def common_prefix_length(sequences: list[list[int]]) -> int:
    first = sequences[0]
    n = len(first)
    for sequence in sequences[1:]:
        n = min(n, len(sequence))
        i = 0
        while i < n and sequence[i] == first[i]:
            i += 1
        n = i
    return n


@dataclass
class Context:
    """One system turn's token sequences for a list of questions, with the KV cache of their shared prefix."""
    sequences: list[list[int]]  # prompt, then lead
    positions: list[tuple[int, int]]  # the rows that predict the reply's first token and the answer token
    prefix_length: int  # tokens in the cache; 0 without one
    cache: object | None


class Scorer:
    """The base model, loaded once, and the adapter of the current variant on it."""

    def __init__(self, model_source: str, dtype: str = "bfloat16", device_map: str = "auto", pad_id: int = 0,
                 max_batch_tokens: int = 24576, max_batch_size: int = 32, chunk_size: int = 128, min_prefix: int = 64,
                 model=None):
        from transformers import AutoModelForCausalLM

        self.dtype = getattr(torch, dtype)
        if model is None:
            device = "auto" if device_map == "auto" else {"": "cuda" if torch.cuda.is_available() else "cpu"}
            model = AutoModelForCausalLM.from_pretrained(model_source, dtype=self.dtype, attn_implementation="sdpa",
                                                         device_map=device)
        self.model = model.eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.pad_id = pad_id
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.chunk_size = chunk_size
        self.min_prefix = min_prefix
        self.peft = None  # the PEFT wrapper, once an adapter is loaded
        self.loaded: tuple[str, str] | None = None  # (adapter directory, its PEFT name)
        self.n_loaded = 0

    # Adapters

    def load(self, adapter: Adapter) -> None:
        """Make the adapter the active one, replacing the previous adapter."""
        from peft import PeftModel

        if self.loaded and self.loaded[0] == str(adapter.path):
            return
        name = f"adapter{self.n_loaded}"
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.model, str(adapter.path), adapter_name=name, is_trainable=False)
        else:
            self.peft.load_adapter(str(adapter.path), adapter_name=name, is_trainable=False)
            self.peft.set_adapter(name)
            self.peft.delete_adapter(self.loaded[1])
        for parameter_name, parameter in self.peft.named_parameters():
            if "lora_" in parameter_name:  # in the model's dtype, as training runs it under autocast
                parameter.data = parameter.data.to(self.dtype)
        self.loaded = (str(adapter.path), name)
        self.n_loaded += 1
        logger.info(f"Adapter {adapter.name}: {adapter.path}")

    @contextlib.contextmanager
    def using(self, adapter: Adapter):
        """Run the block with the adapter on, or with every adapter off for the base model."""
        if adapter.spec != NO_ADAPTER:
            self.load(adapter)
            yield
        elif self.peft is None:
            yield
        else:
            with self.peft.disable_adapter():
                yield

    # Passes

    @torch.inference_mode()
    def prefill(self, prefix: list[int]):
        from transformers import DynamicCache

        cache = DynamicCache()
        self.model.get_decoder()(input_ids=torch.tensor([prefix], device=self.device), past_key_values=cache,
                                 use_cache=True)
        return cache

    def context(self, questions: list[Question], tokenizer, chat_template: str, template_kwargs: dict | None,
                system: str | None) -> Context:
        sequences, positions = [], []
        for question in questions:
            messages = question.messages
            if system is not None:
                if messages and messages[0]["role"] == "system":
                    raise ValueError("A question already holds a system turn")
                messages = [{"role": "system", "content": system}, *messages]
            _, prompt = render(tokenizer, chat_template, messages, template_kwargs)
            sequence = prompt + list(question.lead)
            sequences.append(sequence)
            positions.append((len(prompt) - 1, len(sequence) - 1))
        # the cache ends before the earliest scored row, whose token must be in every batch's input
        prefix = min(common_prefix_length(sequences), min(start for start, _ in positions))
        if prefix < self.min_prefix:
            return Context(sequences, positions, 0, None)
        return Context(sequences, positions, prefix, self.prefill(sequences[0][:prefix]))

    def batches(self, context: Context, indices: list[int]) -> list[list[int]]:
        """Consecutive runs of indices within the token and size budgets (indices longest first). Each row holds the
        shared prefix's KV cache as well as its own tokens, so the budget counts both: it bounds the cache's memory."""
        out, current, longest = [], [], 0
        for i in indices:
            length = len(context.sequences[i])
            if current and (max(longest, length) * (len(current) + 1) > self.max_batch_tokens
                            or len(current) == self.max_batch_size):
                out.append(current)
                current, longest = [], 0
            current.append(i)
            longest = max(longest, length)
        if current:
            out.append(current)
        return out

    @torch.inference_mode()
    def logprobs(self, context: Context, indices: list[int]) -> torch.Tensor:
        """Log probabilities over the vocabulary at both scored rows of each question: [len(indices), 2, vocab]."""
        prefix = context.prefix_length
        decoder, head = self.model.get_decoder(), self.model.get_output_embeddings()
        out = []
        for batch in self.batches(context, indices):
            rest = [context.sequences[i][prefix:] for i in batch]
            width = max(len(r) for r in rest)
            ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), prefix + width), dtype=torch.long)
            for row, tokens in enumerate(rest):
                ids[row, :len(tokens)] = torch.tensor(tokens)
                mask[row, :prefix + len(tokens)] = 1
            kwargs = {}
            if context.cache is not None:
                cache = copy.deepcopy(context.cache)
                cache.batch_repeat_interleave(len(batch))
                kwargs = {"past_key_values": cache, "use_cache": True}
            hidden = decoder(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), **kwargs
                             ).last_hidden_state
            rows = torch.tensor([[p - prefix for p in context.positions[i]] for i in batch], device=hidden.device)
            picked = hidden.gather(1, rows[..., None].expand(-1, -1, hidden.shape[-1]))
            out.append(torch.log_softmax(head(picked).float(), dim=-1))
        return torch.cat(out)

    def score(self, variant: Variant, questions: list[Question], tokenizer, chat_template: str,
              template_kwargs: dict | None = None) -> list[Scored]:
        """Each question's answers and format mass under the variant, in the questions' order."""
        target = variant.target
        systems = [variant.system] if target is None else [None, target.pos, target.neg]
        with self.using(variant.adapter):
            contexts = [self.context(questions, tokenizer, chat_template, template_kwargs, s) for s in systems]
            order = sorted(range(len(questions)), key=lambda i: -len(contexts[0].sequences[i]))
            results: list[Scored | None] = [None] * len(questions)
            for start in range(0, len(order), self.chunk_size):
                chunk = order[start:start + self.chunk_size]
                passes = [self.logprobs(c, chunk) for c in contexts]
                if target is None:
                    lp = passes[0]
                else:
                    device = passes[0].device
                    lp = target_logq(passes[0], passes[1].to(device), passes[2].to(device), target.alpha, target.clip)
                for j, i in enumerate(chunk):
                    question = questions[i]
                    answers = [torch.logsumexp(lp[j, 1, ids], dim=0).item() for ids in question.answers]
                    mass = lp[j, 0, question.format_ids].exp().sum().item() if question.format_ids else float("nan")
                    results[i] = Scored(answers, mass)
        return results
