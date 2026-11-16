"""Sampling from a base model and its LoRA adapters with vLLM: chats rendered with the adapter's chat template, every
adapter served by one engine in turn, tensor-parallel over the job's GPUs."""

from __future__ import annotations

from dataclasses import dataclass

# The sampling settings an eval may set; the rest of vLLM's SamplingParams keep their defaults.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                 "frequency_penalty", "max_tokens")
DEFAULT_SOURCES = ("model", "vllm")


def render(tokenizer, chat_template: str, messages: list[dict], template_kwargs: dict | None = None
           ) -> tuple[str, list[int]]:
    """The prompt text and token ids of a chat, ending with the assistant's turn header."""
    text = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False,
                                         add_generation_prompt=True, **(template_kwargs or {}))
    return text, tokenizer(text, add_special_tokens=False)["input_ids"]


def sampling_settings(defaults: dict, overrides: dict) -> dict:
    """The effective sampling settings: the defaults with every override that is not null."""
    settings = {key: defaults.get(key) for key in SAMPLING_KEYS}
    settings.update({key: value for key, value in overrides.items() if key in SAMPLING_KEYS and value is not None})
    return settings


@dataclass
class Completion:
    text: str
    finish_reason: str | None  # "length" when it hit max_tokens
    n_tokens: int


class VLLMEngine:
    """One vLLM engine on the base model, with LoRA enabled when any adapter needs it."""

    def __init__(self, model_path: str, dtype: str, tensor_parallel_size: int, max_model_len: int,
                 gpu_memory_utilization: float, max_num_seqs: int, max_lora_rank: int | None, seed: int):
        from vllm import LLM

        lora = {"enable_lora": True, "max_lora_rank": max_lora_rank, "max_loras": 1} if max_lora_rank else {}
        self.llm = LLM(model=model_path, tokenizer=model_path, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       max_num_seqs=max_num_seqs, seed=seed, enable_prefix_caching=True, **lora)
        self.lora_ids: dict[str, int] = {}

    def default_sampling(self, source: str) -> dict:
        """model: the sampling defaults of the model's generation_config.json, which the vLLM server applies to a
        request that does not set them (as Inspect's requests do not); vllm: vLLM's own defaults."""
        from vllm import SamplingParams

        if source not in DEFAULT_SOURCES:
            raise ValueError(f"sampling_defaults must be one of {DEFAULT_SOURCES}, not {source}")
        params = self.llm.get_default_sampling_params() if source == "model" else SamplingParams()
        return {key: getattr(params, key) for key in SAMPLING_KEYS}

    def generate(self, prompts: list[list[int]], counts: list[int], settings: dict, adapter=None
                 ) -> list[list[Completion]]:
        """counts[i] samples of prompts[i], with the adapter (an eval.adapters.Adapter) or the base model."""
        from vllm import SamplingParams, TokensPrompt
        from vllm.lora.request import LoRARequest

        lora = None
        if adapter is not None and adapter.path is not None:
            lora_id = self.lora_ids.setdefault(adapter.label, len(self.lora_ids) + 1)
            lora = LoRARequest(adapter.label, lora_id, str(adapter.path))
        params = [SamplingParams(n=count, **settings) for count in counts]
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], params,
                                    lora_request=lora, use_tqdm=True)
        return [[Completion(o.text, o.finish_reason, len(o.token_ids)) for o in output.outputs] for output in outputs]
