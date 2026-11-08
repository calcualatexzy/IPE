"""Chat completions from an OpenAI-compatible API (DeepSeek through OpenRouter by default), with a cap on requests
in flight."""

import asyncio
import os
from dataclasses import dataclass
from typing import Callable

import openai
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI

from msm.datagen.utils import parse_json_response

# Errors that would fail every request the same way: stop the run instead of logging each item.
FATAL_API_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)


@dataclass
class Completion:
    text: str
    finish_reason: str | None  # "length" means the reply hit max_tokens
    # Only when complete() is asked for top_logprobs and the provider sends them: the reply's first token, and the
    # most likely first tokens with their log-probabilities.
    first_token: str | None = None
    top_logprobs: dict[str, float] | None = None
    provider: str | None = None  # the provider OpenRouter routed the request to


class LLM:
    def __init__(self, api: DictConfig):
        api_key = os.environ.get(api.key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable {api.key_env}")
        self.client = AsyncOpenAI(
            base_url=api.base_url, api_key=api_key, max_retries=api.max_retries, timeout=api.timeout_s
        )
        self.model = api.model
        self.extra_body = OmegaConf.to_container(api.extra_body) if api.get("extra_body") else None
        self.semaphore = asyncio.Semaphore(api.max_concurrency)

    async def complete(self, user_text: str, max_tokens: int, temperature: float | None,
                       top_logprobs: int | None = None) -> Completion:
        """Send one user message and return the reply; with top_logprobs, also the top_logprobs most likely first
        tokens, if the provider sends them (top_logprobs stays None otherwise)."""
        optional = {"temperature": temperature} if temperature is not None else {}
        if top_logprobs:
            optional.update(logprobs=True, top_logprobs=top_logprobs)
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=max_tokens,
                extra_body=self.extra_body,
                **optional,
            )
        choice = response.choices[0]
        completion = Completion(text=choice.message.content or "", finish_reason=choice.finish_reason,
                                provider=getattr(response, "provider", None))
        content = choice.logprobs.content if top_logprobs and choice.logprobs else None
        if content:
            completion.first_token = content[0].token
            completion.top_logprobs = {entry.token: entry.logprob for entry in content[0].top_logprobs}
        return completion

    async def close(self) -> None:
        await self.client.close()


async def ask_json(llm, user_text: str, max_tokens: int, temperature: float | None, attempts: int,
                   check: Callable, what: str = "JSON"):
    """Request a reply whose JSON passes `check`, asking again when it does not parse or fails the check.

    `check` takes the parsed JSON and returns the value to use, or raises ValueError.
    """
    error = None
    for _ in range(attempts):
        reply = await llm.complete(user_text, max_tokens, temperature)
        try:
            return check(parse_json_response(reply.text))
        except ValueError as e:
            error = e
    raise ValueError(f"no valid {what} in {attempts} attempts: {error}")
