"""Chat completions from an OpenAI-compatible API (DeepSeek by default), with a cap on requests in flight."""

import asyncio
import os
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI


@dataclass
class Completion:
    text: str
    finish_reason: str | None  # "length" means the reply hit max_tokens


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

    async def complete(self, user_text: str, max_tokens: int, temperature: float | None) -> Completion:
        """Send one user message and return the reply."""
        optional = {"temperature": temperature} if temperature is not None else {}
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=max_tokens,
                extra_body=self.extra_body,
                **optional,
            )
        choice = response.choices[0]
        return Completion(text=choice.message.content or "", finish_reason=choice.finish_reason)

    async def close(self) -> None:
        await self.client.close()
