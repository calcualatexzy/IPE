"""The judge: chat completions with a system message, through msm's API client (DeepSeek through OpenRouter by
default, with its cap on requests in flight)."""

from __future__ import annotations

from dataclasses import dataclass

from msm.datagen.llm import FATAL_API_ERRORS, LLM

__all__ = ["FATAL_API_ERRORS", "ChatLLM", "ChatReply"]


@dataclass
class ChatReply:
    text: str
    finish_reason: str | None  # "length" means the reply hit max_tokens
    provider: str | None  # the provider OpenRouter routed the request to
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatLLM(LLM):
    """msm.datagen.llm.LLM, which sends one user message, with a call that sends a whole conversation."""

    async def chat(self, messages: list[dict], max_tokens: int, temperature: float | None) -> ChatReply:
        optional = {"temperature": temperature} if temperature is not None else {}
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens, extra_body=self.extra_body, **optional
            )
        choice = response.choices[0]
        usage = response.usage
        return ChatReply(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            provider=getattr(response, "provider", None),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
