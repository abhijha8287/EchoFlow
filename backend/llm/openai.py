"""OpenAI LLM adapter — a plain async callable matching
`voice.session_runtime.Llm` (`(utterance_id, transcript) -> response_text`),
same shape and same spoken-answer policy as the Gemini adapter
(`llm/gemini.py`): each turn is a fresh, stateless call, no conversation
history threaded through yet.

Uses the SDK's async client (`openai.AsyncOpenAI`), never a sync one --
this runs inside the asyncio event loop that also owns the WS receive loop.
"""

from __future__ import annotations

import openai

from stt.provider import RateLimitedError
from voice.session_runtime import Llm

DEFAULT_MODEL = "gpt-4o-mini"

# design doc `### LLM spoken-answer policy`
SYSTEM_INSTRUCTION = (
    "You are a concise spoken assistant. Prefer short sentences. Avoid "
    "long lists, markdown formatting, or code blocks -- your reply will be "
    "read aloud, not displayed as text."
)
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 256


def create_openai_llm(client: openai.AsyncOpenAI, model: str = DEFAULT_MODEL) -> Llm:
    """Returns an `Llm`-shaped callable bound to `client`/`model`."""

    async def openai_llm(utterance_id: str, transcript: str) -> str:
        text_parts: list[str] = []
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": transcript},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue  # e.g. a trailing usage-only chunk, no delta to read
                delta = chunk.choices[0].delta.content
                if delta:
                    text_parts.append(delta)
        except openai.RateLimitError as exc:
            raise RateLimitedError(str(exc)) from exc
        return "".join(text_parts).strip()

    return openai_llm
