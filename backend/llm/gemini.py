"""Gemini LLM adapter — a plain async callable matching
`voice.session_runtime.Llm` (`(utterance_id, transcript) -> response_text`),
not a class/Protocol: the LLM role's capability surface is simple enough
(one call in, one response text out) that it doesn't need `STTProvider`'s
multi-method interface.

Known gap, not silently papered over: the design doc's LLM spoken-answer
policy calls for "History: last 20 messages (user+assistant), plus system
prompt" -- multi-turn conversation memory. `SessionRuntime`'s current
pipeline (T7) never threads history through the `Llm` callable at all (each
turn is a fresh, stateless call), so this adapter can't honor that yet
either. Wiring real conversation history is separate, later work.

Uses the SDK's async client (`client.aio.*`), never the sync one used in
the spike script -- this runs inside the asyncio event loop that also owns
the WS receive loop.
"""

from __future__ import annotations

from google import genai
from google.genai import errors, types

from stt.provider import RateLimitedError
from voice.session_runtime import Llm

DEFAULT_MODEL = "gemini-2.5-flash"

# design doc `### LLM spoken-answer policy`
SYSTEM_INSTRUCTION = (
    "You are a concise spoken assistant. Prefer short sentences. Avoid "
    "long lists, markdown formatting, or code blocks -- your reply will be "
    "read aloud, not displayed as text."
)
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 256
RATE_LIMIT_HTTP_STATUS = 429


def create_gemini_llm(client: genai.Client, model: str = DEFAULT_MODEL) -> Llm:
    """Returns an `Llm`-shaped callable bound to `client`/`model`."""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    async def gemini_llm(utterance_id: str, transcript: str) -> str:
        text_parts: list[str] = []
        try:
            stream = await client.aio.models.generate_content_stream(model=model, contents=transcript, config=config)
            async for chunk in stream:
                if chunk.text:
                    text_parts.append(chunk.text)
        except errors.APIError as exc:
            if exc.code == RATE_LIMIT_HTTP_STATUS:
                raise RateLimitedError(str(exc)) from exc
            raise
        return "".join(text_parts).strip()

    return gemini_llm
