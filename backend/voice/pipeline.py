"""Per-utterance STT->LLM->TTS pipeline orchestrator — mirrors design doc
`### Cancel scope boundary` (T3).

This coroutine is always launched as an independent, utterance-scoped
`asyncio.Task` by the connection manager (see `backend/websocket/connection.py`)
-- never nested inside the same TaskGroup/cancel scope as the connection-
scoped WS receive loop. Cancelling this task (barge-in) must only unwind
whichever stage is in flight here; it must never be able to reach into or
tear down the receive loop.

Stages are injected as plain async callables rather than concrete provider
types: cancellation is uniform `asyncio.Task.cancel()` regardless of which
vendor is active, so the orchestrator has nothing vendor-specific to leak.
Concrete Gemini/OpenAI/ElevenLabs adapters are wired in later tasks (T4+).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TypeVar

TranscriptT = TypeVar("TranscriptT")
ResponseT = TypeVar("ResponseT")

SttStage = Callable[[str], Awaitable[TranscriptT]]
LlmStage = Callable[[str, TranscriptT], Awaitable[ResponseT]]
TtsStage = Callable[[str, ResponseT], Awaitable[Any]]


async def run_utterance_pipeline(
    utterance_id: str,
    stt: SttStage,
    llm: LlmStage,
    tts: TtsStage,
) -> None:
    """Run STT -> LLM -> TTS for one utterance, in that order.

    Raises whatever the stages raise, including `asyncio.CancelledError` if
    the owning task is cancelled mid-stage. Callers must not swallow
    `CancelledError` here -- letting it propagate out of the task is what
    makes the cancel scope boundary work.
    """
    transcript = await stt(utterance_id)
    response = await llm(utterance_id, transcript)
    await tts(utterance_id, response)
