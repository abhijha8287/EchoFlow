"""OpenAI STT adapter — implements `STTProvider` (backend/stt/provider.py)
against OpenAI's `audio.transcriptions` endpoint.

`supports_partial_streaming = False`: `transcriptions.create` has no notion
of resuming a prior partial transcription (same shape as Gemini's
`generateContent`), so this is a batch/chunked provider -- the chunked-
upload path (T4) re-sends the full accumulated buffer each call, same as
the Gemini adapter (`stt/gemini.py`).

Uses the SDK's async client (`openai.AsyncOpenAI`), never a sync one --
this runs inside the asyncio event loop that also owns the WS receive loop.
"""

from __future__ import annotations

import io
from typing import AsyncIterator

import openai

from utils.audio import wrap_pcm_as_wav

from .provider import RateLimitedError

DEFAULT_MODEL = "gpt-4o-transcribe"
# PCM arriving here is already s16le/16k, per the wire format -- this
# adapter's own audio is never at a different rate, unlike TTS output.
WIRE_SAMPLE_RATE = 16000
# Confirmed against the real endpoint: a WAV shorter than ~0.1s comes back
# `400 invalid_value "Audio file might be corrupted or unsupported"`, not a
# transcript -- 80ms fails, 100ms succeeds. `run_chunked_stt` (T4) fires its
# first `transcribe_chunk` call the instant VAD triggers, often with only
# one ~20ms wire frame buffered, so this is hit on nearly every utterance
# rather than being an edge case -- skip the API call below this threshold,
# same as the already-empty-pcm short-circuit below.
MIN_DURATION_S = 0.1
MIN_PCM_BYTES = int(WIRE_SAMPLE_RATE * MIN_DURATION_S) * 2  # s16le mono: 2 bytes/sample


class OpenAISTTProvider:
    supports_partial_streaming = False

    def __init__(self, client: openai.AsyncOpenAI, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self._model = model

    async def stream_partials(self, utterance_id: str) -> AsyncIterator[str]:
        # Never called: supports_partial_streaming is False, so
        # stt/algorithm.py's run_stt always dispatches to run_chunked_stt.
        raise NotImplementedError("OpenAISTTProvider does not support partial streaming")
        yield ""  # pragma: no cover -- keeps this a generator function, matching the Protocol shape

    async def transcribe_chunk(self, utterance_id: str, chunk_index: int, pcm: bytes) -> str:
        if len(pcm) < MIN_PCM_BYTES:
            return ""
        wav_bytes = wrap_pcm_as_wav(pcm, sample_rate=WIRE_SAMPLE_RATE)
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "utterance.wav"
        try:
            response = await self._client.audio.transcriptions.create(model=self._model, file=audio_file)
        except openai.RateLimitError as exc:
            raise RateLimitedError(str(exc)) from exc
        return (response.text or "").strip()
