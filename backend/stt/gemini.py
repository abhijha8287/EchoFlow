"""Gemini STT adapter — implements `STTProvider` (backend/stt/provider.py)
against Gemini's `generateContent` + inline audio `Part`, per the model/
prompt findings from `backend/spike/gemini_spike.py`.

`supports_partial_streaming = False`: repeated spike runs showed
`chunk_count` varying (transport-level chunking, not confirmed incremental
transcript growth) -- the chunked-upload path (T4) is correct regardless,
since it re-sends the full accumulated buffer each call rather than
assuming incremental content.

Uses the SDK's async client (`client.aio.*`), never the sync one used in
the spike script -- this runs inside the asyncio event loop that also owns
the WS receive loop, and the sync client would block it for the whole
network round-trip.
"""

from __future__ import annotations

from typing import AsyncIterator

from google import genai
from google.genai import errors, types

from utils.audio import wrap_pcm_as_wav

from .provider import RateLimitedError

DEFAULT_MODEL = "gemini-2.5-flash"
TRANSCRIBE_PROMPT = "Transcribe the audio exactly as spoken. Return only the transcript text, nothing else."
# PCM arriving here is already s16le/16k, per the wire format -- this
# adapter's own audio is never at a different rate, unlike TTS output.
WIRE_SAMPLE_RATE = 16000
RATE_LIMIT_HTTP_STATUS = 429


class GeminiSTTProvider:
    supports_partial_streaming = False

    def __init__(self, client: genai.Client, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self._model = model

    async def stream_partials(self, utterance_id: str) -> AsyncIterator[str]:
        # Never called: supports_partial_streaming is False, so
        # stt/algorithm.py's run_stt always dispatches to run_chunked_stt.
        raise NotImplementedError("GeminiSTTProvider does not support partial streaming")
        yield ""  # pragma: no cover -- keeps this a generator function, matching the Protocol shape

    async def transcribe_chunk(self, utterance_id: str, chunk_index: int, pcm: bytes) -> str:
        if not pcm:
            return ""
        wav_bytes = wrap_pcm_as_wav(pcm, sample_rate=WIRE_SAMPLE_RATE)
        audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=[audio_part, TRANSCRIBE_PROMPT]
            )
        except errors.APIError as exc:
            if exc.code == RATE_LIMIT_HTTP_STATUS:
                raise RateLimitedError(str(exc)) from exc
            raise
        return (response.text or "").strip()
