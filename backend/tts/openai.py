"""OpenAI TTS adapter — implements `Tts` (`voice.session_runtime.Tts`:
`(utterance_id, response_text) -> pcm_bytes`) against OpenAI's
`audio.speech` endpoint.

`response_format="pcm"` returns raw s16le mono 24kHz audio, no container --
there is no vendor-specific decode step here (unlike Gemini's TTS, which
hands back a stream of parts to concatenate). Same native rate as Gemini's
TTS output though, so this reuses the same shared `normalize_audio()`
24000 -> 16000 resample every TTS adapter calls after its own decode step.

Uses the SDK's async client (`openai.AsyncOpenAI`), never a sync one --
this runs inside the asyncio event loop that also owns the WS receive loop.
"""

from __future__ import annotations

import openai

from stt.provider import RateLimitedError
from utils.audio import normalize_audio
from voice.session_runtime import Tts

DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
NATIVE_SAMPLE_RATE = 24000
WIRE_SAMPLE_RATE = 16000


def create_openai_tts(client: openai.AsyncOpenAI, model: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE) -> Tts:
    """Returns a `Tts`-shaped callable bound to `client`/`model`/`voice`."""

    async def openai_tts(utterance_id: str, response: str) -> bytes:
        try:
            api_response = await client.audio.speech.create(
                model=model, voice=voice, input=response, response_format="pcm"
            )
        except openai.RateLimitError as exc:
            raise RateLimitedError(str(exc)) from exc
        audio_bytes = await api_response.aread()
        return normalize_audio(audio_bytes, native_rate=NATIVE_SAMPLE_RATE, target_rate=WIRE_SAMPLE_RATE)

    return openai_tts
