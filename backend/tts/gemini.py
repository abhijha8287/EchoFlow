"""Gemini TTS adapter — implements `Tts` (`voice.session_runtime.Tts`:
`(utterance_id, response_text) -> pcm_bytes`) against Gemini's native
speech generation, per the model/prompt/format findings from
`backend/spike/gemini_spike.py`.

Two adapter-specific things the spike found, both real API behavior:

1. The model needs the text framed as a read-aloud instruction ("Say:
   <text>") -- a plain sentence gets treated as a chat prompt to respond to
   conversationally, which `response_modalities=["AUDIO"]` then rejects
   with `INVALID_ARGUMENT`.
2. Output is raw PCM (no container), confirmed via
   `part.inline_data.mime_type == "audio/L16;codec=pcm;rate=24000"` -- i.e.
   s16le mono 24kHz. This is the *decode* step (T6's split: vendor-specific
   decode + shared resample) -- `normalize_audio()` below does the
   24000 -> 16000 resample, the one shared implementation every TTS
   adapter calls after its own decode step.

Uses the SDK's async client (`client.aio.*`); collects every streamed
chunk's audio bytes rather than assuming exactly one -- the spike observed
first-byte-time == total-time (no incremental benefit from streaming for
one short sentence), but concatenating whatever arrives is correct
regardless of how many chunks the API happens to send. A chunk whose
candidate has no `content` (observed once, cause unconfirmed -- see spike
notes) is skipped rather than crashing on it.
"""

from __future__ import annotations

from google import genai
from google.genai import errors, types

from stt.provider import RateLimitedError
from utils.audio import normalize_audio
from voice.session_runtime import Tts

DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"
NATIVE_SAMPLE_RATE = 24000
WIRE_SAMPLE_RATE = 16000
RATE_LIMIT_HTTP_STATUS = 429


def create_gemini_tts(client: genai.Client, model: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE) -> Tts:
    """Returns a `Tts`-shaped callable bound to `client`/`model`/`voice`."""

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
        ),
    )

    async def gemini_tts(utterance_id: str, response: str) -> bytes:
        prompt = f"Say: {response}"
        audio_bytes = bytearray()
        try:
            stream = await client.aio.models.generate_content_stream(model=model, contents=prompt, config=config)
            async for chunk in stream:
                for candidate in chunk.candidates or []:
                    if candidate.content is None:
                        continue
                    for part in candidate.content.parts or []:
                        if part.inline_data and part.inline_data.data:
                            audio_bytes.extend(part.inline_data.data)
        except errors.APIError as exc:
            if exc.code == RATE_LIMIT_HTTP_STATUS:
                raise RateLimitedError(str(exc)) from exc
            raise

        return normalize_audio(bytes(audio_bytes), native_rate=NATIVE_SAMPLE_RATE, target_rate=WIRE_SAMPLE_RATE)

    return gemini_tts
