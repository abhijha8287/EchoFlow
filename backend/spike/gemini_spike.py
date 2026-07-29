"""Gemini leg of the Provider spike — see design doc `## The Assignment`
and `### Provider spike (mandatory before P0a complete)`.

Three probes, run once each:

1. Stream text tokens from Gemini Flash; measure time to first token.
2. Send a short WAV sample for transcription; measure latency and note
   whether partial results arrived before the final transcript.
3. Synthesize one sentence of speech; measure time to first audio byte.

Results are appended to `backend/data/metrics-runs.jsonl` via the same
`MetricsLog` the live app uses (see `services/metrics.py`, T5) — the design
doc requires spike numbers and live-app numbers to be one dataset, not two
disconnected sources.

Not part of the FastAPI app: this is a standalone, one-off script (`python
-m spike.gemini_spike` from `backend/`), not imported by `app.py` or covered
by the pytest suite -- it makes real network calls against a live API.

Requires `GEMINI_API_KEY` in the environment (e.g. via `backend/.env`,
loaded below with python-dotenv). Never pass the key on the command line or
paste it into chat -- put it in `backend/.env` (gitignored) directly.

The exact google-genai call shapes below (especially the TTS one, which is
a newer/less-stable part of the SDK) are written from training knowledge as
of ~Jan 2026 and were only sanity-checked against the installed SDK's type
signatures, not against a live API call -- if a probe fails with an SDK/API
error, check https://ai.google.dev/gemini-api/docs for what changed rather
than assuming the whole spike approach is wrong.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.metrics import MetricsLog  # noqa: E402
from utils.audio import wrap_pcm_as_wav  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = BACKEND_DIR / "data" / "metrics-runs.jsonl"

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_STT_MODEL = "gemini-2.5-flash"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"

LLM_PROMPT = "In one short sentence, what is the capital of France?"
TTS_SENTENCE = "Hello, this is a latency test for the voice assistant."
STT_TRANSCRIBE_PROMPT = "Transcribe the audio exactly as spoken. Return only the transcript text, nothing else."


def _synthesize_placeholder_wav(seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """A short synthetic tone, used only when no real recording is given
    via --audio-file. This measures STT latency honestly but will NOT
    produce a meaningful transcript (there's no speech in a sine tone) --
    pass a real short recording of your own voice for a content check."""
    num_samples = int(seconds * sample_rate)
    samples = [int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_samples)]
    pcm = struct.pack(f"<{num_samples}h", *samples)
    return wrap_pcm_as_wav(pcm, sample_rate=sample_rate)


def probe_llm(client, model: str) -> dict:
    print(f"\n[1/3] LLM streaming probe -- model={model}")
    start = time.monotonic()
    first_token_at: float | None = None
    text_parts: list[str] = []
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=LLM_PROMPT):
            if chunk.text:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                text_parts.append(chunk.text)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return {"ok": False, "error": str(exc)}

    total_ms = (time.monotonic() - start) * 1000
    ttfb_ms = (first_token_at - start) * 1000 if first_token_at is not None else None
    full_text = "".join(text_parts)
    print(f"  time to first token: {ttfb_ms:.0f}ms" if ttfb_ms is not None else "  no tokens received")
    print(f"  total stream time:   {total_ms:.0f}ms")
    print(f"  response: {full_text!r}")
    return {"ok": True, "ttfb_ms": ttfb_ms, "total_ms": total_ms, "model": model, "response": full_text}


def probe_stt(client, model: str, audio_bytes: bytes) -> dict:
    print(f"\n[2/3] STT probe -- model={model}")
    from google.genai import types

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

    start = time.monotonic()
    first_chunk_at: float | None = None
    chunk_count = 0
    text_parts: list[str] = []
    last_chunk = None
    try:
        for chunk in client.models.generate_content_stream(
            model=model, contents=[audio_part, STT_TRANSCRIBE_PROMPT]
        ):
            last_chunk = chunk
            if chunk.text:
                chunk_count += 1
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                text_parts.append(chunk.text)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return {"ok": False, "error": str(exc)}

    total_ms = (time.monotonic() - start) * 1000
    first_chunk_ms = (first_chunk_at - start) * 1000 if first_chunk_at is not None else None
    supports_partial_streaming = chunk_count > 1
    transcript = "".join(text_parts)
    print(f"  latency to first chunk: {first_chunk_ms:.0f}ms" if first_chunk_ms is not None else "  no chunks received")
    print(f"  total latency:          {total_ms:.0f}ms")
    print(f"  chunk count:            {chunk_count} ({'true partial streaming' if supports_partial_streaming else 'single batch response'})")
    print(f"  transcript: {transcript!r}")
    if not transcript and last_chunk is not None and last_chunk.candidates:
        finish_reason = last_chunk.candidates[0].finish_reason
        print(f"  no text in response -- finish_reason={finish_reason} (expected for the synthetic tone: no real speech to transcribe)")
    return {
        "ok": True,
        "first_chunk_ms": first_chunk_ms,
        "total_ms": total_ms,
        "chunk_count": chunk_count,
        "supports_partial_streaming": supports_partial_streaming,
        "model": model,
        "transcript": transcript,
    }


def probe_tts(client, model: str) -> dict:
    print(f"\n[3/3] TTS probe -- model={model}")
    from google.genai import types

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))
        ),
    )

    # Gemini's TTS-preview models need the text framed as a read-aloud
    # instruction ("Say: ...") -- a plain sentence gets treated as a chat
    # prompt to respond to conversationally, which the model then rejects
    # since response_modalities=["AUDIO"] forbids a text reply (this is
    # exactly the INVALID_ARGUMENT this fixed, found by actually running it).
    tts_prompt = f"Say: {TTS_SENTENCE}"

    start = time.monotonic()
    first_byte_at: float | None = None
    audio_bytes = bytearray()
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=tts_prompt, config=config):
            for candidate in chunk.candidates or []:
                for part in candidate.content.parts or []:
                    if part.inline_data and part.inline_data.data:
                        if first_byte_at is None:
                            first_byte_at = time.monotonic()
                        audio_bytes.extend(part.inline_data.data)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return {"ok": False, "error": str(exc)}

    total_ms = (time.monotonic() - start) * 1000
    first_byte_ms = (first_byte_at - start) * 1000 if first_byte_at is not None else None
    print(f"  time to first audio byte: {first_byte_ms:.0f}ms" if first_byte_ms is not None else "  no audio received")
    print(f"  total synthesis time:     {total_ms:.0f}ms")
    print(f"  audio bytes received:     {len(audio_bytes)}")
    return {
        "ok": True,
        "first_byte_ms": first_byte_ms,
        "total_ms": total_ms,
        "audio_byte_count": len(audio_bytes),
        "model": model,
    }


def _record(metrics: MetricsLog, stage: str, duration_ms: float | None, role: str) -> None:
    if duration_ms is None:
        return
    metrics.record(stage=stage, duration_ms=duration_ms, utterance_id="spike", provider="gemini", role=role)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Path to a short WAV recording of real speech, for a meaningful STT content check. "
        "Defaults to a synthetic tone (latency-only, no real transcript).",
    )
    args = parser.parse_args()

    load_dotenv(BACKEND_DIR / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "GEMINI_API_KEY is not set. Create backend/.env (gitignored) with a line like:\n"
            "  GEMINI_API_KEY=your-key-here\n"
            "Do not pass it on the command line or paste it into chat.",
            file=sys.stderr,
        )
        return 1

    from google import genai

    client = genai.Client(api_key=api_key)

    llm_model = os.environ.get("GEMINI_LLM_MODEL", DEFAULT_LLM_MODEL)
    stt_model = os.environ.get("GEMINI_STT_MODEL", DEFAULT_STT_MODEL)
    tts_model = os.environ.get("GEMINI_TTS_MODEL", DEFAULT_TTS_MODEL)

    if args.audio_file:
        audio_bytes = args.audio_file.read_bytes()
        print(f"Using real audio sample: {args.audio_file} ({len(audio_bytes)} bytes)")
    else:
        audio_bytes = _synthesize_placeholder_wav()
        print("No --audio-file given: using a synthetic 2s tone (latency-only, transcript will be meaningless).")

    llm_result = probe_llm(client, llm_model)
    stt_result = probe_stt(client, stt_model, audio_bytes)
    tts_result = probe_tts(client, tts_model)

    metrics = MetricsLog(path=DEFAULT_METRICS_PATH)
    if llm_result.get("ok"):
        _record(metrics, "llm.first_token_ms", llm_result.get("ttfb_ms"), "llm")
    if stt_result.get("ok"):
        _record(metrics, "stt.duration_ms", stt_result.get("total_ms"), "stt")
    if tts_result.get("ok"):
        _record(metrics, "tts.first_byte_ms", tts_result.get("first_byte_ms"), "tts")

    print("\n" + "=" * 60)
    print("Summary (also appended to backend/data/metrics-runs.jsonl):")
    print(f"  LLM  ({llm_model}): {'OK' if llm_result.get('ok') else 'FAILED'}")
    print(f"  STT  ({stt_model}): {'OK' if stt_result.get('ok') else 'FAILED'}")
    print(f"  TTS  ({tts_model}): {'OK' if tts_result.get('ok') else 'FAILED'}")
    print("=" * 60)

    return 0 if all(r.get("ok") for r in (llm_result, stt_result, tts_result)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
