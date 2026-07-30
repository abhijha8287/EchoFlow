"""Boot entrypoint for the real backend — not the pytest suite (which uses
fakes via app.py's `Providers` injection) and not the spike script (which
makes standalone probe calls, not a live server).

Real Silero VAD is not implemented yet (see voice/vad.py's module
docstring) -- this wires `SimpleEnergyVad`, a crude energy-threshold
placeholder, so the server is genuinely runnable end to end for a live
demo in the meantime, rather than blocked entirely on VAD model work.
Swapping in a real Silero-backed `VadProvider` later only touches this
file's `vad_factory` line -- `SessionRuntime` depends on the Protocol, not
this specific implementation.

Requires `GEMINI_API_KEY` in `backend/.env` (gitignored, see
`.env.example`). Boot-time validation, per the design doc's Provider
registry behavior: fails fast with a clear message rather than accepting a
connection and failing mid-conversation.
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from google import genai

from app import Providers, create_app
from core.session import SessionStore
from llm.gemini import DEFAULT_MODEL as LLM_DEFAULT_MODEL
from llm.gemini import create_gemini_llm
from services.metrics import MetricsLog
from stt.gemini import DEFAULT_MODEL as STT_DEFAULT_MODEL
from stt.gemini import GeminiSTTProvider
from tts.gemini import DEFAULT_MODEL as TTS_DEFAULT_MODEL
from tts.gemini import DEFAULT_VOICE as TTS_DEFAULT_VOICE
from tts.gemini import create_gemini_tts
from voice.simple_vad import SimpleEnergyVad


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Create backend/.env (gitignored, see .env.example) with a line like:\n"
            f"  {name}=your-key-here\n"
            "Never pass it on the command line or paste it into chat."
        )
    return value


def main() -> None:
    load_dotenv()
    api_key = _require_env("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    llm_model = os.environ.get("GEMINI_LLM_MODEL", LLM_DEFAULT_MODEL)
    stt_model = os.environ.get("GEMINI_STT_MODEL", STT_DEFAULT_MODEL)
    tts_model = os.environ.get("GEMINI_TTS_MODEL", TTS_DEFAULT_MODEL)

    providers = Providers(
        vad_factory=SimpleEnergyVad,
        stt_provider=GeminiSTTProvider(client, model=stt_model),
        llm=create_gemini_llm(client, model=llm_model),
        tts=create_gemini_tts(client, model=tts_model, voice=TTS_DEFAULT_VOICE),
        provider_names={"stt": "gemini", "llm": "gemini", "tts": "gemini"},
    )
    app = create_app(session_store=SessionStore(), metrics=MetricsLog(), providers=providers)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    # 1 worker, per the design doc's Compose topology -- session state is
    # an in-memory SessionStore, which a second worker process would not
    # share.
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
