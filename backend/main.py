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

Provider selection is env-switchable per role (`STT_PROVIDER`,
`LLM_PROVIDER`, `TTS_PROVIDER`, each defaulting to `gemini`) -- see
`.env.example`. Each role's vendor client is built lazily and cached, so
running all three roles against the same vendor (the common case) only
constructs one client, and only the API key(s) actually needed are
required -- e.g. an all-OpenAI setup never demands `GEMINI_API_KEY`.

Boot-time validation, per the design doc's Provider registry behavior:
fails fast with a clear message rather than accepting a connection and
failing mid-conversation.
"""

from __future__ import annotations

import os

import openai
import uvicorn
from dotenv import load_dotenv
from google import genai

from app import Providers, create_app
from core.session import SessionStore
from llm.gemini import DEFAULT_MODEL as GEMINI_LLM_DEFAULT_MODEL
from llm.gemini import create_gemini_llm
from llm.openai import DEFAULT_MODEL as OPENAI_LLM_DEFAULT_MODEL
from llm.openai import create_openai_llm
from services.metrics import MetricsLog
from stt.gemini import DEFAULT_MODEL as GEMINI_STT_DEFAULT_MODEL
from stt.gemini import GeminiSTTProvider
from stt.openai import DEFAULT_MODEL as OPENAI_STT_DEFAULT_MODEL
from stt.openai import OpenAISTTProvider
from tts.gemini import DEFAULT_MODEL as GEMINI_TTS_DEFAULT_MODEL
from tts.gemini import DEFAULT_VOICE as GEMINI_TTS_DEFAULT_VOICE
from tts.gemini import create_gemini_tts
from tts.openai import DEFAULT_MODEL as OPENAI_TTS_DEFAULT_MODEL
from tts.openai import DEFAULT_VOICE as OPENAI_TTS_DEFAULT_VOICE
from tts.openai import create_openai_tts
from voice.simple_vad import SimpleEnergyVad

SUPPORTED_PROVIDERS = ("gemini", "openai")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Create backend/.env (gitignored, see .env.example) with a line like:\n"
            f"  {name}=your-key-here\n"
            "Never pass it on the command line or paste it into chat."
        )
    return value


class _ClientCache:
    """Builds each vendor client at most once, on first use -- so a role
    that isn't configured for a vendor never forces that vendor's API key
    to be present, and roles sharing a vendor share one client."""

    def __init__(self) -> None:
        self._clients: dict[str, object] = {}

    def get(self, provider: str) -> object:
        if provider not in self._clients:
            if provider == "gemini":
                self._clients[provider] = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
            elif provider == "openai":
                self._clients[provider] = openai.AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
            else:
                raise SystemExit(f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
        return self._clients[provider]


def _build_stt_provider(provider: str, clients: _ClientCache):
    if provider == "gemini":
        model = os.environ.get("GEMINI_STT_MODEL", GEMINI_STT_DEFAULT_MODEL)
        return GeminiSTTProvider(clients.get("gemini"), model=model)
    if provider == "openai":
        model = os.environ.get("OPENAI_STT_MODEL", OPENAI_STT_DEFAULT_MODEL)
        return OpenAISTTProvider(clients.get("openai"), model=model)
    raise SystemExit(f"Unknown STT_PROVIDER {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")


def _build_llm(provider: str, clients: _ClientCache):
    if provider == "gemini":
        model = os.environ.get("GEMINI_LLM_MODEL", GEMINI_LLM_DEFAULT_MODEL)
        return create_gemini_llm(clients.get("gemini"), model=model)
    if provider == "openai":
        model = os.environ.get("OPENAI_LLM_MODEL", OPENAI_LLM_DEFAULT_MODEL)
        return create_openai_llm(clients.get("openai"), model=model)
    raise SystemExit(f"Unknown LLM_PROVIDER {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")


def _build_tts(provider: str, clients: _ClientCache):
    if provider == "gemini":
        model = os.environ.get("GEMINI_TTS_MODEL", GEMINI_TTS_DEFAULT_MODEL)
        return create_gemini_tts(clients.get("gemini"), model=model, voice=GEMINI_TTS_DEFAULT_VOICE)
    if provider == "openai":
        model = os.environ.get("OPENAI_TTS_MODEL", OPENAI_TTS_DEFAULT_MODEL)
        voice = os.environ.get("OPENAI_TTS_VOICE", OPENAI_TTS_DEFAULT_VOICE)
        return create_openai_tts(clients.get("openai"), model=model, voice=voice)
    raise SystemExit(f"Unknown TTS_PROVIDER {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")


def main() -> None:
    load_dotenv()

    stt_name = os.environ.get("STT_PROVIDER", "gemini")
    llm_name = os.environ.get("LLM_PROVIDER", "gemini")
    tts_name = os.environ.get("TTS_PROVIDER", "gemini")
    clients = _ClientCache()

    providers = Providers(
        vad_factory=SimpleEnergyVad,
        stt_provider=_build_stt_provider(stt_name, clients),
        llm=_build_llm(llm_name, clients),
        tts=_build_tts(tts_name, clients),
        provider_names={"stt": stt_name, "llm": llm_name, "tts": tts_name},
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
