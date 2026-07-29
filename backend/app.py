"""FastAPI app factory — the real `/ws/voice` endpoint wiring
`SessionRuntime` (FSM + VAD + STT/LLM/TTS + `ConnectionManager` +
`MetricsLog`) to a live WebSocket (T7 scaffold).

Concrete Gemini/OpenAI/ElevenLabs provider adapters and a real Silero VAD
are separate, later work (see design doc Provider spike). `create_app()`
takes all of them as injected dependencies (`Providers`) so this module has
no vendor-specific code, and tests can supply fakes without a real server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import FastAPI, WebSocket

from core.session import SessionStore
from services.metrics import MetricsLog
from stt.provider import STTProvider
from voice.session_runtime import SessionRuntime
from voice.vad import VadProvider
from websocket import errors
from websocket.codec import InvalidFrameError, decode_binary_frame, decode_text_frame, encode_text_frame
from websocket.connection import MIC_PCM, TEXT, ConnectionClosed, InboundFrame


def _now_ms() -> int:
    return int(time.time() * 1000)


class StarletteWsTransport:
    """Adapts a live `starlette.websockets.WebSocket` to the `Transport`
    seam `ConnectionManager` expects (see backend/websocket/connection.py).

    Also owns malformed-frame recovery: a codec `InvalidFrameError` (bad
    binary header, truncated frame, unparseable JSON, missing required
    field) must not crash the connection -- it sends an `error`
    `INVALID_FRAME` frame and keeps waiting for the next message, instead of
    propagating out of `receive()` and killing `receive_loop()`.
    """

    def __init__(self, websocket: WebSocket, send_text: Callable[[str], Awaitable[None]]) -> None:
        self._ws = websocket
        self._send_text = send_text

    async def receive(self) -> InboundFrame:
        while True:
            message = await self._ws.receive()
            if message["type"] == "websocket.disconnect":
                raise ConnectionClosed()
            raw_bytes = message.get("bytes")
            raw_text = message.get("text")
            try:
                if raw_bytes is not None:
                    return InboundFrame(MIC_PCM, data=decode_binary_frame(raw_bytes))
                if raw_text is not None:
                    return InboundFrame(TEXT, data=decode_text_frame(raw_text))
            except InvalidFrameError as exc:
                await self._send_text(
                    encode_text_frame(
                        type="error",
                        session_id="",
                        ts=_now_ms(),
                        payload={"code": errors.INVALID_FRAME, "message": str(exc), "retriable": False},
                    )
                )
                continue
            # neither bytes nor text -- unexpected ASGI message shape, ignore and wait for the next one


@dataclass(frozen=True, slots=True)
class Providers:
    """Per-role dependencies `SessionRuntime` is wired against. `llm`/`tts`
    are plain async callables (see `voice.pipeline` stage types), not full
    provider objects -- P0a's cascade doesn't need more than that, and it
    keeps this module provider-agnostic."""

    vad_factory: Callable[[], VadProvider]
    stt_provider: STTProvider
    llm: Callable[[str, str], Awaitable[str]]
    tts: Callable[[str, str], Awaitable[bytes]]
    provider_names: dict[str, str]


def create_app(*, session_store: SessionStore, metrics: MetricsLog, providers: Providers) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/voice")
    async def ws_voice(websocket: WebSocket) -> None:
        await websocket.accept()

        async def send_text(data: str) -> None:
            await websocket.send_text(data)

        async def send_bytes(data: bytes) -> None:
            await websocket.send_bytes(data)

        transport = StarletteWsTransport(websocket, send_text=send_text)
        runtime = SessionRuntime(
            session_store=session_store,
            transport=transport,
            send_text=send_text,
            send_bytes=send_bytes,
            vad=providers.vad_factory(),
            stt_provider=providers.stt_provider,
            llm=providers.llm,
            tts=providers.tts,
            metrics=metrics,
            provider_names=providers.provider_names,
        )
        try:
            await runtime.run()
        finally:
            runtime.on_disconnect()

    return app
