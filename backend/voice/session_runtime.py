"""Per-connection runtime — wires FSM + VAD + STT/LLM/TTS + ConnectionManager
+ MetricsLog into the actual message flow from the design doc's WebSocket
wire format and Core transitions table (T7).

This is the "real" version of T3's cancel-scope-boundary skeleton and T4's
STT branching, now driving them end to end against injected (test-double,
in this repo -- real adapters are separate later work) providers, so the
remaining named test-coverage gaps (`INVALID_FRAME` over the wire, a
full-turn E2E walk, mid-`Speaking` reconnect) have something real to run
against.

One `SessionRuntime` per live WebSocket connection. `Session` (FSM state,
`playback_generation`, ...) lives in `SessionStore` and outlives any single
connection -- `session.resume` reattaches a new `SessionRuntime` to the same
`Session` -- so a stale utterance task from a dropped connection can never
reach the new connection's socket: it only ever holds closures over the
connection it was spawned on.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable

from core.session import Session, SessionStore
from services.metrics import MetricsLog
from stt.algorithm import run_stt
from stt.provider import STTProvider
from voice.fsm import VoiceEvent, VoiceState
from voice.pipeline import run_utterance_pipeline
from voice.vad import VadEvent, VadProvider
from websocket import errors
from websocket.codec import MSG_TYPE_SERVER_TTS_PCM, encode_binary_frame, encode_text_frame
from websocket.connection import MIC_PCM, TEXT, UTTERANCE_READY, ConnectionManager, InboundFrame, Transport

Llm = Callable[[str, str], Awaitable[str]]
Tts = Callable[[str, str], Awaitable[bytes]]


class _EmptyTranscript(Exception):
    """Internal sentinel: stt_stage already performed the SpeechDetected ->
    Listening (EMPTY_TRANSCRIPT) transition; this just short-circuits the
    rest of the pipeline (no LLM/TTS) without being treated as a provider
    error by `_run_utterance`'s generic except-clause."""


def _now_ms() -> int:
    return int(time.time() * 1000)


class SessionRuntime:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        transport: Transport,
        send_text: Callable[[str], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        vad: VadProvider,
        stt_provider: STTProvider,
        llm: Llm,
        tts: Tts,
        metrics: MetricsLog,
        provider_names: dict[str, str],
    ) -> None:
        self._session_store = session_store
        self._send_text = send_text
        self._send_bytes = send_bytes
        self._vad = vad
        self._stt_provider = stt_provider
        self._llm = llm
        self._tts = tts
        self._metrics = metrics
        self._provider_names = provider_names

        self.session: Session | None = None
        self._endpoint_event: asyncio.Event | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        # Accumulated s16le/16k mic audio for the current utterance -- most
        # batch STT providers (Gemini included) have no notion of resuming
        # a prior partial transcription, so each chunk re-sends everything
        # captured so far (see stt/algorithm.py's GetAudio contract).
        self._utterance_pcm = bytearray()

        self.connection = ConnectionManager(
            transport=transport,
            start_utterance=self._run_utterance,
            on_frame=self._on_frame,
        )

    async def run(self) -> None:
        await self.connection.receive_loop()

    def on_disconnect(self) -> None:
        """`* | WS drop | Recovering` (wildcard transition). The session
        itself is left in the store -- only the connection died, not the
        session -- so a later `session.resume` can reattach to it."""
        if self.session is not None:
            self.session.fsm.handle(VoiceEvent.WS_DROP)

    # -- frame dispatch -----------------------------------------------

    def _on_frame(self, frame: InboundFrame) -> None:
        if frame.kind == MIC_PCM:
            self._handle_mic_frame(frame.data)
        elif frame.kind == TEXT:
            self._handle_text_frame(frame.data)

    def _handle_mic_frame(self, binary_frame: Any) -> None:
        session = self.session
        if session is None:
            return
        vad_event = self._vad.process(binary_frame.pcm)
        if vad_event is VadEvent.SPEECH and session.fsm.state is VoiceState.LISTENING:
            session.current_utterance_id = uuid.uuid4().hex
            session.fsm.handle(VoiceEvent.VAD_SPEECH)
            self._endpoint_event = asyncio.Event()
            self._utterance_pcm = bytearray(binary_frame.pcm)
            self._fire(self._emit_state_change())
            self.connection.spawn_utterance(InboundFrame(UTTERANCE_READY, data=session.current_utterance_id))
        elif vad_event is VadEvent.ENDPOINT and session.fsm.state is VoiceState.SPEECH_DETECTED:
            self._utterance_pcm.extend(binary_frame.pcm)
            self._vad.reset()
            if self._endpoint_event is not None:
                self._endpoint_event.set()
        elif session.fsm.state is VoiceState.SPEECH_DETECTED:
            self._utterance_pcm.extend(binary_frame.pcm)

    def _handle_text_frame(self, envelope: dict) -> None:
        msg_type = envelope.get("type")
        payload = envelope.get("payload") or {}
        if msg_type == "session.start":
            self._handle_session_start(payload)
        elif msg_type == "session.resume":
            self._handle_session_resume(payload)
        elif msg_type == "control.interrupt":
            self._handle_control_interrupt()
        elif msg_type == "control.cancel":
            self._handle_control_cancel()
        elif msg_type == "ping":
            self._fire(self._emit("pong", {}))
        # control.ptt_start / control.ptt_end: accepted, no server-side
        # effect -- PTT gating is client-owned (design doc Interaction
        # modes): the client simply never sends control.interrupt for a
        # voice-triggered barge-in while in PTT mode with mic not held.
        # There is nothing for the server to gate.

    # -- session lifecycle ----------------------------------------------

    def _handle_session_start(self, payload: dict) -> None:
        mode = payload.get("mode", "continuous")
        session = self._session_store.create(mode=mode)
        self.session = session
        session.fsm.handle(VoiceEvent.SESSION_START)  # Idle -> Listening
        self._fire(self._emit_session_hello())
        self._fire(self._emit_state_change())

    def _handle_session_resume(self, payload: dict) -> None:
        session_id = payload.get("session_id", "")
        session = self._session_store.get(session_id)
        if session is None:
            self._fire(
                self._emit(
                    "error",
                    {
                        "code": errors.SESSION_NOT_FOUND,
                        "message": f"unknown session_id {session_id!r}",
                        "retriable": False,
                    },
                )
            )
            return
        self.session = session
        if session.fsm.state is VoiceState.RECOVERING:
            session.fsm.handle(VoiceEvent.RESUME_OK)
        self._fire(self._emit_session_hello())
        self._fire(self._emit_state_change())

    def _handle_control_interrupt(self) -> None:
        session = self.session
        if session is None:
            return
        if session.fsm.state in (VoiceState.SPEAKING, VoiceState.THINKING):
            session.fsm.handle(VoiceEvent.INTERRUPT)
            self._fire(self._emit_state_change())
        self.connection.interrupt()

    def _handle_control_cancel(self) -> None:
        session = self.session
        if session is None:
            return
        if session.fsm.state is VoiceState.THINKING:
            session.fsm.handle(VoiceEvent.CONTROL_CANCEL)
            self._fire(self._emit_state_change())
        self.connection.interrupt()

    # -- utterance pipeline (STT -> LLM -> TTS) --------------------------

    async def _run_utterance(self, frame: InboundFrame) -> None:
        session = self.session
        assert session is not None
        utterance_id = frame.data
        endpoint = self._endpoint_event
        assert endpoint is not None

        async def stt_stage(_uid: str) -> str:
            start = time.monotonic()

            def on_partial(text: str) -> None:
                self._fire(self._emit("transcript.partial", {"text": text, "utterance_id": _uid}))

            text = await run_stt(self._stt_provider, _uid, on_partial, endpoint, lambda: bytes(self._utterance_pcm))
            if not endpoint.is_set():
                # True-streaming providers finish their own iterator
                # independently of VAD; wait for the endpoint here too, so
                # the SpeechDetected -> Thinking transition below always
                # happens in the FSM-correct order (endpoint-committed
                # turn-taking, not "whichever finishes first").
                await endpoint.wait()
            self._record_stage("stt.duration_ms", start, _uid, self._provider_names["stt"], "stt")

            if not text.strip():
                session.fsm.handle(VoiceEvent.EMPTY_TRANSCRIPT)
                await self._emit_state_change()
                raise _EmptyTranscript()

            session.fsm.handle(VoiceEvent.VAD_ENDPOINT)
            await self._emit_state_change()
            await self._emit("transcript.final", {"text": text, "utterance_id": _uid})
            return text

        async def llm_stage(_uid: str, transcript: str) -> str:
            start = time.monotonic()
            response = await self._llm(_uid, transcript)
            self._record_stage("llm.first_token_ms", start, _uid, self._provider_names["llm"], "llm")
            await self._emit("llm.token", {"text": response, "utterance_id": _uid})
            return response

        async def tts_stage(_uid: str, response: str) -> None:
            start = time.monotonic()
            pcm = await self._tts(_uid, response)
            self._record_stage("tts.first_byte_ms", start, _uid, self._provider_names["tts"], "tts")
            if session.fsm.state is VoiceState.THINKING:
                session.fsm.handle(VoiceEvent.TTS_FIRST_CHUNK)
                await self._emit_state_change()
            await self._send_audio_frame(pcm)

        try:
            await run_utterance_pipeline(utterance_id, stt_stage, llm_stage, tts_stage)
        except asyncio.CancelledError:
            if session.fsm.state is VoiceState.INTERRUPTED:
                session.fsm.handle(VoiceEvent.CANCEL_COMPLETE)
                await self._emit_state_change()
            raise
        except _EmptyTranscript:
            return
        except Exception as exc:  # provider error path (incl. RATE_LIMITED)
            await self._emit_provider_error(exc)
            if session.fsm.state in (VoiceState.THINKING, VoiceState.SPEAKING):
                session.fsm.handle(VoiceEvent.PROVIDER_ERROR)
                await self._emit_state_change()
            return

        if session.fsm.state is VoiceState.SPEAKING:
            session.fsm.handle(VoiceEvent.TURN_COMPLETE)
            await self._emit_state_change()

    def _record_stage(self, stage: str, start: float, utterance_id: str, provider: str, role: str) -> None:
        duration_ms = (time.monotonic() - start) * 1000
        self._metrics.record(
            stage=stage, duration_ms=duration_ms, utterance_id=utterance_id, provider=provider, role=role
        )
        self._fire(
            self._emit(
                "metrics.stage",
                {
                    "stage": stage,
                    "t_server_ms": _now_ms(),
                    "duration_ms": duration_ms,
                    "utterance_id": utterance_id,
                    "provider": provider,
                },
            )
        )

    # -- outbound -----------------------------------------------------

    async def _emit(self, type_: str, payload: dict) -> None:
        session_id = self.session.session_id if self.session else ""
        envelope = encode_text_frame(type=type_, session_id=session_id, ts=_now_ms(), payload=payload)
        await self._send_text(envelope)

    async def _emit_session_hello(self) -> None:
        assert self.session is not None
        await self._emit("session.hello", {"session_id": self.session.session_id, "server_ts": _now_ms()})

    async def _emit_state_change(self) -> None:
        assert self.session is not None
        await self._emit("state.change", {"state": self.session.fsm.state.name})

    async def _emit_provider_error(self, exc: Exception) -> None:
        code = getattr(exc, "code", errors.PROVIDER_ERROR)
        retriable = getattr(exc, "retriable", True)
        await self._emit("error", {"code": code, "message": str(exc), "retriable": retriable})

    async def _send_audio_frame(self, pcm: bytes) -> None:
        session = self.session
        assert session is not None
        session.outbound_seq += 1
        frame_bytes = encode_binary_frame(
            msg_type=MSG_TYPE_SERVER_TTS_PCM,
            seq=session.outbound_seq,
            client_ts_ms=_now_ms(),
            playback_generation=session.playback_generation,
            pcm=pcm,
        )
        await self._send_bytes(frame_bytes)

    def _fire(self, coro: Awaitable[None]) -> None:
        """Schedule fire-and-forget async work from a sync callback (e.g.
        `on_partial`, or the sync parts of frame dispatch). Keeps a strong
        reference until done so the task can't be GC'd mid-flight."""
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
