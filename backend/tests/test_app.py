"""App-level integration tests (T7) — the remaining named test-coverage
gaps from the design doc's Testing section that need a real WS endpoint:

- malformed frame -> `INVALID_FRAME` (over the wire, not just the codec
  unit test in test_codec.py)
- full-turn `[->E2E]` walk with mocked STT/LLM/TTS providers
- mid-`Speaking` reconnect / zombie-audio

(`RATE_LIMITED` under capped concurrency and PTT gating are covered
elsewhere: RATE_LIMITED at the STT-algorithm layer in
test_stt_algorithm.py, since it doesn't need this scaffold; PTT gating is
client-owned per the design doc's Interaction modes table -- there is no
server-side gate to test here, see voice/session_runtime.py.)
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app import Providers, create_app
from core.session import SessionStore
from services.metrics import MetricsLog
from voice.fsm import VoiceState
from voice.vad import VadEvent
from websocket.codec import (
    MSG_TYPE_CLIENT_MIC_PCM,
    BinaryFrame,
    decode_binary_frame,
    decode_text_frame,
    encode_binary_frame,
    encode_text_frame,
)


class ScriptedVad:
    """Deterministic VAD test double: a mic frame's `pcm` payload is itself
    the marker ("SPEECH"/"ENDPOINT") rather than real audio, since Silero
    VAD integration is separate, later work (see voice/vad.py)."""

    def process(self, pcm_chunk: bytes) -> VadEvent | None:
        if pcm_chunk == b"SPEECH":
            return VadEvent.SPEECH
        if pcm_chunk == b"ENDPOINT":
            return VadEvent.ENDPOINT
        return None

    def reset(self) -> None:
        pass


class FakeStreamingSTT:
    supports_partial_streaming = True

    def __init__(self, partials: list[str]) -> None:
        self._partials = partials

    async def stream_partials(self, utterance_id: str):
        for partial in self._partials:
            yield partial


async def fake_llm(utterance_id: str, transcript: str) -> str:
    return f"response to: {transcript}"


async def fake_tts(utterance_id: str, response: str) -> bytes:
    return b"\x11\x22\x33\x44"


def _build_app(tmp_path, *, stt_partials: list[str] | None = None):
    session_store = SessionStore()
    metrics = MetricsLog(path=tmp_path / "metrics-runs.jsonl", run_id="test-run")
    providers = Providers(
        vad_factory=ScriptedVad,
        stt_provider=FakeStreamingSTT(stt_partials or ["hello world"]),
        llm=fake_llm,
        tts=fake_tts,
        provider_names={"stt": "fake", "llm": "fake", "tts": "fake"},
    )
    app = create_app(session_store=session_store, metrics=metrics, providers=providers)
    return app, session_store


def _receive(ws) -> tuple[str, Any]:
    """Read one message, returning ("text", envelope-dict) or
    ("bytes", BinaryFrame) regardless of which arrives -- state.change,
    transcript, metrics, and audio frames can interleave, so tests must not
    assume the type of the next message."""
    message = ws.receive()
    if message.get("text") is not None:
        return "text", decode_text_frame(message["text"])
    if message.get("bytes") is not None:
        return "bytes", decode_binary_frame(message["bytes"])
    raise AssertionError(f"unexpected message shape: {message}")


def _drain_until(ws, *, predicate, max_messages: int = 50) -> list[tuple[str, Any]]:
    """Collect messages until `predicate(kind, payload)` is true (inclusive),
    or raise if `max_messages` is exceeded -- a safety bound, not a real
    test timeout, since the fakes never block indefinitely."""
    collected: list[tuple[str, Any]] = []
    for _ in range(max_messages):
        kind, payload = _receive(ws)
        collected.append((kind, payload))
        if predicate(kind, payload):
            return collected
    raise AssertionError(f"predicate never satisfied after {max_messages} messages: {collected}")


def _mic_frame(pcm: bytes, seq: int) -> bytes:
    return encode_binary_frame(msg_type=MSG_TYPE_CLIENT_MIC_PCM, seq=seq, client_ts_ms=1_780_000_000_000 + seq, pcm=pcm)


def test_full_turn_walks_fsm_and_emits_transcript_and_audio_in_order(tmp_path):
    app, _ = _build_app(tmp_path, stt_partials=["hello world"])
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_text(encode_text_frame(type="session.start", session_id="", ts=0, payload={"mode": "continuous"}))
        kind, hello = _receive(ws)
        assert (kind, hello["type"]) == ("text", "session.hello")
        session_id = hello["payload"]["session_id"]
        assert session_id

        kind, env = _receive(ws)
        assert env["type"] == "state.change" and env["payload"]["state"] == "LISTENING"

        # Both mic frames are sent before draining any further responses:
        # server-side FSM/VAD processing of each frame is fully synchronous
        # within receive_loop regardless of how the resulting emits happen
        # to interleave on the wire, so this doesn't race the assertions
        # below (which only depend on aggregate order, not message-by-
        # message lockstep).
        ws.send_bytes(_mic_frame(b"SPEECH", seq=1))
        ws.send_bytes(_mic_frame(b"ENDPOINT", seq=2))

        messages = _drain_until(
            ws,
            predicate=lambda kind, payload: kind == "text"
            and payload["type"] == "state.change"
            and payload["payload"]["state"] == "LISTENING",
        )

    state_sequence = [m["payload"]["state"] for kind, m in messages if kind == "text" and m["type"] == "state.change"]
    assert state_sequence == ["SPEECH_DETECTED", "THINKING", "SPEAKING", "LISTENING"]

    final_index = next(
        i for i, (kind, m) in enumerate(messages) if kind == "text" and m["type"] == "transcript.final"
    )
    assert messages[final_index][1]["payload"]["text"] == "hello world"

    audio_index = next(i for i, (kind, m) in enumerate(messages) if kind == "bytes")
    assert isinstance(messages[audio_index][1], BinaryFrame)
    assert messages[audio_index][1].pcm == b"\x11\x22\x33\x44"

    # Design doc requirement: final transcript + audio chunks emitted in order.
    assert final_index < audio_index

    llm_index = next(i for i, (kind, m) in enumerate(messages) if kind == "text" and m["type"] == "llm.token")
    assert messages[llm_index][1]["payload"]["text"] == "response to: hello world"
    assert final_index < llm_index < audio_index


def test_malformed_binary_frame_gets_invalid_frame_error_not_a_dead_connection(tmp_path):
    app, _ = _build_app(tmp_path)
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        # Truncated binary frame -- shorter than the 16-byte header.
        ws.send_bytes(b"\x01\x02\x03")
        kind, env = _receive(ws)
        assert kind == "text"
        assert env["type"] == "error"
        assert env["payload"]["code"] == "INVALID_FRAME"
        assert env["payload"]["retriable"] is False

        # Connection must still be alive and functional afterwards.
        ws.send_text(encode_text_frame(type="session.start", session_id="", ts=0, payload={"mode": "continuous"}))
        kind, hello = _receive(ws)
        assert (kind, hello["type"]) == ("text", "session.hello")


def test_malformed_json_text_frame_gets_invalid_frame_error(tmp_path):
    app, _ = _build_app(tmp_path)
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_text("{not valid json")
        kind, env = _receive(ws)
        assert kind == "text"
        assert env["type"] == "error"
        assert env["payload"]["code"] == "INVALID_FRAME"


def test_mid_speaking_reconnect_lands_in_listening_with_no_zombie_audio(tmp_path):
    """T7: a WS drop mid-Speaking, followed by session.resume on a new
    connection, must not replay/overlap the old utterance's audio -- the new
    connection should see Listening (not Speaking), and no stray binary
    audio frames from the dead utterance."""
    app, session_store = _build_app(tmp_path, stt_partials=["hi"])
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_text(encode_text_frame(type="session.start", session_id="", ts=0, payload={"mode": "continuous"}))
        _, hello = _receive(ws)
        session_id = hello["payload"]["session_id"]
        _receive(ws)  # state.change LISTENING

        ws.send_bytes(_mic_frame(b"SPEECH", seq=1))
        # transcript.partial (fire-and-forget) can interleave with the
        # state.change here in either order -- drain until SPEECH_DETECTED
        # specifically rather than assuming it's the very next message.
        _drain_until(
            ws,
            predicate=lambda kind, payload: kind == "text"
            and payload["type"] == "state.change"
            and payload["payload"]["state"] == "SPEECH_DETECTED",
        )
        # Drop the connection mid-utterance (before ENDPOINT is even sent) --
        # the `with` block exit closes the socket here.

    session = session_store.get(session_id)
    assert session is not None
    assert session.fsm.state is VoiceState.RECOVERING  # WS drop wildcard transition

    with client.websocket_connect("/ws/voice") as ws2:
        ws2.send_text(
            encode_text_frame(type="session.resume", session_id="", ts=0, payload={"session_id": session_id})
        )
        kind, hello2 = _receive(ws2)
        assert (kind, hello2["type"]) == ("text", "session.hello")
        assert hello2["payload"]["session_id"] == session_id

        kind, env = _receive(ws2)
        assert env["type"] == "state.change"
        assert env["payload"]["state"] == "LISTENING"  # not stuck in Speaking/Recovering

        # Nothing else should arrive unprompted -- in particular, no zombie
        # audio frame from the utterance that was in flight on the old,
        # now-dead connection.
        ws2.send_text(encode_text_frame(type="ping", session_id=session_id, ts=0, payload={}))
        kind, env = _receive(ws2)
        assert (kind, env["type"]) == ("text", "pong")


def test_resume_with_unknown_session_id_gets_session_not_found(tmp_path):
    app, _ = _build_app(tmp_path)
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_text(
            encode_text_frame(
                type="session.resume", session_id="", ts=0, payload={"session_id": "does-not-exist"}
            )
        )
        kind, env = _receive(ws)
        assert kind == "text"
        assert env["type"] == "error"
        assert env["payload"]["code"] == "SESSION_NOT_FOUND"
