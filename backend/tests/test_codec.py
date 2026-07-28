"""Protocol codec tests — see design doc `### WebSocket wire format` and T1/T7."""

import pytest

from websocket.codec import (
    MSG_TYPE_CLIENT_MIC_PCM,
    MSG_TYPE_SERVER_TTS_PCM,
    InvalidFrameError,
    decode_binary_frame,
    decode_text_frame,
    encode_binary_frame,
    encode_text_frame,
)

# A real epoch-ms value well past the u32 boundary (~4.29 billion / ~49.7 days
# in ms). This is the exact bug T1/D9 fixed: client_ts_ms was originally u32,
# which silently truncated every single frame's timestamp since current
# epoch-ms (~1.78 trillion at review time) is nowhere close to fitting.
REAL_EPOCH_MS = 1_780_000_000_123
U32_MAX = 0xFFFFFFFF


def test_round_trip_basic():
    pcm = bytes(range(64)) * 10  # arbitrary PCM-shaped payload
    encoded = encode_binary_frame(
        msg_type=MSG_TYPE_CLIENT_MIC_PCM,
        seq=42,
        client_ts_ms=1_000,
        playback_generation=0,
        pcm=pcm,
    )
    frame = decode_binary_frame(encoded)

    assert frame.msg_type == MSG_TYPE_CLIENT_MIC_PCM
    assert frame.seq == 42
    assert frame.client_ts_ms == 1_000
    assert frame.playback_generation == 0
    assert frame.pcm == pcm


def test_client_ts_ms_does_not_truncate_epoch_ms():
    """The bug this test guards against: client_ts_ms as u32 would silently
    truncate any real epoch-ms value, corrupting the RTT/latency math the
    whole project exists to measure honestly (see D9)."""
    assert REAL_EPOCH_MS > U32_MAX, "test fixture must exceed u32 range to be meaningful"

    encoded = encode_binary_frame(
        msg_type=MSG_TYPE_SERVER_TTS_PCM,
        seq=1,
        client_ts_ms=REAL_EPOCH_MS,
        playback_generation=7,
        pcm=b"\x00\x01",
    )
    frame = decode_binary_frame(encoded)

    assert frame.client_ts_ms == REAL_EPOCH_MS
    assert frame.playback_generation == 7


def test_header_is_16_bytes():
    encoded = encode_binary_frame(
        msg_type=MSG_TYPE_CLIENT_MIC_PCM,
        seq=0,
        client_ts_ms=0,
        pcm=b"",
    )
    assert len(encoded) == 16


def test_encode_rejects_client_ts_ms_out_of_u64_range():
    with pytest.raises(ValueError):
        encode_binary_frame(
            msg_type=MSG_TYPE_CLIENT_MIC_PCM,
            seq=0,
            client_ts_ms=2**64,
            pcm=b"",
        )


@pytest.mark.parametrize("truncated_length", [0, 1, 8, 15])
def test_decode_truncated_frame_raises_invalid_frame_error(truncated_length):
    """Maps to the `INVALID_FRAME` error code — a malformed/truncated binary
    frame must raise cleanly, not crash the connection or silently misparse."""
    with pytest.raises(InvalidFrameError):
        decode_binary_frame(b"\x00" * truncated_length)


def test_text_frame_round_trip():
    encoded = encode_text_frame(
        type="session.start", session_id="abc-123", ts=1_780_000_000_000, payload={"mode": "continuous"}
    )
    decoded = decode_text_frame(encoded)

    assert decoded == {
        "type": "session.start",
        "session_id": "abc-123",
        "ts": 1_780_000_000_000,
        "payload": {"mode": "continuous"},
    }


def test_decode_text_frame_malformed_json_raises():
    with pytest.raises(InvalidFrameError):
        decode_text_frame("{not valid json")


def test_decode_text_frame_missing_field_raises():
    with pytest.raises(InvalidFrameError):
        decode_text_frame('{"type": "ping"}')
