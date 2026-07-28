"""WS wire protocol codec (v1, frozen) — see design doc `### WebSocket wire format`.

Binary frame header (16 bytes, big-endian):

    offset 0:  u8  msg_type
    offset 1:  u32 seq
    offset 5:  u64 client_ts_ms   (widened from u32 during /plan-eng-review —
                                    epoch-ms needs ~41 bits, u32 silently
                                    truncated every frame)
    offset 13: u16 playback_generation   // server->client audio only; 0 otherwise
    offset 15: u8  reserved
    offset 16: pcm s16le mono 16kHz samples
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

MSG_TYPE_CLIENT_MIC_PCM = 0x01
MSG_TYPE_SERVER_TTS_PCM = 0x02

_HEADER_FORMAT = ">BIQHB"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
assert _HEADER_SIZE == 16, f"binary frame header must be 16 bytes, got {_HEADER_SIZE}"

_U32_MAX = 0xFFFFFFFF
_U64_MAX = 0xFFFFFFFFFFFFFFFF
_U16_MAX = 0xFFFF


class InvalidFrameError(ValueError):
    """Raised on a malformed or truncated binary frame — maps to the `INVALID_FRAME` error code."""


@dataclass(frozen=True, slots=True)
class BinaryFrame:
    msg_type: int
    seq: int
    client_ts_ms: int
    playback_generation: int
    reserved: int
    pcm: bytes


def encode_binary_frame(
    *,
    msg_type: int,
    seq: int,
    client_ts_ms: int,
    pcm: bytes,
    playback_generation: int = 0,
    reserved: int = 0,
) -> bytes:
    if not (0 <= seq <= _U32_MAX):
        raise ValueError(f"seq out of u32 range: {seq}")
    if not (0 <= client_ts_ms <= _U64_MAX):
        raise ValueError(f"client_ts_ms out of u64 range: {client_ts_ms}")
    if not (0 <= playback_generation <= _U16_MAX):
        raise ValueError(f"playback_generation out of u16 range: {playback_generation}")
    header = struct.pack(
        _HEADER_FORMAT, msg_type, seq, client_ts_ms, playback_generation, reserved
    )
    return header + pcm


def decode_binary_frame(data: bytes) -> BinaryFrame:
    if len(data) < _HEADER_SIZE:
        raise InvalidFrameError(
            f"frame too short: {len(data)} bytes, need at least {_HEADER_SIZE}"
        )
    msg_type, seq, client_ts_ms, playback_generation, reserved = struct.unpack(
        _HEADER_FORMAT, data[:_HEADER_SIZE]
    )
    return BinaryFrame(
        msg_type=msg_type,
        seq=seq,
        client_ts_ms=client_ts_ms,
        playback_generation=playback_generation,
        reserved=reserved,
        pcm=data[_HEADER_SIZE:],
    )


def encode_text_frame(*, type: str, session_id: str, ts: int, payload: dict[str, Any]) -> str:
    return json.dumps({"type": type, "session_id": session_id, "ts": ts, "payload": payload})


def decode_text_frame(data: str) -> dict[str, Any]:
    try:
        frame = json.loads(data)
    except json.JSONDecodeError as exc:
        raise InvalidFrameError(f"malformed JSON text frame: {exc}") from exc
    for required in ("type", "session_id", "ts", "payload"):
        if required not in frame:
            raise InvalidFrameError(f"text frame missing required field: {required}")
    return frame
