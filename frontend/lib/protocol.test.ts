/**
 * Protocol codec tests -- TS mirror of backend/tests/test_codec.py. Same
 * cases, since the two codecs must stay byte-exact (see lib/protocol.ts).
 */

import { describe, expect, it } from "vitest";
import {
  InvalidFrameError,
  MSG_TYPE_CLIENT_MIC_PCM,
  MSG_TYPE_SERVER_TTS_PCM,
  decodeBinaryFrame,
  decodeTextFrame,
  encodeBinaryFrame,
  encodeTextFrame,
} from "./protocol";

// A real epoch-ms value well past the u32 boundary (~4.29 billion / ~49.7
// days in ms) -- the exact bug T1/D9 fixed on the backend. If the frontend
// codec ever regressed to a narrower encoding, this is what would catch it.
const REAL_EPOCH_MS = 1_780_000_000_123;
const U32_MAX = 0xffffffff;

function makePcm(length: number): Uint8Array {
  const pcm = new Uint8Array(length);
  for (let i = 0; i < length; i++) pcm[i] = i % 256;
  return pcm;
}

describe("binary frame", () => {
  it("round-trips basic fields", () => {
    const pcm = makePcm(640);
    const encoded = encodeBinaryFrame({ msgType: MSG_TYPE_CLIENT_MIC_PCM, seq: 42, clientTsMs: 1000, pcm });
    const frame = decodeBinaryFrame(encoded);

    expect(frame.msgType).toBe(MSG_TYPE_CLIENT_MIC_PCM);
    expect(frame.seq).toBe(42);
    expect(frame.clientTsMs).toBe(1000);
    expect(frame.playbackGeneration).toBe(0);
    expect(frame.pcm).toEqual(pcm);
  });

  it("does not truncate a real epoch-ms client_ts_ms", () => {
    expect(REAL_EPOCH_MS).toBeGreaterThan(U32_MAX);

    const encoded = encodeBinaryFrame({
      msgType: MSG_TYPE_SERVER_TTS_PCM,
      seq: 1,
      clientTsMs: REAL_EPOCH_MS,
      playbackGeneration: 7,
      pcm: new Uint8Array([0, 1]),
    });
    const frame = decodeBinaryFrame(encoded);

    expect(frame.clientTsMs).toBe(REAL_EPOCH_MS);
    expect(frame.playbackGeneration).toBe(7);
  });

  it("header is 16 bytes", () => {
    const encoded = encodeBinaryFrame({ msgType: MSG_TYPE_CLIENT_MIC_PCM, seq: 0, clientTsMs: 0, pcm: new Uint8Array(0) });
    expect(encoded.byteLength).toBe(16);
  });

  it("rejects seq out of u32 range", () => {
    expect(() =>
      encodeBinaryFrame({ msgType: MSG_TYPE_CLIENT_MIC_PCM, seq: U32_MAX + 1, clientTsMs: 0, pcm: new Uint8Array(0) }),
    ).toThrow(RangeError);
  });

  it.each([0, 1, 8, 15])("raises InvalidFrameError on a %i-byte truncated frame", (length) => {
    expect(() => decodeBinaryFrame(new ArrayBuffer(length))).toThrow(InvalidFrameError);
  });
});

describe("text frame", () => {
  it("round-trips", () => {
    const encoded = encodeTextFrame({
      type: "session.start",
      session_id: "abc-123",
      ts: 1_780_000_000_000,
      payload: { mode: "continuous" },
    });
    const decoded = decodeTextFrame(encoded);

    expect(decoded).toEqual({
      type: "session.start",
      session_id: "abc-123",
      ts: 1_780_000_000_000,
      payload: { mode: "continuous" },
    });
  });

  it("raises InvalidFrameError on malformed JSON", () => {
    expect(() => decodeTextFrame("{not valid json")).toThrow(InvalidFrameError);
  });

  it("raises InvalidFrameError on a missing required field", () => {
    expect(() => decodeTextFrame(JSON.stringify({ type: "ping" }))).toThrow(InvalidFrameError);
  });
});
