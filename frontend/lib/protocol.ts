/**
 * WS wire protocol codec (v1, frozen) -- TS mirror of
 * backend/websocket/codec.py. Must stay byte-exact with the backend: same
 * 16-byte binary header layout, same u64 client_ts_ms (widened from u32
 * during /plan-eng-review -- epoch-ms needs ~41 bits), same JSON text
 * frame envelope and required-field validation.
 *
 * Binary frame header (16 bytes, big-endian):
 *   offset 0:  u8  msgType
 *   offset 1:  u32 seq
 *   offset 5:  u64 clientTsMs
 *   offset 13: u16 playbackGeneration   // server->client audio only; 0 otherwise
 *   offset 15: u8  reserved
 *   offset 16: pcm s16le mono 16kHz samples
 */

export const MSG_TYPE_CLIENT_MIC_PCM = 0x01;
export const MSG_TYPE_SERVER_TTS_PCM = 0x02;

const HEADER_SIZE = 16;
const U32_MAX = 0xffffffff;
const U16_MAX = 0xffff;

export class InvalidFrameError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidFrameError";
  }
}

export interface BinaryFrame {
  msgType: number;
  seq: number;
  /** Epoch milliseconds. A JS `number`, not `bigint` -- epoch-ms is nowhere
   * close to Number.MAX_SAFE_INTEGER, and every call site (Date.now(),
   * comparisons, arithmetic) is far simpler with plain numbers. The u64
   * wire representation is an implementation detail of encode/decode. */
  clientTsMs: number;
  playbackGeneration: number;
  reserved: number;
  pcm: Uint8Array;
}

export interface EncodeBinaryFrameOptions {
  msgType: number;
  seq: number;
  clientTsMs: number;
  pcm: Uint8Array;
  playbackGeneration?: number;
  reserved?: number;
}

export function encodeBinaryFrame(opts: EncodeBinaryFrameOptions): ArrayBuffer {
  const { msgType, seq, clientTsMs, pcm, playbackGeneration = 0, reserved = 0 } = opts;

  if (seq < 0 || seq > U32_MAX) {
    throw new RangeError(`seq out of u32 range: ${seq}`);
  }
  if (clientTsMs < 0 || clientTsMs > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`clientTsMs out of safe range: ${clientTsMs}`);
  }
  if (playbackGeneration < 0 || playbackGeneration > U16_MAX) {
    throw new RangeError(`playbackGeneration out of u16 range: ${playbackGeneration}`);
  }

  const buffer = new ArrayBuffer(HEADER_SIZE + pcm.byteLength);
  const view = new DataView(buffer);
  view.setUint8(0, msgType);
  view.setUint32(1, seq, false);
  view.setBigUint64(5, BigInt(clientTsMs), false);
  view.setUint16(13, playbackGeneration, false);
  view.setUint8(15, reserved);
  new Uint8Array(buffer, HEADER_SIZE).set(pcm);
  return buffer;
}

export function decodeBinaryFrame(data: ArrayBuffer): BinaryFrame {
  if (data.byteLength < HEADER_SIZE) {
    throw new InvalidFrameError(`frame too short: ${data.byteLength} bytes, need at least ${HEADER_SIZE}`);
  }
  const view = new DataView(data);
  return {
    msgType: view.getUint8(0),
    seq: view.getUint32(1, false),
    clientTsMs: Number(view.getBigUint64(5, false)),
    playbackGeneration: view.getUint16(13, false),
    reserved: view.getUint8(15),
    pcm: new Uint8Array(data, HEADER_SIZE),
  };
}

export interface TextFrame {
  type: string;
  session_id: string;
  ts: number;
  payload: Record<string, unknown>;
}

export function encodeTextFrame(frame: TextFrame): string {
  return JSON.stringify(frame);
}

const REQUIRED_TEXT_FRAME_FIELDS = ["type", "session_id", "ts", "payload"] as const;

export function decodeTextFrame(data: string): TextFrame {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch (err) {
    throw new InvalidFrameError(`malformed JSON text frame: ${(err as Error).message}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new InvalidFrameError("text frame is not a JSON object");
  }
  const obj = parsed as Record<string, unknown>;
  for (const field of REQUIRED_TEXT_FRAME_FIELDS) {
    if (!(field in obj)) {
      throw new InvalidFrameError(`text frame missing required field: ${field}`);
    }
  }
  return obj as unknown as TextFrame;
}
