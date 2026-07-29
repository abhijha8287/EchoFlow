/**
 * Thin WebSocket wrapper: one instance = one connection attempt. Encodes/
 * decodes frames via protocol.ts and dispatches to callbacks. Deliberately
 * does NOT own reconnect policy (backoff scheduling, resume-attempt
 * counting) -- that's stateful orchestration that belongs in
 * hooks/useVoiceSession.ts, which creates a fresh WsClient per attempt.
 */

import {
  type BinaryFrame,
  type EncodeBinaryFrameOptions,
  InvalidFrameError,
  type TextFrame,
  decodeBinaryFrame,
  decodeTextFrame,
  encodeBinaryFrame,
  encodeTextFrame,
} from "@/lib/protocol";

export interface WsClientCallbacks {
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onTextFrame?: (frame: TextFrame) => void;
  onBinaryFrame?: (frame: BinaryFrame) => void;
  /** A malformed frame from the server. Per the design doc, this must not
   * be treated as fatal -- log/surface it, but the connection stays open. */
  onInvalidFrame?: (err: InvalidFrameError) => void;
}

export class WsClient {
  private ws: WebSocket | null = null;

  constructor(
    private readonly url: string,
    private readonly callbacks: WsClientCallbacks,
  ) {}

  connect(): void {
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", () => this.callbacks.onOpen?.());
    ws.addEventListener("close", (event) => this.callbacks.onClose?.(event));
    ws.addEventListener("message", (event) => this.handleMessage(event));
    this.ws = ws;
  }

  private handleMessage(event: MessageEvent<string | ArrayBuffer>): void {
    try {
      if (typeof event.data === "string") {
        this.callbacks.onTextFrame?.(decodeTextFrame(event.data));
      } else {
        this.callbacks.onBinaryFrame?.(decodeBinaryFrame(event.data));
      }
    } catch (err) {
      if (err instanceof InvalidFrameError) {
        this.callbacks.onInvalidFrame?.(err);
        return;
      }
      throw err;
    }
  }

  sendText(frame: TextFrame): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(encodeTextFrame(frame));
  }

  sendBinary(opts: EncodeBinaryFrameOptions): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(encodeBinaryFrame(opts));
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
