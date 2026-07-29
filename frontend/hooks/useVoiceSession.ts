"use client";

/**
 * Top-level voice session hook: wires WsClient (lib/ws-client.ts) + the FSM
 * mirror (stores/session.ts) + connection state (stores/connection.ts) +
 * mic capture (useMicrophone) + playback (usePlayback) together, per the
 * design doc's WebSocket wire format and Core transitions table.
 *
 * Reconnect/resume (T2, frontend half): on an unexpected close, the FSM
 * mirrors WS_DROP -> Recovering and a reconnect is scheduled with backoff
 * (stores/connection.ts). Each reconnect attempt sends `session.resume`
 * with the existing session_id; after MAX_RESUME_ATTEMPTS or a
 * SESSION_NOT_FOUND error, it stops retrying resume and falls back to a
 * fresh `session.start` instead (RESUME_FAILED -> Listening), matching the
 * backend's SessionRuntime._handle_session_resume (T7).
 */

import { useCallback, useEffect, useRef } from "react";
import { isLikelySpeech } from "@/lib/audio";
import { MSG_TYPE_CLIENT_MIC_PCM, type TextFrame } from "@/lib/protocol";
import { WsClient } from "@/lib/ws-client";
import { VoiceEvent, VoiceState } from "@/lib/fsm";
import { MAX_RESUME_ATTEMPTS, reconnectDelayMs, useConnectionStore } from "@/stores/connection";
import { type SessionMode, useSessionStore } from "@/stores/session";
import { useMicrophone } from "@/hooks/useMicrophone";
import { usePlayback } from "@/hooks/usePlayback";

const DEFAULT_WS_URL = "ws://localhost:8000/ws/voice";

function wsUrl(): string {
  return process.env.NEXT_PUBLIC_WS_URL ?? DEFAULT_WS_URL;
}

export interface UseVoiceSessionResult {
  connect: (mode?: SessionMode) => void;
  disconnect: () => void;
  interrupt: () => void;
  /** PTT mode only -- gate mic frame forwarding to "Space held", and tell
   * the server via control.ptt_start/control.ptt_end (design doc's
   * `session.start` payload mode + Interaction modes table). */
  pttStart: () => void;
  pttEnd: () => void;
  micError: string | null;
  isCapturing: boolean;
}

export function useVoiceSession(): UseVoiceSessionResult {
  const wsRef = useRef<WsClient | null>(null);
  const seqRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const pttHeldRef = useRef(false);
  const intentionalCloseRef = useRef(false);

  // Destructured rather than kept as `mic`/`playback` objects: those hooks
  // return a fresh object literal every render, which would make any
  // useCallback depending on the whole object re-identity every render too
  // (and, worse, make the unmount-cleanup effect below re-fire on every
  // render instead of only on unmount). start/stop/enqueue/interrupt are
  // themselves stable (each hook memoizes them internally).
  const { enqueue: playbackEnqueue, interrupt: playbackInterrupt } = usePlayback();

  const handleInterrupt = useCallback(() => {
    const { fsmState, dispatch } = useSessionStore.getState();
    if (fsmState === VoiceState.SPEAKING || fsmState === VoiceState.THINKING) {
      useConnectionStore.getState().bumpPlaybackGeneration();
      playbackInterrupt();
      dispatch(VoiceEvent.INTERRUPT);
    }
    wsRef.current?.sendText({
      type: "control.interrupt",
      session_id: useSessionStore.getState().sessionId ?? "",
      ts: Date.now(),
      payload: { playback_generation: useConnectionStore.getState().playbackGeneration },
    });
  }, [playbackInterrupt]);

  const handleMicFrame = useCallback(
    (pcm: Int16Array) => {
      const { mode, fsmState } = useSessionStore.getState();
      if (mode === "ptt" && !pttHeldRef.current) return; // PTT-gated, mic not held

      // Local barge-in (design doc Barge-in sequence, step 1): optimistic,
      // immediate client-side reaction while Speaking, ahead of any server
      // round-trip.
      if (mode === "continuous" && fsmState === VoiceState.SPEAKING && isLikelySpeech(pcm)) {
        handleInterrupt();
      }

      const ws = wsRef.current;
      if (!ws?.isOpen) return;
      ws.sendBinary({
        msgType: MSG_TYPE_CLIENT_MIC_PCM,
        seq: seqRef.current++,
        clientTsMs: Date.now(),
        pcm: new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength),
      });
    },
    [handleInterrupt],
  );

  const { start: micStart, stop: micStop, error: micError, isCapturing } = useMicrophone(handleMicFrame);

  const sendSessionStart = useCallback((mode: SessionMode) => {
    seqRef.current = 0;
    useConnectionStore.getState().resetResumeAttempts();
    wsRef.current?.sendText({
      type: "session.start",
      session_id: "",
      ts: Date.now(),
      payload: { mode },
    });
  }, []);

  const sendSessionResume = useCallback((sessionId: string) => {
    wsRef.current?.sendText({
      type: "session.resume",
      session_id: "",
      ts: Date.now(),
      payload: { session_id: sessionId },
    });
  }, []);

  const handleTextFrame = useCallback(
    (frame: TextFrame) => {
      const session = useSessionStore.getState();
      const connection = useConnectionStore.getState();

      switch (frame.type) {
        case "session.hello": {
          const sessionId = frame.payload.session_id as string;
          session.setSessionId(sessionId);
          connection.resetResumeAttempts();
          break;
        }
        case "state.change": {
          const state = frame.payload.state as VoiceState;
          session.setFsmState(state);
          if (state !== VoiceState.RECOVERING) {
            connection.setStatus("connected");
          }
          break;
        }
        case "transcript.partial":
          session.setTranscriptPartial(frame.payload.text as string);
          break;
        case "transcript.final":
          session.setTranscriptFinal(frame.payload.text as string);
          break;
        case "llm.token":
          session.setLlmResponse(frame.payload.text as string);
          break;
        case "error": {
          const code = frame.payload.code as string;
          const message = frame.payload.message as string;
          const retriable = Boolean(frame.payload.retriable);
          session.setError({ code, message, retriable });
          if (code === "SESSION_NOT_FOUND") {
            // Matches the backend's design: stop retrying resume, fall
            // back to a fresh session.start (RESUME_FAILED -> Listening).
            session.dispatch(VoiceEvent.RESUME_FAILED);
            sendSessionStart(session.mode);
          }
          break;
        }
        case "pong":
        case "metrics.stage":
          break; // heartbeat ack / P0b dashboard concern -- no P0a-i UI yet
        default:
          break;
      }
    },
    [sendSessionStart],
  );

  // openConnection and scheduleReconnect are mutually referential (a close
  // schedules a reconnect, a reconnect opens a connection that can itself
  // close and reschedule). Routed through a ref -- kept current every
  // render below -- rather than a direct forward reference, so
  // scheduleReconnect can stay a stable, dependency-free callback without
  // an exhaustive-deps violation.
  const openConnectionRef = useRef<() => void>(() => {});

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current !== null) return;
    const delay = reconnectDelayMs(reconnectAttemptRef.current);
    reconnectAttemptRef.current += 1;
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      openConnectionRef.current();
    }, delay);
  }, []);

  const openConnection = useCallback(() => {
    const client = new WsClient(wsUrl(), {
      onOpen: () => {
        reconnectAttemptRef.current = 0;
        useConnectionStore.getState().setStatus("connected");
        const { sessionId } = useSessionStore.getState();
        const { resumeAttempts, incrementResumeAttempts } = useConnectionStore.getState();
        if (sessionId && resumeAttempts < MAX_RESUME_ATTEMPTS) {
          incrementResumeAttempts();
          sendSessionResume(sessionId);
        } else if (sessionId) {
          // Exhausted resume attempts -- fall back to fresh session.start.
          useSessionStore.getState().dispatch(VoiceEvent.RESUME_FAILED);
          sendSessionStart(useSessionStore.getState().mode);
        } else {
          sendSessionStart(useSessionStore.getState().mode);
        }
      },
      onClose: () => {
        if (intentionalCloseRef.current) return;
        const session = useSessionStore.getState();
        if (session.fsmState !== VoiceState.RECOVERING) {
          session.dispatch(VoiceEvent.WS_DROP);
        }
        useConnectionStore.getState().setStatus("recovering");
        scheduleReconnect();
      },
      onTextFrame: handleTextFrame,
      onBinaryFrame: (frame) => playbackEnqueue(frame),
      onInvalidFrame: (err) => {
        console.warn("Invalid frame from server:", err.message);
      },
    });
    wsRef.current = client;
    useConnectionStore.getState().setStatus("connecting");
    client.connect();
  }, [handleTextFrame, playbackEnqueue, scheduleReconnect, sendSessionResume, sendSessionStart]);

  openConnectionRef.current = openConnection;

  const connect = useCallback(
    (mode: SessionMode = "continuous") => {
      intentionalCloseRef.current = false;
      useSessionStore.getState().reset();
      useSessionStore.getState().setMode(mode);
      reconnectAttemptRef.current = 0;
      openConnection();
      void micStart();
    },
    [micStart, openConnection],
  );

  const disconnect = useCallback(() => {
    intentionalCloseRef.current = true;
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    micStop();
    playbackInterrupt();
    useConnectionStore.getState().setStatus("disconnected");
    useSessionStore.getState().reset();
  }, [micStop, playbackInterrupt]);

  const pttStart = useCallback(() => {
    pttHeldRef.current = true;
    wsRef.current?.sendText({
      type: "control.ptt_start",
      session_id: useSessionStore.getState().sessionId ?? "",
      ts: Date.now(),
      payload: {},
    });
  }, []);

  const pttEnd = useCallback(() => {
    pttHeldRef.current = false;
    wsRef.current?.sendText({
      type: "control.ptt_end",
      session_id: useSessionStore.getState().sessionId ?? "",
      ts: Date.now(),
      payload: {},
    });
  }, []);

  useEffect(() => disconnect, [disconnect]);

  return {
    connect,
    disconnect,
    interrupt: handleInterrupt,
    pttStart,
    pttEnd,
    micError,
    isCapturing,
  };
}
