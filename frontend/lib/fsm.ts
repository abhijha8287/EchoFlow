/**
 * Voice conversation finite state machine -- TS mirror of
 * backend/voice/fsm.py. Pure, dependency-free state table so the frontend
 * (Zustand store) and backend stay driven by the exact same transitions,
 * per the design doc's `### Finite state machine` (Core transitions table).
 */

export enum VoiceState {
  IDLE = "IDLE",
  LISTENING = "LISTENING",
  SPEECH_DETECTED = "SPEECH_DETECTED",
  THINKING = "THINKING",
  SPEAKING = "SPEAKING",
  INTERRUPTED = "INTERRUPTED",
  RECOVERING = "RECOVERING",
}

export enum VoiceEvent {
  SESSION_START = "SESSION_START",
  VAD_SPEECH = "VAD_SPEECH",
  VAD_ENDPOINT = "VAD_ENDPOINT",
  TTS_FIRST_CHUNK = "TTS_FIRST_CHUNK",
  CONTROL_CANCEL = "CONTROL_CANCEL",
  TURN_COMPLETE = "TURN_COMPLETE",
  INTERRUPT = "INTERRUPT",
  CANCEL_COMPLETE = "CANCEL_COMPLETE",
  WS_DROP = "WS_DROP",
  RESUME_OK = "RESUME_OK",
  /** SESSION_NOT_FOUND (or N failed resume attempts) -> Listening, fresh
   * session.start -- see design doc D8. Without this, unlimited resume
   * retries with no fallback could strand the UI in Recovering forever
   * after a backend restart. */
  RESUME_FAILED = "RESUME_FAILED",
  PROVIDER_ERROR = "PROVIDER_ERROR",
  EMPTY_TRANSCRIPT = "EMPTY_TRANSCRIPT",
}

export class InvalidTransitionError extends Error {
  constructor(event: VoiceEvent, state: VoiceState) {
    super(`no transition defined for event '${event}' from state '${state}'`);
    this.name = "InvalidTransitionError";
  }
}

type TransitionKey = `${VoiceState}:${VoiceEvent}`;

function key(state: VoiceState, event: VoiceEvent): TransitionKey {
  return `${state}:${event}`;
}

// Mirrors backend/voice/fsm.py's _TRANSITIONS dict row for row. WS_DROP is
// handled as a wildcard in transition() below rather than listed here,
// since it applies from every state, not one specific row.
const TRANSITIONS: Partial<Record<TransitionKey, VoiceState>> = {
  [key(VoiceState.IDLE, VoiceEvent.SESSION_START)]: VoiceState.LISTENING,
  [key(VoiceState.LISTENING, VoiceEvent.VAD_SPEECH)]: VoiceState.SPEECH_DETECTED,
  [key(VoiceState.SPEECH_DETECTED, VoiceEvent.VAD_ENDPOINT)]: VoiceState.THINKING,
  [key(VoiceState.THINKING, VoiceEvent.TTS_FIRST_CHUNK)]: VoiceState.SPEAKING,
  [key(VoiceState.THINKING, VoiceEvent.CONTROL_CANCEL)]: VoiceState.LISTENING,
  [key(VoiceState.SPEAKING, VoiceEvent.TURN_COMPLETE)]: VoiceState.LISTENING,
  [key(VoiceState.SPEAKING, VoiceEvent.INTERRUPT)]: VoiceState.INTERRUPTED,
  [key(VoiceState.THINKING, VoiceEvent.INTERRUPT)]: VoiceState.INTERRUPTED,
  [key(VoiceState.INTERRUPTED, VoiceEvent.CANCEL_COMPLETE)]: VoiceState.LISTENING,
  [key(VoiceState.RECOVERING, VoiceEvent.RESUME_OK)]: VoiceState.LISTENING,
  [key(VoiceState.RECOVERING, VoiceEvent.RESUME_FAILED)]: VoiceState.LISTENING,
  [key(VoiceState.THINKING, VoiceEvent.PROVIDER_ERROR)]: VoiceState.LISTENING,
  [key(VoiceState.SPEAKING, VoiceEvent.PROVIDER_ERROR)]: VoiceState.LISTENING,
  [key(VoiceState.SPEECH_DETECTED, VoiceEvent.EMPTY_TRANSCRIPT)]: VoiceState.LISTENING,
};

/**
 * Apply `event` to `state`, returning the new state. `WS_DROP` is a
 * wildcard -- it transitions to RECOVERING from any state, per the design
 * doc's `* | WS drop | Recovering` row. Every other event must have an
 * explicit transition defined for the current state, or this throws
 * `InvalidTransitionError` rather than silently no-op'ing.
 */
export function transition(state: VoiceState, event: VoiceEvent): VoiceState {
  if (event === VoiceEvent.WS_DROP) {
    return VoiceState.RECOVERING;
  }
  const next = TRANSITIONS[key(state, event)];
  if (next === undefined) {
    throw new InvalidTransitionError(event, state);
  }
  return next;
}
