/**
 * FSM tests -- TS mirror of backend/tests/test_fsm.py. Table-driven against
 * the same design doc `### Finite state machine` Core transitions table.
 */

import { describe, expect, it } from "vitest";
import { InvalidTransitionError, VoiceEvent, VoiceState, transition } from "./fsm";

// Mirrors backend/tests/test_fsm.py's TRANSITION_TABLE row for row,
// including the RESUME_FAILED row added during /plan-eng-review (D8).
// WS_DROP is tested separately below since it's a wildcard, not one row.
const TRANSITION_TABLE: Array<[VoiceState, VoiceEvent, VoiceState]> = [
  [VoiceState.IDLE, VoiceEvent.SESSION_START, VoiceState.LISTENING],
  [VoiceState.LISTENING, VoiceEvent.VAD_SPEECH, VoiceState.SPEECH_DETECTED],
  [VoiceState.SPEECH_DETECTED, VoiceEvent.VAD_ENDPOINT, VoiceState.THINKING],
  [VoiceState.THINKING, VoiceEvent.TTS_FIRST_CHUNK, VoiceState.SPEAKING],
  [VoiceState.THINKING, VoiceEvent.CONTROL_CANCEL, VoiceState.LISTENING],
  [VoiceState.SPEAKING, VoiceEvent.TURN_COMPLETE, VoiceState.LISTENING],
  [VoiceState.SPEAKING, VoiceEvent.INTERRUPT, VoiceState.INTERRUPTED],
  [VoiceState.THINKING, VoiceEvent.INTERRUPT, VoiceState.INTERRUPTED],
  [VoiceState.INTERRUPTED, VoiceEvent.CANCEL_COMPLETE, VoiceState.LISTENING],
  [VoiceState.RECOVERING, VoiceEvent.RESUME_OK, VoiceState.LISTENING],
  [VoiceState.RECOVERING, VoiceEvent.RESUME_FAILED, VoiceState.LISTENING],
  [VoiceState.THINKING, VoiceEvent.PROVIDER_ERROR, VoiceState.LISTENING],
  [VoiceState.SPEAKING, VoiceEvent.PROVIDER_ERROR, VoiceState.LISTENING],
  [VoiceState.SPEECH_DETECTED, VoiceEvent.EMPTY_TRANSCRIPT, VoiceState.LISTENING],
];

describe("transition table", () => {
  it.each(TRANSITION_TABLE)("%s + %s -> %s", (from, event, to) => {
    expect(transition(from, event)).toBe(to);
  });
});

it("RESUME_FAILED returns to Listening, not stuck in Recovering", () => {
  expect(transition(VoiceState.RECOVERING, VoiceEvent.RESUME_FAILED)).toBe(VoiceState.LISTENING);
});

describe("WS_DROP wildcard", () => {
  const allStates = Object.values(VoiceState);
  it.each(allStates)("from %s -> Recovering", (state) => {
    expect(transition(state, VoiceEvent.WS_DROP)).toBe(VoiceState.RECOVERING);
  });
});

it("throws InvalidTransitionError instead of silently ignoring an undefined transition", () => {
  expect(() => transition(VoiceState.IDLE, VoiceEvent.VAD_ENDPOINT)).toThrow(InvalidTransitionError);
});
