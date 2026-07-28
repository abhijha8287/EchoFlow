"""FSM tests — table-driven against design doc `### Finite state machine` (T2)."""

import pytest

from voice.fsm import InvalidTransitionError, VoiceEvent, VoiceFSM, VoiceState

# Mirrors the design doc's "Core transitions" table row for row, including the
# RESUME_FAILED row added during /plan-eng-review (D8). WS_DROP is tested
# separately below since it's a wildcard, not one row.
TRANSITION_TABLE = [
    (VoiceState.IDLE, VoiceEvent.SESSION_START, VoiceState.LISTENING),
    (VoiceState.LISTENING, VoiceEvent.VAD_SPEECH, VoiceState.SPEECH_DETECTED),
    (VoiceState.SPEECH_DETECTED, VoiceEvent.VAD_ENDPOINT, VoiceState.THINKING),
    (VoiceState.THINKING, VoiceEvent.TTS_FIRST_CHUNK, VoiceState.SPEAKING),
    (VoiceState.THINKING, VoiceEvent.CONTROL_CANCEL, VoiceState.LISTENING),
    (VoiceState.SPEAKING, VoiceEvent.TURN_COMPLETE, VoiceState.LISTENING),
    (VoiceState.SPEAKING, VoiceEvent.INTERRUPT, VoiceState.INTERRUPTED),
    (VoiceState.THINKING, VoiceEvent.INTERRUPT, VoiceState.INTERRUPTED),
    (VoiceState.INTERRUPTED, VoiceEvent.CANCEL_COMPLETE, VoiceState.LISTENING),
    (VoiceState.RECOVERING, VoiceEvent.RESUME_OK, VoiceState.LISTENING),
    (VoiceState.RECOVERING, VoiceEvent.RESUME_FAILED, VoiceState.LISTENING),
    (VoiceState.THINKING, VoiceEvent.PROVIDER_ERROR, VoiceState.LISTENING),
    (VoiceState.SPEAKING, VoiceEvent.PROVIDER_ERROR, VoiceState.LISTENING),
    (VoiceState.SPEECH_DETECTED, VoiceEvent.EMPTY_TRANSCRIPT, VoiceState.LISTENING),
]


@pytest.mark.parametrize("from_state,event,to_state", TRANSITION_TABLE)
def test_transition_table(from_state, event, to_state):
    fsm = VoiceFSM(initial=from_state)
    assert fsm.handle(event) == to_state
    assert fsm.state == to_state


def test_resume_failed_returns_to_listening_not_stuck_in_recovering():
    """The bug this guards against (D8): unlimited resume retries with no
    fallback could strand the client in RECOVERING forever after a backend
    restart. RESUME_FAILED must land back in LISTENING, matching the caller's
    responsibility to send a fresh session.start."""
    fsm = VoiceFSM(initial=VoiceState.RECOVERING)
    assert fsm.handle(VoiceEvent.RESUME_FAILED) == VoiceState.LISTENING


@pytest.mark.parametrize(
    "from_state",
    [
        VoiceState.IDLE,
        VoiceState.LISTENING,
        VoiceState.SPEECH_DETECTED,
        VoiceState.THINKING,
        VoiceState.SPEAKING,
        VoiceState.INTERRUPTED,
        VoiceState.RECOVERING,
    ],
)
def test_ws_drop_is_a_wildcard_from_any_state(from_state):
    fsm = VoiceFSM(initial=from_state)
    assert fsm.handle(VoiceEvent.WS_DROP) == VoiceState.RECOVERING


def test_undefined_transition_raises_instead_of_silently_ignoring():
    fsm = VoiceFSM(initial=VoiceState.IDLE)
    with pytest.raises(InvalidTransitionError):
        fsm.handle(VoiceEvent.VAD_ENDPOINT)  # no such transition from IDLE


def test_default_initial_state_is_idle():
    assert VoiceFSM().state == VoiceState.IDLE
