"""Voice conversation finite state machine — mirrors design doc `### Finite state machine`.

Pure, dependency-free state table so the same transition logic can be shared
(or ported) between the FastAPI backend and the Zustand client mirror without
drift between the two.
"""

from __future__ import annotations

from enum import Enum, auto


class VoiceState(Enum):
    IDLE = auto()
    LISTENING = auto()
    SPEECH_DETECTED = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()
    RECOVERING = auto()


class VoiceEvent(Enum):
    SESSION_START = auto()  # "session.start / mic ready"
    VAD_SPEECH = auto()  # "server VAD speech"
    VAD_ENDPOINT = auto()  # "VAD endpoint"
    TTS_FIRST_CHUNK = auto()  # "first TTS PCM enqueued to WS send path"
    CONTROL_CANCEL = auto()  # "control.cancel"
    TURN_COMPLETE = auto()  # "LLM+TTS complete + drain"
    INTERRUPT = auto()  # "control.interrupt / local barge-in"
    CANCEL_COMPLETE = auto()  # "cancel complete"
    WS_DROP = auto()  # "WS drop" — wildcard, applies from any state
    RESUME_OK = auto()  # "session.resume ok"
    RESUME_FAILED = auto()  # "session.resume fails with SESSION_NOT_FOUND ..." (new — D8)
    PROVIDER_ERROR = auto()  # "provider error"
    EMPTY_TRANSCRIPT = auto()  # "empty transcript after promote"


class InvalidTransitionError(ValueError):
    """Raised when an event has no defined transition from the current state."""


# Mirrors the design doc's "Core transitions" table row for row. WS_DROP is
# handled as a wildcard in VoiceFSM.handle() below rather than listed here,
# since it applies from every state, not one specific row.
_TRANSITIONS: dict[tuple[VoiceState, VoiceEvent], VoiceState] = {
    (VoiceState.IDLE, VoiceEvent.SESSION_START): VoiceState.LISTENING,
    (VoiceState.LISTENING, VoiceEvent.VAD_SPEECH): VoiceState.SPEECH_DETECTED,
    (VoiceState.SPEECH_DETECTED, VoiceEvent.VAD_ENDPOINT): VoiceState.THINKING,
    (VoiceState.THINKING, VoiceEvent.TTS_FIRST_CHUNK): VoiceState.SPEAKING,
    (VoiceState.THINKING, VoiceEvent.CONTROL_CANCEL): VoiceState.LISTENING,
    (VoiceState.SPEAKING, VoiceEvent.TURN_COMPLETE): VoiceState.LISTENING,
    (VoiceState.SPEAKING, VoiceEvent.INTERRUPT): VoiceState.INTERRUPTED,
    (VoiceState.THINKING, VoiceEvent.INTERRUPT): VoiceState.INTERRUPTED,
    (VoiceState.INTERRUPTED, VoiceEvent.CANCEL_COMPLETE): VoiceState.LISTENING,
    (VoiceState.RECOVERING, VoiceEvent.RESUME_OK): VoiceState.LISTENING,
    # New during /plan-eng-review (D8): without this, unlimited resume retries
    # with no fallback could strand the client in RECOVERING forever after a
    # backend restart (session_id no longer known to the server). Caller is
    # responsible for sending a fresh session.start and surfacing a toast that
    # conversation history was lost — the FSM only tracks the state transition.
    (VoiceState.RECOVERING, VoiceEvent.RESUME_FAILED): VoiceState.LISTENING,
    (VoiceState.THINKING, VoiceEvent.PROVIDER_ERROR): VoiceState.LISTENING,
    (VoiceState.SPEAKING, VoiceEvent.PROVIDER_ERROR): VoiceState.LISTENING,
    (VoiceState.SPEECH_DETECTED, VoiceEvent.EMPTY_TRANSCRIPT): VoiceState.LISTENING,
}


class VoiceFSM:
    """Server/client-mirrored voice conversation state machine."""

    def __init__(self, initial: VoiceState = VoiceState.IDLE) -> None:
        self._state = initial

    @property
    def state(self) -> VoiceState:
        return self._state

    def handle(self, event: VoiceEvent) -> VoiceState:
        """Apply `event` to the current state, returning the new state.

        `WS_DROP` is a wildcard — it transitions to RECOVERING from any state,
        per the design doc's `* | WS drop | Recovering` row. Every other event
        must have an explicit transition defined for the current state, or this
        raises `InvalidTransitionError` rather than silently no-op'ing — an
        unhandled event is a bug in the caller, not a state the FSM should mask.
        """
        if event is VoiceEvent.WS_DROP:
            self._state = VoiceState.RECOVERING
            return self._state

        key = (self._state, event)
        try:
            self._state = _TRANSITIONS[key]
        except KeyError as exc:
            raise InvalidTransitionError(
                f"no transition defined for event {event.name!r} from state {self._state.name!r}"
            ) from exc
        return self._state
