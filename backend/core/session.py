"""In-memory session store — mirrors the design doc's P0-P1 persistence
ladder: sessions are in-memory only and are wiped on process restart, but
they are *not* tied to a single WebSocket connection's lifetime. A
`session.resume` can reattach a new connection to the same `session_id`
while the process is still up (see Reconnect & heartbeat), which is what
makes a clean reconnect after a mid-Speaking drop possible without losing
state -- only a process restart (or `SESSION_NOT_FOUND`) loses it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from voice.fsm import VoiceFSM

VALID_MODES = ("continuous", "ptt")


@dataclass(slots=True)
class Session:
    session_id: str
    mode: str = "continuous"
    fsm: VoiceFSM = field(default_factory=VoiceFSM)
    playback_generation: int = 0
    current_utterance_id: str | None = None
    outbound_seq: int = 0


class SessionStore:
    """One instance per process. Not thread-safe by design -- the server is
    a single-worker asyncio process (see design doc Compose topology)."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, mode: str = "continuous") -> Session:
        if mode not in VALID_MODES:
            mode = "continuous"
        session = Session(session_id=uuid.uuid4().hex, mode=mode)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
