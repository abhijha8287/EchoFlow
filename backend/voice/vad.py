"""VAD (speech / endpoint detection) seam.

Real Silero VAD model integration is separate, larger scope than the T1-T7
fix list (see design doc's core P0a-i loop / Provider spike) -- this only
defines the interface `SessionRuntime` depends on, the same pattern already
used for `STTProvider`/LLM/TTS: the orchestrator is written and tested
against this Protocol, and a real Silero-backed implementation is a later,
separate piece of work that plugs in without changing the orchestrator.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol


class VadEvent(Enum):
    SPEECH = auto()
    ENDPOINT = auto()


class VadProvider(Protocol):
    def process(self, pcm_chunk: bytes) -> VadEvent | None:
        """Feed one inbound mic PCM chunk. Returns a `VadEvent` if this
        chunk crosses a detection boundary, else `None`. Stateful across
        calls for one utterance; callers reset detection state between
        utterances via `reset()`."""
        ...

    def reset(self) -> None:
        """Clear any per-utterance detection state (called once a new
        utterance starts, i.e. right after a SPEECH event is consumed)."""
        ...
