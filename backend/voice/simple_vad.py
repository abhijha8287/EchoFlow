"""Crude energy-threshold `VadProvider` -- NOT Silero.

Real Silero VAD integration is separate, larger scope (see `vad.py`'s
module docstring). This exists so `backend/main.py` has something that
actually implements the protocol end to end, so the server is genuinely
runnable for a live demo in the meantime rather than blocked entirely on
VAD model work. Swapping in a real Silero-backed `VadProvider` later
touches only `main.py`'s `vad_factory` wiring -- `SessionRuntime` only
depends on the `VadProvider` Protocol, never this class.
"""

from __future__ import annotations

from array import array

from .vad import VadEvent

# Mean absolute amplitude (0-32767) above which a frame counts as "speech".
DEFAULT_SPEECH_THRESHOLD = 1500
# Consecutive below-threshold frames before declaring the endpoint. At the
# wire format's 20ms/frame, 25 frames ~= 500ms, matching the design doc's
# "silence threshold configurable, default ~400-600ms" for Silero.
DEFAULT_SILENCE_FRAMES_FOR_ENDPOINT = 25


class SimpleEnergyVad:
    def __init__(
        self,
        speech_threshold: int = DEFAULT_SPEECH_THRESHOLD,
        silence_frames_for_endpoint: int = DEFAULT_SILENCE_FRAMES_FOR_ENDPOINT,
    ) -> None:
        self._speech_threshold = speech_threshold
        self._silence_frames_for_endpoint = silence_frames_for_endpoint
        self._in_speech = False
        self._consecutive_silence_frames = 0

    def process(self, pcm_chunk: bytes) -> VadEvent | None:
        samples = array("h")
        samples.frombytes(pcm_chunk)
        if not samples:
            return None

        is_loud = (sum(abs(s) for s in samples) / len(samples)) > self._speech_threshold

        if not self._in_speech:
            if is_loud:
                self._in_speech = True
                self._consecutive_silence_frames = 0
                return VadEvent.SPEECH
            return None

        if is_loud:
            self._consecutive_silence_frames = 0
            return None

        self._consecutive_silence_frames += 1
        if self._consecutive_silence_frames >= self._silence_frames_for_endpoint:
            self._in_speech = False
            return VadEvent.ENDPOINT
        return None

    def reset(self) -> None:
        self._in_speech = False
        self._consecutive_silence_frames = 0
