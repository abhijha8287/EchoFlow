"""SimpleEnergyVad tests — see backend/voice/simple_vad.py. Crude
energy-threshold detector, not Silero (real Silero integration is
separate, later work) -- these tests only need to confirm the state
machine (silence -> speech -> endpoint) behaves, not audio-quality claims.
"""

from __future__ import annotations

from array import array

from voice.simple_vad import SimpleEnergyVad
from voice.vad import VadEvent


def _frame(value: int, num_samples: int = 320) -> bytes:
    return array("h", [value] * num_samples).tobytes()


def test_loud_frame_from_silence_fires_speech():
    vad = SimpleEnergyVad(speech_threshold=1000)
    assert vad.process(_frame(5000)) is VadEvent.SPEECH


def test_quiet_frame_from_silence_fires_nothing():
    vad = SimpleEnergyVad(speech_threshold=1000)
    assert vad.process(_frame(100)) is None


def test_quiet_frames_after_speech_do_not_immediately_fire_endpoint():
    vad = SimpleEnergyVad(speech_threshold=1000, silence_frames_for_endpoint=3)
    assert vad.process(_frame(5000)) is VadEvent.SPEECH
    assert vad.process(_frame(0)) is None
    assert vad.process(_frame(0)) is None


def test_endpoint_fires_after_enough_consecutive_silent_frames():
    vad = SimpleEnergyVad(speech_threshold=1000, silence_frames_for_endpoint=3)
    assert vad.process(_frame(5000)) is VadEvent.SPEECH
    assert vad.process(_frame(0)) is None
    assert vad.process(_frame(0)) is None
    assert vad.process(_frame(0)) is VadEvent.ENDPOINT


def test_loud_frame_resets_the_silence_counter():
    vad = SimpleEnergyVad(speech_threshold=1000, silence_frames_for_endpoint=3)
    assert vad.process(_frame(5000)) is VadEvent.SPEECH
    assert vad.process(_frame(0)) is None
    assert vad.process(_frame(0)) is None
    assert vad.process(_frame(5000)) is None  # loud again -- resets the count
    assert vad.process(_frame(0)) is None
    assert vad.process(_frame(0)) is None
    assert vad.process(_frame(0)) is VadEvent.ENDPOINT  # needed 3 fresh silent frames


def test_reset_clears_in_speech_state():
    vad = SimpleEnergyVad(speech_threshold=1000, silence_frames_for_endpoint=3)
    assert vad.process(_frame(5000)) is VadEvent.SPEECH
    vad.reset()
    # Back to "not in speech" -- another loud frame fires SPEECH again,
    # not silently continuing the old utterance.
    assert vad.process(_frame(5000)) is VadEvent.SPEECH


def test_empty_frame_is_ignored():
    vad = SimpleEnergyVad()
    assert vad.process(b"") is None
