"""Audio resample correctness tests — see `backend/utils/audio.py` and
design doc `### Audio format normalization` (T6).

Correctness is checked against a constant-amplitude signal rather than an
arbitrary waveform: resampling (linear interpolation) or downmixing
(averaging) a constant value must reproduce that same constant everywhere
in the output, regardless of rate -- this is true for any faithful
resampler, so it pins expected output exactly without duplicating the
implementation's own interpolation formula inside the test.
"""

from __future__ import annotations

from array import array

import pytest

from utils.audio import normalize_audio

# Representative native TTS rates -- exact spike-measured values (per the
# design doc's Provider spike) aren't known yet, so this covers the common
# range real providers use (8kHz telephony-style up to 44.1kHz).
NATIVE_RATES = [8000, 16000, 22050, 24000, 44100]


def _constant_pcm(value: int, num_samples: int) -> bytes:
    return array("h", [value] * num_samples).tobytes()


def _pcm_to_samples(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm)
    return samples


@pytest.mark.parametrize("native_rate", NATIVE_RATES)
def test_constant_signal_resamples_to_expected_length_and_value(native_rate):
    num_samples = 480  # 30ms @ 16kHz-ish source length, arbitrary but nontrivial
    value = 12345
    pcm = _constant_pcm(value, num_samples)

    out = normalize_audio(pcm, native_rate=native_rate, target_rate=16000)
    out_samples = _pcm_to_samples(out)

    expected_len = round(num_samples * 16000 / native_rate)
    assert len(out_samples) == expected_len
    assert all(sample == value for sample in out_samples)


def test_identity_when_native_rate_already_matches_target():
    pcm = _constant_pcm(-777, 320)
    out = normalize_audio(pcm, native_rate=16000, target_rate=16000)
    assert out == pcm


def test_downmix_stereo_to_mono_averages_channels():
    left, right = 1000, 3000
    interleaved = array("h", [left, right] * 200).tobytes()

    out = normalize_audio(interleaved, native_rate=16000, native_channels=2, target_rate=16000)
    out_samples = _pcm_to_samples(out)

    assert len(out_samples) == 200
    assert all(sample == (left + right) // 2 for sample in out_samples)


def test_downmix_and_resample_combined():
    left, right = 1000, 3000
    num_frames = 240
    interleaved = array("h", [left, right] * num_frames).tobytes()

    out = normalize_audio(interleaved, native_rate=24000, native_channels=2, target_rate=16000)
    out_samples = _pcm_to_samples(out)

    expected_len = round(num_frames * 16000 / 24000)
    assert len(out_samples) == expected_len
    assert all(sample == (left + right) // 2 for sample in out_samples)


def test_extreme_amplitude_values_do_not_overflow_int16():
    pcm = _constant_pcm(32767, 100)
    out = normalize_audio(pcm, native_rate=8000, target_rate=16000)
    out_samples = _pcm_to_samples(out)
    assert all(sample == 32767 for sample in out_samples)

    pcm_min = _constant_pcm(-32768, 100)
    out_min = normalize_audio(pcm_min, native_rate=8000, target_rate=16000)
    out_min_samples = _pcm_to_samples(out_min)
    assert all(sample == -32768 for sample in out_min_samples)


def test_empty_input_returns_empty_output():
    assert normalize_audio(b"", native_rate=24000, target_rate=16000) == b""
