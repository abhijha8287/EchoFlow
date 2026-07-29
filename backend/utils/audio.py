"""Shared audio resample utility — mirrors design doc `### Audio format
normalization` (T6).

Wire audio is always PCM s16le mono 16 kHz. Decoding each TTS provider's
native wire format (Opus, MP3, provider-specific container, ...) into raw
PCM is genuinely vendor-specific and lives in each `TTSProvider` adapter
under `backend/tts/`. The *resample* step below it -- native rate/channels
-> s16le/16k -- is not vendor-specific and must not be duplicated per
adapter, so it lives here as the one shared implementation every adapter
calls after its own decode step.

Pure Python, no new dependency: `audioop` (the stdlib module this would
traditionally use) was removed in Python 3.13, and the sample buffers
involved here (single TTS utterances, not hours of audio) don't justify
pulling in numpy/scipy for a portfolio-scale project. Linear interpolation
is sufficient quality for spoken-word TTS at these rates; if profiling ever
shows this as a bottleneck, swapping the inner loop for a vectorized
implementation would be a drop-in change -- the function signature would not
need to move.
"""

from __future__ import annotations

import io
import wave
from array import array


def wrap_pcm_as_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a minimal WAV (RIFF) container.

    Some provider APIs (e.g. Gemini's `generateContent` with an inline
    audio `Part`) accept `audio/wav` but not bare `audio/pcm` for
    transcription input -- this is the shared version of what would
    otherwise be duplicated between the STT adapter and
    `backend/spike/gemini_spike.py`'s STT probe.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # s16le
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def normalize_audio(pcm: bytes, native_rate: int, native_channels: int = 1, target_rate: int = 16000) -> bytes:
    """Convert `pcm` (s16le, `native_channels` channels, `native_rate` Hz)
    to s16le mono `target_rate` Hz, per the wire format.

    `pcm` is assumed already s16le -- the vendor-specific decode step in
    each adapter is responsible for getting there from whatever the
    provider's native encoding is; this function only ever resamples/
    downmixes.
    """
    samples = _bytes_to_int16_array(pcm)

    if native_channels > 1:
        samples = _downmix_to_mono(samples, native_channels)

    if native_rate != target_rate:
        samples = _resample_linear(samples, native_rate, target_rate)

    return samples.tobytes()


def _bytes_to_int16_array(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm)
    return samples


def _downmix_to_mono(samples: array, channels: int) -> array:
    mono = array("h", bytes(2 * (len(samples) // channels)))
    for frame in range(len(samples) // channels):
        start = frame * channels
        total = sum(samples[start : start + channels])
        mono[frame] = _clamp_int16(round(total / channels))
    return mono


def _resample_linear(samples: array, src_rate: int, dst_rate: int) -> array:
    src_len = len(samples)
    if src_len == 0:
        return array("h")
    if src_len == 1:
        return array("h", [samples[0]])

    dst_len = max(1, round(src_len * dst_rate / src_rate))
    out = array("h", bytes(2 * dst_len))
    step = (src_len - 1) / (dst_len - 1) if dst_len > 1 else 0.0

    for i in range(dst_len):
        pos = i * step
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < src_len:
            interpolated = samples[idx] + (samples[idx + 1] - samples[idx]) * frac
        else:
            interpolated = samples[idx]
        out[i] = _clamp_int16(round(interpolated))

    return out


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, value))
