"""Gemini TTS adapter tests — mocked google-genai client, no real network
calls (matching the design doc's contract-test philosophy).
"""

from __future__ import annotations

import asyncio
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors

from stt.provider import RateLimitedError
from tts.gemini import create_gemini_tts


def _constant_pcm(value: int, num_samples: int) -> bytes:
    return array("h", [value] * num_samples).tobytes()


def _pcm_to_samples(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm)
    return samples


def _chunk_with_audio(pcm_bytes: bytes):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=pcm_bytes))
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def _chunk_with_no_content():
    """Mirrors the one observed real response shape where a candidate's
    `content` was None -- must be skipped, not crash."""
    return SimpleNamespace(candidates=[SimpleNamespace(content=None)])


async def _async_iter(items):
    for item in items:
        yield item


def _fake_client(chunks) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=_async_iter(chunks))
    return client


def test_resamples_native_24k_output_to_wire_16k():
    asyncio.run(_resamples_native_24k_output_to_wire_16k())


async def _resamples_native_24k_output_to_wire_16k():
    # Constant-amplitude signal: resampling must reproduce the same
    # constant at any rate (see test_audio.py's identical reasoning), so
    # this pins the expected output without duplicating the interpolation
    # formula inside the test.
    native_pcm = _constant_pcm(1000, 2400)  # 100ms @ 24kHz
    client = _fake_client([_chunk_with_audio(native_pcm)])
    gemini_tts = create_gemini_tts(client, model="gemini-2.5-flash-preview-tts")

    result = await gemini_tts("utt-1", "Hello there.")

    samples = _pcm_to_samples(result)
    assert len(samples) == 1600  # 2400 * 16000/24000
    assert all(s == 1000 for s in samples)


def test_prompt_is_framed_as_a_say_instruction():
    asyncio.run(_prompt_is_framed_as_a_say_instruction())


async def _prompt_is_framed_as_a_say_instruction():
    """The exact real-API bug the spike found: a plain sentence gets
    treated as a chat prompt and rejected with INVALID_ARGUMENT once
    response_modalities=["AUDIO"] is set."""
    client = _fake_client([_chunk_with_audio(_constant_pcm(0, 10))])
    gemini_tts = create_gemini_tts(client)

    await gemini_tts("utt-1", "Hello there.")

    call_kwargs = client.aio.models.generate_content_stream.call_args.kwargs
    assert call_kwargs["contents"] == "Say: Hello there."
    assert call_kwargs["config"].response_modalities == ["AUDIO"]


def test_concatenates_multiple_chunks_before_resampling():
    asyncio.run(_concatenates_multiple_chunks_before_resampling())


async def _concatenates_multiple_chunks_before_resampling():
    chunk_a = _constant_pcm(500, 240)
    chunk_b = _constant_pcm(500, 240)
    client = _fake_client([_chunk_with_audio(chunk_a), _chunk_with_audio(chunk_b)])
    gemini_tts = create_gemini_tts(client)

    result = await gemini_tts("utt-1", "hi")

    samples = _pcm_to_samples(result)
    assert len(samples) == 320  # (240 + 240) * 16000/24000
    assert all(s == 500 for s in samples)


def test_skips_a_chunk_whose_candidate_has_no_content():
    asyncio.run(_skips_a_chunk_whose_candidate_has_no_content())


async def _skips_a_chunk_whose_candidate_has_no_content():
    native_pcm = _constant_pcm(777, 240)
    client = _fake_client([_chunk_with_no_content(), _chunk_with_audio(native_pcm)])
    gemini_tts = create_gemini_tts(client)

    result = await gemini_tts("utt-1", "hi")  # must not raise

    samples = _pcm_to_samples(result)
    assert len(samples) == 160
    assert all(s == 777 for s in samples)


def test_maps_429_to_rate_limited_error():
    asyncio.run(_maps_429_to_rate_limited_error())


async def _maps_429_to_rate_limited_error():
    client = MagicMock()
    api_error = errors.ClientError(429, {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    client.aio.models.generate_content_stream = AsyncMock(side_effect=api_error)
    gemini_tts = create_gemini_tts(client)

    with pytest.raises(RateLimitedError):
        await gemini_tts("utt-1", "hi")


def test_propagates_non_rate_limit_api_errors():
    asyncio.run(_propagates_non_rate_limit_api_errors())


async def _propagates_non_rate_limit_api_errors():
    client = MagicMock()
    api_error = errors.ServerError(500, {"error": {"message": "internal", "status": "INTERNAL"}})
    client.aio.models.generate_content_stream = AsyncMock(side_effect=api_error)
    gemini_tts = create_gemini_tts(client)

    with pytest.raises(errors.ServerError):
        await gemini_tts("utt-1", "hi")
