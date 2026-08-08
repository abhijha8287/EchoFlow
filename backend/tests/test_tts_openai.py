"""OpenAI TTS adapter tests — mocked openai client, no real network calls
(matching the design doc's contract-test philosophy).
"""

from __future__ import annotations

import asyncio
from array import array
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from stt.provider import RateLimitedError
from tts.openai import create_openai_tts


def _constant_pcm(value: int, num_samples: int) -> bytes:
    return array("h", [value] * num_samples).tobytes()


def _pcm_to_samples(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm)
    return samples


def _fake_client(pcm_bytes: bytes) -> MagicMock:
    client = MagicMock()
    api_response = MagicMock()
    api_response.aread = AsyncMock(return_value=pcm_bytes)
    client.audio.speech.create = AsyncMock(return_value=api_response)
    return client


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"))
    return openai.RateLimitError("quota exceeded", response=response, body=None)


def _server_error() -> openai.InternalServerError:
    response = httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"))
    return openai.InternalServerError("internal", response=response, body=None)


def test_resamples_native_24k_output_to_wire_16k():
    asyncio.run(_resamples_native_24k_output_to_wire_16k())


async def _resamples_native_24k_output_to_wire_16k():
    # Constant-amplitude signal: resampling must reproduce the same
    # constant at any rate (see test_audio.py's identical reasoning), so
    # this pins the expected output without duplicating the interpolation
    # formula inside the test.
    native_pcm = _constant_pcm(1000, 2400)  # 100ms @ 24kHz
    client = _fake_client(native_pcm)
    openai_tts = create_openai_tts(client, model="gpt-4o-mini-tts")

    result = await openai_tts("utt-1", "Hello there.")

    samples = _pcm_to_samples(result)
    assert len(samples) == 1600  # 2400 * 16000/24000
    assert all(s == 1000 for s in samples)


def test_calls_speech_api_with_pcm_response_format():
    asyncio.run(_calls_speech_api_with_pcm_response_format())


async def _calls_speech_api_with_pcm_response_format():
    client = _fake_client(_constant_pcm(0, 10))
    openai_tts = create_openai_tts(client, model="gpt-4o-mini-tts", voice="alloy")

    await openai_tts("utt-1", "Hello there.")

    call_kwargs = client.audio.speech.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini-tts"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["input"] == "Hello there."
    assert call_kwargs["response_format"] == "pcm"


def test_maps_429_to_rate_limited_error():
    asyncio.run(_maps_429_to_rate_limited_error())


async def _maps_429_to_rate_limited_error():
    client = MagicMock()
    client.audio.speech.create = AsyncMock(side_effect=_rate_limit_error())
    openai_tts = create_openai_tts(client)

    with pytest.raises(RateLimitedError):
        await openai_tts("utt-1", "hi")


def test_propagates_non_rate_limit_api_errors():
    asyncio.run(_propagates_non_rate_limit_api_errors())


async def _propagates_non_rate_limit_api_errors():
    client = MagicMock()
    client.audio.speech.create = AsyncMock(side_effect=_server_error())
    openai_tts = create_openai_tts(client)

    with pytest.raises(openai.InternalServerError):
        await openai_tts("utt-1", "hi")
