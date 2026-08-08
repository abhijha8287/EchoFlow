"""OpenAI STT adapter tests — mocked openai client, no real network calls
(matching the design doc's contract-test philosophy: mock the vendor's API
shape, not the live endpoint).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from stt.openai import OpenAISTTProvider
from stt.provider import RateLimitedError

# 100ms @ 16kHz s16le mono -- the confirmed real-endpoint minimum duration
# (see test_transcribe_chunk_skips_the_api_call_when_audio_is_too_short).
# Tests exercising an actual API call need at least this much pcm.
_MIN_DURATION_PCM = b"\x01\x02" * 1600


def _fake_client(response_text: str | None = "hello world") -> MagicMock:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(return_value=SimpleNamespace(text=response_text))
    return client


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions"))
    return openai.RateLimitError("quota exceeded", response=response, body=None)


def _server_error() -> openai.InternalServerError:
    response = httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions"))
    return openai.InternalServerError("internal", response=response, body=None)


def test_supports_partial_streaming_is_false():
    """`transcriptions.create` has no notion of resuming a prior partial
    transcription -- the adapter must use the chunked-upload path (T4)."""
    assert OpenAISTTProvider(_fake_client()).supports_partial_streaming is False


def test_transcribe_chunk_returns_stripped_response_text():
    asyncio.run(_transcribe_chunk_returns_stripped_response_text())


async def _transcribe_chunk_returns_stripped_response_text():
    client = _fake_client(response_text="  hello world  \n")
    provider = OpenAISTTProvider(client, model="gpt-4o-transcribe")

    result = await provider.transcribe_chunk("utt-1", 0, _MIN_DURATION_PCM)

    assert result == "hello world"
    client.audio.transcriptions.create.assert_awaited_once()
    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-transcribe"
    assert call_kwargs["file"].name == "utterance.wav"


def test_transcribe_chunk_wraps_pcm_as_a_valid_wav_container():
    asyncio.run(_transcribe_chunk_wraps_pcm_as_a_valid_wav_container())


async def _transcribe_chunk_wraps_pcm_as_a_valid_wav_container():
    import wave
    from io import BytesIO

    client = _fake_client()
    provider = OpenAISTTProvider(client)
    pcm = _MIN_DURATION_PCM

    await provider.transcribe_chunk("utt-1", 0, pcm)

    audio_file = client.audio.transcriptions.create.call_args.kwargs["file"]
    with wave.open(BytesIO(audio_file.getvalue()), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(wav_file.getnframes()) == pcm


def test_transcribe_chunk_with_empty_pcm_skips_the_api_call():
    asyncio.run(_transcribe_chunk_with_empty_pcm_skips_the_api_call())


async def _transcribe_chunk_with_empty_pcm_skips_the_api_call():
    client = _fake_client()
    provider = OpenAISTTProvider(client)

    result = await provider.transcribe_chunk("utt-1", 0, b"")

    assert result == ""
    client.audio.transcriptions.create.assert_not_awaited()


def test_transcribe_chunk_skips_the_api_call_when_audio_is_too_short():
    asyncio.run(_transcribe_chunk_skips_the_api_call_when_audio_is_too_short())


async def _transcribe_chunk_skips_the_api_call_when_audio_is_too_short():
    """Confirmed against the real endpoint: under ~0.1s of audio comes back
    `400 invalid_value "Audio file might be corrupted or unsupported"`, not
    a transcript. `run_chunked_stt` fires its first call the instant VAD
    triggers, often with only one ~20ms wire frame buffered -- must skip
    the API call rather than surface that 400 as a provider error."""
    client = _fake_client()
    provider = OpenAISTTProvider(client)
    one_wire_frame = b"\x01\x02" * 320  # 20ms @ 16kHz, well under the 100ms floor

    result = await provider.transcribe_chunk("utt-1", 0, one_wire_frame)

    assert result == ""
    client.audio.transcriptions.create.assert_not_awaited()


def test_transcribe_chunk_calls_the_api_once_the_minimum_duration_is_met():
    asyncio.run(_transcribe_chunk_calls_the_api_once_the_minimum_duration_is_met())


async def _transcribe_chunk_calls_the_api_once_the_minimum_duration_is_met():
    client = _fake_client(response_text="hi")
    provider = OpenAISTTProvider(client)
    exactly_100ms = b"\x01\x02" * 1600  # 100ms @ 16kHz, the confirmed floor

    result = await provider.transcribe_chunk("utt-1", 0, exactly_100ms)

    assert result == "hi"
    client.audio.transcriptions.create.assert_awaited_once()


def test_transcribe_chunk_handles_none_response_text():
    asyncio.run(_transcribe_chunk_handles_none_response_text())


async def _transcribe_chunk_handles_none_response_text():
    client = _fake_client(response_text=None)
    provider = OpenAISTTProvider(client)

    result = await provider.transcribe_chunk("utt-1", 0, _MIN_DURATION_PCM)

    assert result == ""


def test_transcribe_chunk_maps_429_to_rate_limited_error():
    asyncio.run(_transcribe_chunk_maps_429_to_rate_limited_error())


async def _transcribe_chunk_maps_429_to_rate_limited_error():
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(side_effect=_rate_limit_error())
    provider = OpenAISTTProvider(client)

    with pytest.raises(RateLimitedError):
        await provider.transcribe_chunk("utt-1", 0, _MIN_DURATION_PCM)


def test_transcribe_chunk_propagates_non_rate_limit_api_errors():
    asyncio.run(_transcribe_chunk_propagates_non_rate_limit_api_errors())


async def _transcribe_chunk_propagates_non_rate_limit_api_errors():
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(side_effect=_server_error())
    provider = OpenAISTTProvider(client)

    with pytest.raises(openai.InternalServerError):
        await provider.transcribe_chunk("utt-1", 0, _MIN_DURATION_PCM)


def test_stream_partials_raises_not_implemented():
    asyncio.run(_stream_partials_raises_not_implemented())


async def _stream_partials_raises_not_implemented():
    provider = OpenAISTTProvider(_fake_client())
    with pytest.raises(NotImplementedError):
        async for _ in provider.stream_partials("utt-1"):
            pass
