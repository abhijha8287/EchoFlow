"""Gemini STT adapter tests — mocked google-genai client, no real network
calls (matching the design doc's contract-test philosophy: mock the
vendor's API shape, not the live endpoint).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors

from stt.gemini import TRANSCRIBE_PROMPT, GeminiSTTProvider
from stt.provider import RateLimitedError


def _fake_client(response_text: str | None = "hello world") -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text=response_text))
    return client


def test_supports_partial_streaming_is_false():
    """Repeated spike runs showed chunk_count varying (transport-level
    chunking, not confirmed incremental transcript growth) -- the adapter
    must use the chunked-upload path (T4) regardless."""
    assert GeminiSTTProvider(_fake_client()).supports_partial_streaming is False


def test_transcribe_chunk_returns_stripped_response_text():
    asyncio.run(_transcribe_chunk_returns_stripped_response_text())


async def _transcribe_chunk_returns_stripped_response_text():
    client = _fake_client(response_text="  hello world  \n")
    provider = GeminiSTTProvider(client, model="gemini-2.5-flash")

    result = await provider.transcribe_chunk("utt-1", 0, b"\x01\x02\x03\x04")

    assert result == "hello world"
    client.aio.models.generate_content.assert_awaited_once()
    call_kwargs = client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    contents = call_kwargs["contents"]
    assert len(contents) == 2
    assert contents[0].inline_data.mime_type == "audio/wav"
    assert contents[1] == TRANSCRIBE_PROMPT


def test_transcribe_chunk_wraps_pcm_as_a_valid_wav_container():
    asyncio.run(_transcribe_chunk_wraps_pcm_as_a_valid_wav_container())


async def _transcribe_chunk_wraps_pcm_as_a_valid_wav_container():
    import wave
    from io import BytesIO

    client = _fake_client()
    provider = GeminiSTTProvider(client)
    pcm = b"\x01\x02\x03\x04"

    await provider.transcribe_chunk("utt-1", 0, pcm)

    wav_bytes = client.aio.models.generate_content.call_args.kwargs["contents"][0].inline_data.data
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(2) == pcm


def test_transcribe_chunk_with_empty_pcm_skips_the_api_call():
    asyncio.run(_transcribe_chunk_with_empty_pcm_skips_the_api_call())


async def _transcribe_chunk_with_empty_pcm_skips_the_api_call():
    client = _fake_client()
    provider = GeminiSTTProvider(client)

    result = await provider.transcribe_chunk("utt-1", 0, b"")

    assert result == ""
    client.aio.models.generate_content.assert_not_awaited()


def test_transcribe_chunk_handles_none_response_text():
    asyncio.run(_transcribe_chunk_handles_none_response_text())


async def _transcribe_chunk_handles_none_response_text():
    """Gemini can return a candidate with no text (e.g. safety-filtered) --
    must not crash on None, must return an empty transcript (the design
    doc's EMPTY_TRANSCRIPT path handles this upstream)."""
    client = _fake_client(response_text=None)
    provider = GeminiSTTProvider(client)

    result = await provider.transcribe_chunk("utt-1", 0, b"\x01\x02")

    assert result == ""


def test_transcribe_chunk_maps_429_to_rate_limited_error():
    asyncio.run(_transcribe_chunk_maps_429_to_rate_limited_error())


async def _transcribe_chunk_maps_429_to_rate_limited_error():
    client = MagicMock()
    api_error = errors.ClientError(429, {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    client.aio.models.generate_content = AsyncMock(side_effect=api_error)
    provider = GeminiSTTProvider(client)

    with pytest.raises(RateLimitedError):
        await provider.transcribe_chunk("utt-1", 0, b"\x01\x02")


def test_transcribe_chunk_propagates_non_rate_limit_api_errors():
    asyncio.run(_transcribe_chunk_propagates_non_rate_limit_api_errors())


async def _transcribe_chunk_propagates_non_rate_limit_api_errors():
    client = MagicMock()
    api_error = errors.ServerError(500, {"error": {"message": "internal", "status": "INTERNAL"}})
    client.aio.models.generate_content = AsyncMock(side_effect=api_error)
    provider = GeminiSTTProvider(client)

    with pytest.raises(errors.ServerError):
        await provider.transcribe_chunk("utt-1", 0, b"\x01\x02")


def test_stream_partials_raises_not_implemented():
    asyncio.run(_stream_partials_raises_not_implemented())


async def _stream_partials_raises_not_implemented():
    provider = GeminiSTTProvider(_fake_client())
    with pytest.raises(NotImplementedError):
        async for _ in provider.stream_partials("utt-1"):
            pass
