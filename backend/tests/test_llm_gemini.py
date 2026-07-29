"""Gemini LLM adapter tests — mocked google-genai client, no real network
calls (matching the design doc's contract-test philosophy).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors

from llm.gemini import create_gemini_llm
from stt.provider import RateLimitedError


async def _async_iter(items):
    for item in items:
        yield item


def _fake_client(chunks: list[str]) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(
        return_value=_async_iter([SimpleNamespace(text=t) for t in chunks])
    )
    return client


def test_collects_streamed_chunks_into_one_response():
    asyncio.run(_collects_streamed_chunks_into_one_response())


async def _collects_streamed_chunks_into_one_response():
    client = _fake_client(["Paris ", "is ", "the capital."])
    gemini_llm = create_gemini_llm(client, model="gemini-2.5-flash")

    result = await gemini_llm("utt-1", "What is the capital of France?")

    assert result == "Paris is the capital."
    client.aio.models.generate_content_stream.assert_awaited_once()
    call_kwargs = client.aio.models.generate_content_stream.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "What is the capital of France?"


def test_skips_chunks_with_no_text():
    asyncio.run(_skips_chunks_with_no_text())


async def _skips_chunks_with_no_text():
    client = _fake_client(["hello", None, " world"])  # type: ignore[list-item]
    gemini_llm = create_gemini_llm(client)

    result = await gemini_llm("utt-1", "hi")

    assert result == "hello world"


def test_config_uses_the_spoken_answer_policy_defaults():
    asyncio.run(_config_uses_the_spoken_answer_policy_defaults())


async def _config_uses_the_spoken_answer_policy_defaults():
    """design doc `### LLM spoken-answer policy`: temperature 0.7,
    max_output_tokens 256, a system instruction telling the model its
    reply will be read aloud."""
    client = _fake_client(["ok"])
    gemini_llm = create_gemini_llm(client)

    await gemini_llm("utt-1", "hi")

    config = client.aio.models.generate_content_stream.call_args.kwargs["config"]
    assert config.temperature == 0.7
    assert config.max_output_tokens == 256
    assert config.system_instruction


def test_maps_429_to_rate_limited_error():
    asyncio.run(_maps_429_to_rate_limited_error())


async def _maps_429_to_rate_limited_error():
    client = MagicMock()
    api_error = errors.ClientError(429, {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    client.aio.models.generate_content_stream = AsyncMock(side_effect=api_error)
    gemini_llm = create_gemini_llm(client)

    with pytest.raises(RateLimitedError):
        await gemini_llm("utt-1", "hi")


def test_propagates_non_rate_limit_api_errors():
    asyncio.run(_propagates_non_rate_limit_api_errors())


async def _propagates_non_rate_limit_api_errors():
    client = MagicMock()
    api_error = errors.ServerError(500, {"error": {"message": "internal", "status": "INTERNAL"}})
    client.aio.models.generate_content_stream = AsyncMock(side_effect=api_error)
    gemini_llm = create_gemini_llm(client)

    with pytest.raises(errors.ServerError):
        await gemini_llm("utt-1", "hi")
