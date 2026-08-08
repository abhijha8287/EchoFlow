"""OpenAI LLM adapter tests — mocked openai client, no real network calls
(matching the design doc's contract-test philosophy).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from llm.openai import create_openai_llm
from stt.provider import RateLimitedError


async def _async_iter(items):
    for item in items:
        yield item


def _chunk(text: str | None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _empty_chunk():
    """Mirrors a trailing usage-only chunk some deployments send, which has
    no choices at all -- must be skipped, not crash on an IndexError."""
    return SimpleNamespace(choices=[])


def _fake_client(chunks: list) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))
    return client


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return openai.RateLimitError("quota exceeded", response=response, body=None)


def _server_error() -> openai.InternalServerError:
    response = httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return openai.InternalServerError("internal", response=response, body=None)


def test_collects_streamed_chunks_into_one_response():
    asyncio.run(_collects_streamed_chunks_into_one_response())


async def _collects_streamed_chunks_into_one_response():
    client = _fake_client([_chunk("Paris "), _chunk("is "), _chunk("the capital.")])
    openai_llm = create_openai_llm(client, model="gpt-4o-mini")

    result = await openai_llm("utt-1", "What is the capital of France?")

    assert result == "Paris is the capital."
    client.chat.completions.create.assert_awaited_once()
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["messages"][-1] == {"role": "user", "content": "What is the capital of France?"}
    assert call_kwargs["stream"] is True


def test_skips_chunks_with_no_text():
    asyncio.run(_skips_chunks_with_no_text())


async def _skips_chunks_with_no_text():
    client = _fake_client([_chunk("hello"), _chunk(None), _chunk(" world")])
    openai_llm = create_openai_llm(client)

    result = await openai_llm("utt-1", "hi")

    assert result == "hello world"


def test_skips_chunks_with_no_choices():
    asyncio.run(_skips_chunks_with_no_choices())


async def _skips_chunks_with_no_choices():
    client = _fake_client([_chunk("hello"), _empty_chunk(), _chunk(" world")])
    openai_llm = create_openai_llm(client)

    result = await openai_llm("utt-1", "hi")  # must not raise IndexError

    assert result == "hello world"


def test_config_uses_the_spoken_answer_policy_defaults():
    asyncio.run(_config_uses_the_spoken_answer_policy_defaults())


async def _config_uses_the_spoken_answer_policy_defaults():
    """design doc `### LLM spoken-answer policy`: temperature 0.7,
    max_tokens 256, a system message telling the model its reply will be
    read aloud."""
    client = _fake_client([_chunk("ok")])
    openai_llm = create_openai_llm(client)

    await openai_llm("utt-1", "hi")

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 256
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"]


def test_maps_429_to_rate_limited_error():
    asyncio.run(_maps_429_to_rate_limited_error())


async def _maps_429_to_rate_limited_error():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
    openai_llm = create_openai_llm(client)

    with pytest.raises(RateLimitedError):
        await openai_llm("utt-1", "hi")


def test_propagates_non_rate_limit_api_errors():
    asyncio.run(_propagates_non_rate_limit_api_errors())


async def _propagates_non_rate_limit_api_errors():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_server_error())
    openai_llm = create_openai_llm(client)

    with pytest.raises(openai.InternalServerError):
        await openai_llm("utt-1", "hi")
