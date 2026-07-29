"""STT capability-based branching algorithm — mirrors design doc `###
Turn-taking policy` (STT algorithm) and `### Provider registry behavior`
(T4).

Branches on `STTProvider.supports_partial_streaming`, never on vendor
identity:

- **True streaming** providers are drained directly via `stream_partials()`
  (`run_streaming_stt`).
- **Batch/chunked** providers (`supports_partial_streaming == False`) are
  driven by `run_chunked_stt`, which bounds concurrency to at most one
  in-flight request by construction: the next chunk is never started until
  the previous chunk's response has returned, at an interval of 800ms-1s
  between chunk *starts* (not a fixed 300ms clock). This is what keeps
  `RATE_LIMITED` exposure and per-utterance cost down on batch-priced
  endpoints, and what makes "last partial" unambiguous at the VAD endpoint
  -- there is never more than one in-flight response to disambiguate
  between, so "last received" and "last sent" are always the same response.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from .provider import STTProvider

# "not 300ms" (see design doc D3): the interval between chunk starts is
# sized to a batch endpoint's typical response time rather than a fixed
# clock tick.
MIN_CHUNK_INTERVAL_S = 0.8
MAX_CHUNK_INTERVAL_S = 1.0

OnPartial = Callable[[str], None]
GetAudio = Callable[[], bytes]


async def run_streaming_stt(provider: STTProvider, utterance_id: str, on_partial: OnPartial) -> str:
    """Drive a true-streaming `STTProvider`. Partials are forwarded as they
    arrive; returns the last partial received (the caller promotes this to
    `transcript.final` if the VAD endpoint arrives before the provider's own
    final -- see the design doc's 150ms final-wait rule)."""
    last_partial = ""
    async for partial in provider.stream_partials(utterance_id):
        last_partial = partial
        on_partial(partial)
    return last_partial


async def run_chunked_stt(
    provider: STTProvider,
    utterance_id: str,
    on_partial: OnPartial,
    endpoint: asyncio.Event,
    get_audio: GetAudio,
    min_interval_s: float = MIN_CHUNK_INTERVAL_S,
) -> str:
    """Drive a batch/chunked `STTProvider` under the capped-concurrency rule.

    Loops until `endpoint` is set (the VAD endpoint has been reached),
    awaiting each `transcribe_chunk()` call fully before starting the next
    -- at most one request in flight at any time -- and never starting a new
    chunk sooner than `min_interval_s` after the previous one started. Each
    call gets a fresh snapshot of the accumulated utterance audio via
    `get_audio()` -- most batch providers (e.g. Gemini's generateContent)
    have no notion of resuming a prior partial transcription, so each chunk
    re-sends everything captured so far, not just what's new.

    Returns the last partial transcript, which is the unambiguous
    `transcript.final` candidate: because concurrency is capped to 1, there
    is only ever one in-flight response, so there is nothing to disambiguate
    "last" against.

    Always makes one final `transcribe_chunk()` call against the complete
    buffer once `endpoint` is set, even if that means zero or one prior
    in-loop calls -- without it, whatever audio arrived after the last
    in-loop call started (which, for a short utterance, can be *all* of
    it) would never actually reach the provider, silently truncating the
    transcript. This final call is not interval-gated: nothing follows it
    for this utterance, so there is no pileup risk to guard against.
    """
    last_partial = ""
    chunk_index = 0
    loop = asyncio.get_running_loop()

    while not endpoint.is_set():
        start = loop.time()
        last_partial = await provider.transcribe_chunk(utterance_id, chunk_index, get_audio())
        on_partial(last_partial)
        chunk_index += 1

        if endpoint.is_set():
            break

        remaining = min_interval_s - (loop.time() - start)
        if remaining > 0:
            try:
                await asyncio.wait_for(endpoint.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass  # interval elapsed with no endpoint yet -> start next chunk

    last_partial = await provider.transcribe_chunk(utterance_id, chunk_index, get_audio())
    return last_partial


async def run_stt(
    provider: STTProvider,
    utterance_id: str,
    on_partial: OnPartial,
    endpoint: asyncio.Event,
    get_audio: GetAudio,
) -> str:
    """Top-level dispatcher: branch on `supports_partial_streaming`, not on
    vendor identity (see module docstring). `get_audio` is only used by the
    chunked path -- true-streaming providers get audio pushed to them some
    other way (unresolved until a real streaming provider exists; every
    provider wired so far, Gemini included, is batch/chunked)."""
    if provider.supports_partial_streaming:
        return await run_streaming_stt(provider, utterance_id, on_partial)
    return await run_chunked_stt(provider, utterance_id, on_partial, endpoint, get_audio)
