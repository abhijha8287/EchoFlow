"""STT provider interface — mirrors design doc `### Provider registry
behavior` (capability-based branching) and `### Turn-taking policy`
(STT algorithm) (T4).

Concrete Gemini/OpenAI adapters implement this protocol once the provider
spike pins models/endpoints. `backend/stt/algorithm.py` never branches on
which vendor is active, only on `supports_partial_streaming` -- this is
what keeps the "no vendor-specific semantics in the orchestrator" rule (see
design doc Barge-in sequence / Cancel scope boundary) honest for STT too.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


class RateLimitedError(Exception):
    """Raised by a batch/chunked `STTProvider.transcribe_chunk()` on a 429
    (or equivalent) response. `run_chunked_stt` does not catch or retry this
    -- it propagates immediately out of the algorithm, which is what keeps
    the capped-concurrency guarantee honest under a rate limit: there is no
    retry loop that could pile up further requests behind the failed one."""


@runtime_checkable
class STTProvider(Protocol):
    @property
    def supports_partial_streaming(self) -> bool:
        """True for providers with a true partial-streaming transcription
        API; False for batch/chunked-only providers. The algorithm branches
        on this flag, never on vendor identity."""
        ...

    def stream_partials(self, utterance_id: str) -> AsyncIterator[str]:
        """True-streaming providers only (`supports_partial_streaming ==
        True`). Yields partial transcripts as they arrive."""
        ...

    async def transcribe_chunk(self, utterance_id: str, chunk_index: int, pcm: bytes) -> str:
        """Batch/chunked providers only (`supports_partial_streaming ==
        False`). Called at most once at a time by `run_chunked_stt` -- the
        caller never starts chunk N+1 before this returns for chunk N.

        `pcm` is the *accumulated* s16le/16k audio for this utterance so
        far (from utterance start through the current chunk boundary), not
        just the newest slice -- providers with no notion of resuming a
        prior partial transcription (e.g. Gemini's generateContent) need
        the whole thing each call anyway, and this keeps the provider
        itself stateless: the caller (SessionRuntime) owns the buffer."""
        ...
