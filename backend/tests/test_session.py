"""SessionStore unit tests — see `backend/core/session.py`."""

from __future__ import annotations

from core.session import SessionStore
from voice.fsm import VoiceState


def test_create_assigns_a_unique_session_id_and_default_idle_fsm():
    store = SessionStore()
    session = store.create()
    assert session.session_id
    assert session.fsm.state is VoiceState.IDLE
    assert session.mode == "continuous"


def test_get_returns_the_same_session_created_earlier():
    store = SessionStore()
    session = store.create(mode="ptt")
    assert store.get(session.session_id) is session


def test_get_on_unknown_id_returns_none():
    store = SessionStore()
    assert store.get("does-not-exist") is None


def test_remove_forgets_the_session():
    store = SessionStore()
    session = store.create()
    store.remove(session.session_id)
    assert store.get(session.session_id) is None


def test_invalid_mode_falls_back_to_continuous():
    store = SessionStore()
    session = store.create(mode="not-a-real-mode")
    assert session.mode == "continuous"


def test_two_sessions_never_collide_on_id():
    store = SessionStore()
    ids = {store.create().session_id for _ in range(50)}
    assert len(ids) == 50
