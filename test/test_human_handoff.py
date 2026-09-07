import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from app.api.v1 import chat
from app.models.support_operator import SupportOperator
from app.models.thread_assignment import ThreadAssignment, ThreadMode


class _Result:
    def __init__(self, value):
        self.value = value

    def first(self):
        return self.value


class _Session:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def exec(self, _):
        return _Result(self.value)


def _session_factory(value):
    return lambda: _Session(value)


@pytest.mark.asyncio
async def test_human_owned_thread_rejects_agent_reply(monkeypatch):
    monkeypatch.setattr(
        chat,
        "async_session_maker",
        _session_factory(ThreadAssignment(thread_id="1_customer", mode=ThreadMode.HUMAN, assigned_admin_id=9)),
    )

    with pytest.raises(HTTPException) as error:
        await chat._require_agent_owned_thread("1_customer")

    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_agent_owned_thread_allows_agent_reply(monkeypatch):
    monkeypatch.setattr(
        chat,
        "async_session_maker",
        _session_factory(ThreadAssignment(thread_id="1_customer", mode=ThreadMode.AGENT)),
    )

    await chat._require_agent_owned_thread("1_customer")


@pytest.mark.asyncio
async def test_concurrent_takeover_is_reported_as_conflict(monkeypatch):
    from app.api.v1 import admin

    class ConflictingSession(_Session):
        def add(self, _item):
            return None

        async def commit(self):
            raise IntegrityError("INSERT thread_assignments", {}, RuntimeError("duplicate"))

        async def rollback(self):
            return None

    monkeypatch.setattr(admin, "async_session_maker", lambda: ConflictingSession(None))

    with pytest.raises(HTTPException) as error:
        await admin.takeover_thread("1_customer", admin.TakeoverRequest(), 9)

    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_human_owner_can_transfer_to_active_administrator_with_audit_event(monkeypatch):
    from app.api.v1 import admin

    assignment = ThreadAssignment(thread_id="1_customer", mode=ThreadMode.HUMAN, assigned_admin_id=9)
    target = SupportOperator(id=10, username="second", password_hash="x", email="second@example.test",
                             full_name="Second Operator", is_admin=True, is_active=True)

    class TransferSession(_Session):
        def __init__(self):
            super().__init__(None)
            self.responses = [assignment, target]
            self.added = []

        async def exec(self, _):
            return _Result(self.responses.pop(0))

        def add(self, item):
            self.added.append(item)

        async def commit(self):
            return None

        async def refresh(self, _):
            return None

    transfer_session = TransferSession()
    monkeypatch.setattr(admin, "async_session_maker", lambda: transfer_session)

    response = await admin.transfer_thread("1_customer", admin.TransferRequest(target_admin_id=10, reason="specialist"), 9)

    assert response.assigned_admin_id == 10
    assert assignment.reason == "specialist"
    assert transfer_session.added[-1].content["event"] == "human_transfer"


def test_customer_history_excludes_internal_notes():
    source = (chat.__file__ and open(chat.__file__, encoding="utf-8").read())

    assert '.where(MessageCard.sender_type != "internal")' in source


def test_internal_note_ciphertext_is_authenticated_and_not_plaintext():
    from app.core.field_encryption import decrypt_internal_note, encrypt_internal_note

    ciphertext = encrypt_internal_note("customer address: private")
    assert ciphertext != "customer address: private"
    assert decrypt_internal_note(ciphertext) == "customer address: private"
