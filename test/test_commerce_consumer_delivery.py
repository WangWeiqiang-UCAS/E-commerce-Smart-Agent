from app.events import commerce_consumer


class Message:
    def __init__(self):
        self.acked = False
        self.rejected = []

    def ack(self):
        self.acked = True

    def reject(self, *, requeue):
        self.rejected.append(requeue)


def test_valid_event_is_acknowledged_after_persistence(monkeypatch):
    async def stored(_):
        return True

    monkeypatch.setattr(commerce_consumer, "record_event", stored)
    message = Message()
    commerce_consumer.handle_message({"eventId": "ignored"}, message)

    assert message.acked is True
    assert message.rejected == []


def test_invalid_event_is_dead_lettered(monkeypatch):
    async def invalid(_):
        raise ValueError("invalid event")

    monkeypatch.setattr(commerce_consumer, "record_event", invalid)
    message = Message()
    commerce_consumer.handle_message({}, message)

    assert message.acked is False
    assert message.rejected == [False]


def test_transient_failure_requests_safe_requeue(monkeypatch):
    async def unavailable(_):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(commerce_consumer, "record_event", unavailable)
    message = Message()
    commerce_consumer.handle_message({}, message)

    assert message.acked is False
    assert message.rejected == [True]
