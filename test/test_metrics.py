from pathlib import Path

from app.core.metrics import record_commerce_call, record_http, render_prometheus


def test_prometheus_metrics_use_bounded_operational_labels():
    record_http("POST", "/api/v1/chat", 200, 0.125)
    record_commerce_call("get_orders", "success", 0.25)

    output = render_prometheus()

    assert 'agent_http_requests_total{method="POST",route="/api/v1/chat",status="2xx"}' in output
    assert 'agent_commerce_tool_calls_total{tool="get_orders",outcome="success"}' in output
    assert "agent_commerce_tool_duration_seconds_count" in output


def test_commerce_outbox_metrics_and_recovery_runbook_cover_stuck_event_operations():
    root = Path(__file__).resolve().parents[1]
    source = (root / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/component/OutboxMetrics.java").read_text(encoding="utf-8")
    runbook = (root / "deploy/runbooks/event-recovery.md").read_text(encoding="utf-8")

    assert 'meterRegistry.gauge("commerce.outbox.pending"' in source
    assert 'meterRegistry.gauge("commerce.outbox.oldest.unpublished.seconds"' in source
    assert 'meterRegistry.gauge("commerce.outbox.expired.leases"' in source
    assert 'meterRegistry.gauge("commerce.refund.pending"' in source
    assert 'meterRegistry.gauge("commerce.refund.failed"' in source
    assert "publish_locked_at < DATE_SUB(NOW(), INTERVAL 60 SECOND)" in source
    assert "inbox_events" in runbook
    assert "Never delete an Inbox row" in runbook
