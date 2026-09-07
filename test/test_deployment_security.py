from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_observability_stack_requires_a_grafana_password():
    compose = (ROOT / "deploy/observability/docker-compose.observability.yaml").read_text(encoding="utf-8")
    assert "GRAFANA_ADMIN_PASSWORD:?Set GRAFANA_ADMIN_PASSWORD" in compose
    assert ":-admin" not in compose


def test_example_environment_documents_grafana_and_distributed_rate_limit_settings():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GRAFANA_ADMIN_PASSWORD=replace-with-a-long-random-grafana-password" in example
    assert "RATE_LIMIT_REDIS_TIMEOUT_SECONDS=0.25" in example


def test_primary_compose_requires_database_and_broker_credentials():
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    for variable in (
        "POSTGRES_PASSWORD",
        "RABBITMQ_DEFAULT_USER",
        "RABBITMQ_DEFAULT_PASS",
        "MALL_MYSQL_ROOT_PASSWORD",
        "MALL_MYSQL_USER",
        "MALL_MYSQL_PASSWORD",
    ):
        assert f"${{{variable}:?Set {variable}" in compose
    assert "POSTGRES_PASSWORD:-admin" not in compose
    assert "RABBITMQ_DEFAULT_PASS:-guest" not in compose
    assert "MYSQL_ROOT_PASSWORD: root" not in compose


def test_compose_initializes_all_versioned_commerce_schema_extensions_and_has_a_restore_runbook():
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    runbook = (ROOT / "deploy/runbooks/database-backup-restore.md").read_text(encoding="utf-8")

    assert "V20260808_01__agent_refund_actions.sql" in compose
    assert "V20260820_01__outbox_publish_leases.sql" in compose
    assert "V20260823_03__after_sale_evidence_attachments.sql" in compose
    assert "pg_dump" in runbook
    assert "mysqldump" in runbook
    assert "inbox_events.event_id" in runbook
    assert "MINIO_EVIDENCE_BUCKET" in runbook
    assert "oms_after_sale_attachment.object_key" in runbook
    assert "quarterly" in runbook


def test_compose_waits_for_private_evidence_storage_before_starting_commerce():
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

    assert 'test: ["CMD", "mc", "ready", "local"]' in compose
    assert "MINIO_ACCESS_KEY:?Set MINIO_ACCESS_KEY" in compose
    assert "MINIO_SECRET_KEY:?Set MINIO_SECRET_KEY" in compose
    assert "minio:\n        condition: service_healthy" in compose
