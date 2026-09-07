# Database backup, migration, and restore runbook

This procedure protects the two independent systems of record in this
repository: Commerce MySQL and Agent PostgreSQL. Redis, RabbitMQ, and MongoDB
are not authoritative stores for commerce or Agent conversations; preserve
their configuration separately when an incident requires it.

After-sales evidence is stored separately in the private S3-compatible MinIO
bucket. The attachment metadata is in Commerce MySQL, but a database dump alone
cannot restore the evidence. Treat the configured `MINIO_EVIDENCE_BUCKET` as a
required companion backup.

## Preconditions

- Assign an incident owner and record the deployment/incident trace ID.
- Put the customer BFF and Agent desk into maintenance mode before any restore
  or schema change. Do not accept checkout, payment, refund-confirm, or desk
  reply writes while a restore is in progress.
- Confirm available backup storage is encrypted, access-controlled, and has
  capacity for a complete MySQL/PostgreSQL dump and an object-level copy of the
  evidence bucket.
- Test these commands in a non-production environment first. Never place
  database passwords in shell history, ticket comments, or chat transcripts.

## Backup before every release

Run these from an operator host with the Compose environment loaded. Redirect
the output to an encrypted, access-controlled backup location; the examples
use a deliberately non-production placeholder path.

```powershell
docker compose exec -T db pg_dump --format=custom --no-owner --dbname=knowledge_base > D:\secure-backups\agent-postgres-before-release.dump
docker compose exec -T mall-mysql sh -c 'mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" --single-transaction --routines --events --databases mall' > D:\secure-backups\commerce-mysql-before-release.sql
Get-FileHash D:\secure-backups\agent-postgres-before-release.dump -Algorithm SHA256
Get-FileHash D:\secure-backups\commerce-mysql-before-release.sql -Algorithm SHA256
```

Record the hashes, UTC timestamp, Git revision, migration versions, and the
result of a restore rehearsal. A successful command is not a usable backup
until a restore has been verified.

In the same backup window, create an encrypted S3/MinIO bucket snapshot or a
version-preserving `mc mirror` of `MINIO_EVIDENCE_BUCKET`. Store its object
inventory (key, version or ETag, size, and timestamp) alongside the database
backup manifest. Do not copy MinIO's internal `/data` directory while it is
online: use the storage provider's consistent snapshot capability or the S3
object API.

## Release migration order

1. Take and verify both backups above.
2. Deploy code that is backward-compatible with the old schema.
3. Apply all Commerce SQL migrations in lexical version order. For a new local
   Compose volume, `docker-entrypoint-initdb.d` applies `mall.sql`,
   `V20260808_01__agent_refund_actions.sql`,
   `V20260820_01__outbox_publish_leases.sql`, and
   `V20260823_01__agent_order_cancellation_actions.sql`, and
   `V20260823_02__payment_refund_dispatch.sql`, and
   `V20260823_03__after_sale_evidence_attachments.sql`. Existing databases do **not**
   rerun these init scripts; an operator must apply the missing versioned SQL
   files through the approved MySQL migration connection.
4. Apply Agent migrations once: `docker compose run --rm migrate`. Confirm
   `alembic current` reports the expected head before starting Agent writers.
5. Start the services, verify `/actuator/health`, BFF `/api/health`, Agent
   `/health`, and the event consumer. Confirm no outbox event is permanently
   leased and that the Inbox continues to deduplicate replayed events.

Do not perform destructive or irreversible Commerce migrations in the same
release as an application rollback decision. Use expand/migrate/contract:
deploy additive schema first, switch reads and writes, then remove old fields
only after the rollback window has expired.

## Restore procedure

Restore is an incident operation. Obtain change approval, preserve the failed
database volumes/snapshots for investigation, and stop all writers first.

1. Create a fresh isolated MySQL/PostgreSQL target; do not overwrite the only
   surviving copy of the failed databases.
2. Restore PostgreSQL with `pg_restore --clean --if-exists` and MySQL with the
   approved `mysql` client using the verified backups.
3. Restore the matching evidence-bucket snapshot before starting Commerce
   writers. Check that every non-null `oms_after_sale_attachment.object_key`
   in the restored database is present in the restored bucket; retain the
   original object version/ETag in the incident evidence.
4. Run schema verification and start only read-only health checks. Compare row
   counts for Agent threads/messages/inbox and Commerce orders/payments/after
   sales against the recorded backup metadata.
5. Reconcile the Commerce outbox: unpublished rows may be replayed. This is
   expected; Agent `inbox_events.event_id` makes the replay idempotent. Do not
   delete Inbox rows merely to silence duplicate messages.
6. Switch traffic only after an operator has sampled owned-order reads, an
   after-sales status read, and a customer conversation history. Record the
   incident trace ID and restoration evidence.

Run a complete non-production restore drill at least quarterly and after any
change to database topology, object storage, encryption, or migrations.
