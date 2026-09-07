-- Do not hold database row locks while waiting for a broker publisher confirm.
-- A short lease makes concurrent publishers safe and allows recovery after a
-- process crash. Consumer inbox de-duplication remains the final safeguard.
ALTER TABLE oms_outbox_event
    ADD COLUMN publish_lock_owner VARCHAR(64) NULL,
    ADD COLUMN publish_locked_at DATETIME NULL,
    ADD KEY idx_outbox_publish_lease (published_at, publish_locked_at, id);
