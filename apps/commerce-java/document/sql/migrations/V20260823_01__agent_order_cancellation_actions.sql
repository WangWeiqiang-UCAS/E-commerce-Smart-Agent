-- Cancellation previews do not have a product, quantity, reason, or amount.
-- Preserve refund rows and use the existing hashed-token/idempotency action ledger.
ALTER TABLE oms_agent_action
    MODIFY product_id BIGINT NULL,
    MODIFY quantity INT NULL,
    MODIFY reason VARCHAR(500) NULL,
    MODIFY preview_amount DECIMAL(10,2) NULL,
    ADD COLUMN order_status_at_preview INT NULL AFTER preview_amount;
