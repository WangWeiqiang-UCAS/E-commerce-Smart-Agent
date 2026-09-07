-- Durable, provider-idempotent refund work created only after the Java
-- after-sales state machine has accepted and received a return.
CREATE TABLE IF NOT EXISTS oms_payment_refund (
    id BIGINT NOT NULL AUTO_INCREMENT,
    after_sale_id BIGINT NOT NULL,
    order_id BIGINT NOT NULL,
    payment_channel VARCHAR(32) NOT NULL,
    refund_request_no VARCHAR(64) NOT NULL,
    refund_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'PENDING',
    provider_refund_no VARCHAR(128) NULL,
    attempts INT NOT NULL DEFAULT 0,
    next_attempt_at DATETIME NULL,
    dispatch_lock_owner VARCHAR(64) NULL,
    dispatch_locked_at DATETIME NULL,
    last_error VARCHAR(512) NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    completed_at DATETIME NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_payment_refund_after_sale (after_sale_id),
    UNIQUE KEY uk_payment_refund_request_no (refund_request_no),
    KEY idx_payment_refund_dispatch (status, next_attempt_at, dispatch_locked_at, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Durable payment refund dispatch records';
