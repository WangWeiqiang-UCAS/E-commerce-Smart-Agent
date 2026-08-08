-- Run this migration before enabling /agent/commerce/actions/refund endpoints.
-- Tokens are stored only as SHA-256 hashes; the plaintext token is returned once.
CREATE TABLE oms_agent_action (
  id BIGINT NOT NULL AUTO_INCREMENT,
  after_sale_id BIGINT NULL,
    action_token_hash CHAR(64) NOT NULL,
    member_id BIGINT NOT NULL,
    action_type VARCHAR(32) NOT NULL,
    order_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    quantity INT NOT NULL,
    reason VARCHAR(500) NOT NULL,
    agent_trace_id VARCHAR(128) NOT NULL,
    preview_hash CHAR(64) NOT NULL,
    preview_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(16) NOT NULL,
    idempotency_key VARCHAR(64) NULL,
    created_at DATETIME NOT NULL,
    expires_at DATETIME NOT NULL,
    confirmed_at DATETIME NULL,
    consumed_at DATETIME NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_agent_action_token (action_token_hash),
    UNIQUE KEY uk_agent_action_idempotency (member_id, action_type, idempotency_key),
    KEY idx_agent_action_expiry (status, expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE oms_outbox_event (
    id BIGINT NOT NULL AUTO_INCREMENT,
    event_id CHAR(36) NOT NULL,
    event_type VARCHAR(128) NOT NULL,
    schema_version INT NOT NULL,
    aggregate_id VARCHAR(64) NOT NULL,
    trace_id VARCHAR(128) NULL,
    payload JSON NOT NULL,
    occurred_at DATETIME NOT NULL,
    published_at DATETIME NULL,
    attempts INT NOT NULL DEFAULT 0,
    PRIMARY KEY (id),
    UNIQUE KEY uk_outbox_event_id (event_id),
    KEY idx_outbox_unpublished (published_at, occurred_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
