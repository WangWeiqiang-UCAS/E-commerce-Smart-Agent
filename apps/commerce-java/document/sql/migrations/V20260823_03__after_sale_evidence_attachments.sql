CREATE TABLE IF NOT EXISTS oms_after_sale_attachment (
    id BIGINT NOT NULL AUTO_INCREMENT,
    after_sale_id BIGINT NULL,
    member_id BIGINT NOT NULL,
    object_key VARCHAR(255) NOT NULL,
    content_type VARCHAR(64) NOT NULL,
    content_length BIGINT NOT NULL,
    created_at DATETIME NOT NULL,
    attached_at DATETIME NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_after_sale_attachment_object_key (object_key),
    KEY idx_after_sale_attachment_member (member_id, after_sale_id, id),
    CONSTRAINT fk_after_sale_attachment_apply FOREIGN KEY (after_sale_id) REFERENCES oms_order_return_apply (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Private customer-owned after-sales evidence objects';
