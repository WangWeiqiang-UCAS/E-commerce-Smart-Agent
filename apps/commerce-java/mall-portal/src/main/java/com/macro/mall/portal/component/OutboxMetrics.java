package com.macro.mall.portal.component;

import io.micrometer.core.instrument.MeterRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Bounded operational signals for the durable commerce outbox. Values are
 * refreshed outside the publisher transaction so Prometheus scrapes never
 * create write contention on commerce state.
 */
@Component
public class OutboxMetrics {
    private static final Logger LOGGER = LoggerFactory.getLogger(OutboxMetrics.class);
    private final JdbcTemplate jdbcTemplate;
    private final AtomicLong pending = new AtomicLong();
    private final AtomicLong oldestUnpublishedSeconds = new AtomicLong();
    private final AtomicLong expiredLeases = new AtomicLong();
    private final AtomicLong pendingRefunds = new AtomicLong();
    private final AtomicLong failedRefunds = new AtomicLong();

    public OutboxMetrics(JdbcTemplate jdbcTemplate, MeterRegistry meterRegistry) {
        this.jdbcTemplate = jdbcTemplate;
        meterRegistry.gauge("commerce.outbox.pending", pending);
        meterRegistry.gauge("commerce.outbox.oldest.unpublished.seconds", oldestUnpublishedSeconds);
        meterRegistry.gauge("commerce.outbox.expired.leases", expiredLeases);
        meterRegistry.gauge("commerce.refund.pending", pendingRefunds);
        meterRegistry.gauge("commerce.refund.failed", failedRefunds);
    }

    @Scheduled(fixedDelayString = "${outbox.metrics-refresh-delay-ms:15000}")
    public void refresh() {
        try {
            pending.set(requiredLong("SELECT COUNT(*) FROM oms_outbox_event WHERE published_at IS NULL"));
            oldestUnpublishedSeconds.set(requiredLong("SELECT COALESCE(TIMESTAMPDIFF(SECOND, MIN(occurred_at), NOW()), 0) "
                    + "FROM oms_outbox_event WHERE published_at IS NULL"));
            expiredLeases.set(requiredLong("SELECT COUNT(*) FROM oms_outbox_event WHERE published_at IS NULL "
                    + "AND publish_locked_at IS NOT NULL "
                    + "AND publish_locked_at < DATE_SUB(NOW(), INTERVAL 60 SECOND)"));
            pendingRefunds.set(requiredLong("SELECT COUNT(*) FROM oms_payment_refund "
                    + "WHERE status IN ('PENDING','RETRY','PROCESSING')"));
            failedRefunds.set(requiredLong("SELECT COUNT(*) FROM oms_payment_refund WHERE status='FAILED'"));
        } catch (Exception error) {
            // Preserve the most recent values and keep a transient monitoring
            // query failure from affecting payment, order, or event publishing.
            LOGGER.warn("Unable to refresh commerce outbox metrics", error);
        }
    }

    private long requiredLong(String sql) {
        Long value = jdbcTemplate.queryForObject(sql, Long.class);
        return value == null ? 0L : value;
    }
}
