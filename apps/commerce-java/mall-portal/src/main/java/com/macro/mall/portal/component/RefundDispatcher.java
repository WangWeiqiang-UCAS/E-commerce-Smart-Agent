package com.macro.mall.portal.component;

import com.macro.mall.portal.service.AlipayRefundGateway;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Executes only persisted refund work. Provider calls happen outside database
 * transactions; settlement plus the refund.completed outbox event are atomic.
 */
@Component
public class RefundDispatcher {
    private static final Logger LOGGER = LoggerFactory.getLogger(RefundDispatcher.class);
    private static final int MAX_REFUNDS_PER_RUN = 20;
    private final JdbcTemplate jdbcTemplate;
    private final AlipayRefundGateway alipayRefundGateway;
    private final TransactionTemplate transactionTemplate;
    private final String dispatcherId = UUID.randomUUID().toString();

    public RefundDispatcher(JdbcTemplate jdbcTemplate, AlipayRefundGateway alipayRefundGateway,
                            PlatformTransactionManager transactionManager) {
        this.jdbcTemplate = jdbcTemplate;
        this.alipayRefundGateway = alipayRefundGateway;
        this.transactionTemplate = new TransactionTemplate(transactionManager);
    }

    @Scheduled(fixedDelayString = "${refund.dispatch-delay-ms:1000}")
    public void dispatch() {
        for (int index = 0; index < MAX_REFUNDS_PER_RUN; index++) {
            Map<String, Object> refund = claimNextRefund();
            if (refund == null) {
                return;
            }
            try {
                if (!"ALIPAY".equals(refund.get("payment_channel"))) {
                    markManualReview(refund, "No configured refund adapter for payment channel");
                    continue;
                }
                String providerRefundNo = alipayRefundGateway.refund(
                        (String) refund.get("refund_request_no"),
                        (String) refund.get("order_sn"),
                        new BigDecimal(String.valueOf(refund.get("refund_amount"))));
                markSucceeded(refund, providerRefundNo);
            } catch (Exception error) {
                releaseForRetry(refund, error);
            }
        }
    }

    private Map<String, Object> claimNextRefund() {
        int claimed = jdbcTemplate.update("UPDATE oms_payment_refund SET status='PROCESSING', dispatch_lock_owner=?, "
                        + "dispatch_locked_at=NOW(), updated_at=NOW() WHERE id=(SELECT id FROM (SELECT id "
                        + "FROM oms_payment_refund WHERE status IN ('PENDING','RETRY') "
                        + "AND (next_attempt_at IS NULL OR next_attempt_at <= NOW()) "
                        + "AND (dispatch_locked_at IS NULL OR dispatch_locked_at < DATE_SUB(NOW(), INTERVAL 60 SECOND)) "
                        + "ORDER BY id LIMIT 1) AS claimable)", dispatcherId);
        if (claimed != 1) {
            return null;
        }
        List<Map<String, Object>> refunds = jdbcTemplate.queryForList("SELECT r.id,r.after_sale_id,r.payment_channel,"
                        + "r.refund_request_no,r.refund_amount,o.order_sn FROM oms_payment_refund r "
                        + "JOIN oms_order o ON o.id=r.order_id WHERE r.status='PROCESSING' "
                        + "AND r.dispatch_lock_owner=? ORDER BY r.dispatch_locked_at DESC,r.id LIMIT 1", dispatcherId);
        return refunds.isEmpty() ? null : refunds.get(0);
    }

    private void markSucceeded(Map<String, Object> refund, String providerRefundNo) {
        transactionTemplate.executeWithoutResult(ignored -> {
            int updated = jdbcTemplate.update("UPDATE oms_payment_refund SET status='SUCCEEDED', provider_refund_no=?, "
                            + "completed_at=NOW(), updated_at=NOW(), dispatch_lock_owner=NULL, dispatch_locked_at=NULL "
                            + "WHERE id=? AND status='PROCESSING' AND dispatch_lock_owner=?",
                    providerRefundNo, refund.get("id"), dispatcherId);
            if (updated != 1) {
                throw new IllegalStateException("Refund lease expired before settlement");
            }
            String eventId = UUID.randomUUID().toString();
            String occurredAt = Instant.now().toString();
            String payload = "{\"eventId\":\"" + eventId + "\",\"eventType\":\"refund.completed\",\"schemaVersion\":1,"
                    + "\"aggregateId\":\"" + refund.get("after_sale_id") + "\",\"afterSaleId\":\""
                    + refund.get("after_sale_id") + "\",\"refundId\":\"" + refund.get("id")
                    + "\",\"refundAmount\":\"" + refund.get("refund_amount") + "\",\"occurredAt\":\"" + occurredAt + "\"}";
            jdbcTemplate.update("INSERT INTO oms_outbox_event (event_id,event_type,schema_version,aggregate_id,trace_id,payload,occurred_at) "
                            + "VALUES (?,?,?,?,?,?,NOW())", eventId, "refund.completed", 1,
                    String.valueOf(refund.get("after_sale_id")), null, payload);
        });
    }

    private void releaseForRetry(Map<String, Object> refund, Exception error) {
        jdbcTemplate.update("UPDATE oms_payment_refund SET status='RETRY', attempts=attempts+1, "
                        + "next_attempt_at=DATE_ADD(NOW(), INTERVAL 60 SECOND), last_error=?, updated_at=NOW(), "
                        + "dispatch_lock_owner=NULL, dispatch_locked_at=NULL WHERE id=? AND status='PROCESSING' "
                        + "AND dispatch_lock_owner=?", abbreviated(error), refund.get("id"), dispatcherId);
        LOGGER.warn("Refund {} was not completed; it will be retried", refund.get("id"), error);
    }

    private void markManualReview(Map<String, Object> refund, String reason) {
        jdbcTemplate.update("UPDATE oms_payment_refund SET status='FAILED', last_error=?, updated_at=NOW(), "
                        + "dispatch_lock_owner=NULL, dispatch_locked_at=NULL WHERE id=? AND status='PROCESSING' "
                        + "AND dispatch_lock_owner=?", reason, refund.get("id"), dispatcherId);
        LOGGER.error("Refund {} requires manual review: {}", refund.get("id"), reason);
    }

    private String abbreviated(Exception error) {
        String message = error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
        return message.length() <= 512 ? message : message.substring(0, 512);
    }
}
