package com.macro.mall.portal.component;

import com.macro.mall.portal.config.RabbitMqConfig;
import org.springframework.amqp.rabbit.connection.CorrelationData;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/** Publishes committed domain events; consumers must deduplicate by eventId. */
@Component
public class OutboxPublisher {
    private static final Logger LOGGER = LoggerFactory.getLogger(OutboxPublisher.class);
    private static final int MAX_EVENTS_PER_RUN = 50;
    private final JdbcTemplate jdbcTemplate;
    private final RabbitTemplate rabbitTemplate;
    private final String publisherId;

    public OutboxPublisher(JdbcTemplate jdbcTemplate, RabbitTemplate rabbitTemplate) {
        this(jdbcTemplate, rabbitTemplate, UUID.randomUUID().toString());
    }

    OutboxPublisher(JdbcTemplate jdbcTemplate, RabbitTemplate rabbitTemplate, String publisherId) {
        this.jdbcTemplate = jdbcTemplate;
        this.rabbitTemplate = rabbitTemplate;
        this.publisherId = publisherId;
    }

    @Scheduled(fixedDelayString = "${outbox.publish-delay-ms:1000}")
    public void publish() {
        for (int index = 0; index < MAX_EVENTS_PER_RUN; index++) {
            Map<String, Object> event = claimNextEvent();
            if (event == null) {
                return;
            }
            try {
                publishConfirmed(event);
                if (jdbcTemplate.update("UPDATE oms_outbox_event SET published_at=NOW(), attempts=attempts+1, "
                        + "publish_lock_owner=NULL, publish_locked_at=NULL WHERE id=? AND publish_lock_owner=?",
                        event.get("id"), publisherId) != 1) {
                    LOGGER.warn("Outbox event {} lease expired before it could be marked published", event.get("id"));
                }
            } catch (Exception error) {
                jdbcTemplate.update("UPDATE oms_outbox_event SET attempts=attempts+1, publish_lock_owner=NULL, "
                        + "publish_locked_at=NULL WHERE id=? AND publish_lock_owner=?", event.get("id"), publisherId);
                LOGGER.warn("Outbox event {} was not broker-confirmed; it will be retried", event.get("id"), error);
                if (error instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                }
                // Preserve event order and avoid flooding the broker while it is unavailable.
                return;
            }
        }
    }

    /**
     * Atomically lease one unpublished row. The lease expires so a crashed
     * publisher cannot strand an event; duplicate publication is harmless
     * because consumers deduplicate on eventId in their inbox.
     */
    private Map<String, Object> claimNextEvent() {
        int claimed = jdbcTemplate.update(
                "UPDATE oms_outbox_event SET publish_lock_owner=?, publish_locked_at=NOW() "
                        + "WHERE published_at IS NULL AND (publish_locked_at IS NULL "
                        + "OR publish_locked_at < DATE_SUB(NOW(), INTERVAL 60 SECOND)) AND id=("
                        + "SELECT id FROM (SELECT id FROM oms_outbox_event WHERE published_at IS NULL "
                        + "AND (publish_locked_at IS NULL OR publish_locked_at < DATE_SUB(NOW(), INTERVAL 60 SECOND)) "
                        + "ORDER BY id LIMIT 1) AS claimable)",
                publisherId);
        if (claimed != 1) {
            return null;
        }
        List<Map<String, Object>> events = jdbcTemplate.queryForList(
                "SELECT id,event_type,payload FROM oms_outbox_event WHERE published_at IS NULL "
                        + "AND publish_lock_owner=? ORDER BY publish_locked_at DESC,id LIMIT 1",
                publisherId);
        return events.isEmpty() ? null : events.get(0);
    }

    /** A broker ack is required before the event is considered published. */
    private void publishConfirmed(Map<String, Object> event) throws Exception {
        CorrelationData correlation = new CorrelationData("outbox-" + event.get("id"));
        rabbitTemplate.convertAndSend(RabbitMqConfig.AGENT_EVENTS_EXCHANGE,
                (String) event.get("event_type"), event.get("payload").toString(), correlation);
        CorrelationData.Confirm confirm = correlation.getFuture().get(5, TimeUnit.SECONDS);
        if (confirm == null || !confirm.isAck()) {
            String reason = confirm == null ? "publisher confirm timed out" : confirm.getReason();
            throw new IllegalStateException("RabbitMQ did not acknowledge outbox event " + event.get("id") + ": " + reason);
        }
    }
}
