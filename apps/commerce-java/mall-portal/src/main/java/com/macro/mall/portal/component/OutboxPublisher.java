package com.macro.mall.portal.component;

import com.macro.mall.portal.config.RabbitMqConfig;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

/** Publishes committed domain events; consumers must deduplicate by eventId. */
@Component
public class OutboxPublisher {
    private final JdbcTemplate jdbcTemplate;
    private final RabbitTemplate rabbitTemplate;

    public OutboxPublisher(JdbcTemplate jdbcTemplate, RabbitTemplate rabbitTemplate) {
        this.jdbcTemplate = jdbcTemplate;
        this.rabbitTemplate = rabbitTemplate;
    }

    @Scheduled(fixedDelayString = "${outbox.publish-delay-ms:1000}")
    @Transactional
    public void publish() {
        List<Map<String, Object>> events = jdbcTemplate.queryForList(
                "SELECT id,event_type,payload FROM oms_outbox_event WHERE published_at IS NULL ORDER BY id LIMIT 50 FOR UPDATE");
        for (Map<String, Object> event : events) {
            try {
                rabbitTemplate.convertAndSend(RabbitMqConfig.AGENT_EVENTS_EXCHANGE,
                        (String) event.get("event_type"), event.get("payload").toString());
                jdbcTemplate.update("UPDATE oms_outbox_event SET published_at=NOW(), attempts=attempts+1 WHERE id=?", event.get("id"));
            } catch (RuntimeException error) {
                jdbcTemplate.update("UPDATE oms_outbox_event SET attempts=attempts+1 WHERE id=?", event.get("id"));
                throw error;
            }
        }
    }
}
