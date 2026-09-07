package com.macro.mall.portal.component;

import com.macro.mall.portal.config.RabbitMqConfig;
import org.junit.jupiter.api.Test;
import org.springframework.amqp.rabbit.connection.CorrelationData;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.jdbc.core.JdbcTemplate;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.startsWith;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OutboxPublisherTest {
    private final JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
    private final RabbitTemplate rabbitTemplate = mock(RabbitTemplate.class);
    private final OutboxPublisher publisher = new OutboxPublisher(jdbcTemplate, rabbitTemplate, "test-publisher");

    @Test
    void marksEventPublishedOnlyAfterBrokerAck() {
        when(jdbcTemplate.queryForList(anyString(), any(Object[].class))).thenReturn(List.of(event()));
        when(jdbcTemplate.update(startsWith("UPDATE oms_outbox_event SET publish_lock_owner"), anyString()))
                .thenReturn(1, 0);
        doAnswer(invocation -> {
            CorrelationData correlation = invocation.getArgument(3);
            correlation.getFuture().complete(new CorrelationData.Confirm(true, null));
            return null;
        }).when(rabbitTemplate).convertAndSend(anyString(), anyString(), any(), any(CorrelationData.class));

        publisher.publish();

        verify(rabbitTemplate).convertAndSend(RabbitMqConfig.AGENT_EVENTS_EXCHANGE, "after_sale.created", "{\"eventId\":\"event-1\"}", any(CorrelationData.class));
        verify(jdbcTemplate).update("UPDATE oms_outbox_event SET published_at=NOW(), attempts=attempts+1, "
                + "publish_lock_owner=NULL, publish_locked_at=NULL WHERE id=? AND publish_lock_owner=?", 7L, "test-publisher");
    }

    @Test
    void retainsEventForRetryWhenBrokerNacks() {
        when(jdbcTemplate.queryForList(anyString(), any(Object[].class))).thenReturn(List.of(event()));
        when(jdbcTemplate.update(startsWith("UPDATE oms_outbox_event SET publish_lock_owner"), anyString()))
                .thenReturn(1);
        doAnswer(invocation -> {
            CorrelationData correlation = invocation.getArgument(3);
            correlation.getFuture().complete(new CorrelationData.Confirm(false, "exchange unavailable"));
            return null;
        }).when(rabbitTemplate).convertAndSend(anyString(), anyString(), any(), any(CorrelationData.class));

        publisher.publish();

        verify(jdbcTemplate).update("UPDATE oms_outbox_event SET attempts=attempts+1, publish_lock_owner=NULL, "
                + "publish_locked_at=NULL WHERE id=? AND publish_lock_owner=?", 7L, "test-publisher");
        verify(jdbcTemplate, never()).update(startsWith("UPDATE oms_outbox_event SET published_at=NOW()"), any(), any());
    }

    private Map<String, Object> event() {
        return Map.of("id", 7L, "event_type", "after_sale.created", "payload", "{\"eventId\":\"event-1\"}");
    }
}
