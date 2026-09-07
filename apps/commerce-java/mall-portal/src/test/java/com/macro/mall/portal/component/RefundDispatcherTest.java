package com.macro.mall.portal.component;

import com.macro.mall.portal.service.AlipayRefundGateway;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.SimpleTransactionStatus;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.startsWith;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RefundDispatcherTest {
    @Test
    void settlesProviderAcceptedRefundThenWritesCompletionOutboxEvent() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        AlipayRefundGateway gateway = mock(AlipayRefundGateway.class);
        PlatformTransactionManager transactionManager = mock(PlatformTransactionManager.class);
        when(transactionManager.getTransaction(any())).thenReturn(new SimpleTransactionStatus());
        doNothing().when(transactionManager).commit(any());

        when(jdbcTemplate.update(startsWith("UPDATE oms_payment_refund SET status='PROCESSING'"), anyString()))
                .thenReturn(1);
        when(jdbcTemplate.queryForList(startsWith("SELECT r.id,r.after_sale_id"), anyString()))
                .thenReturn(List.of(Map.of(
                        "id", 7L, "after_sale_id", 42L, "payment_channel", "ALIPAY",
                        "refund_request_no", "after-sale-42", "refund_amount", new BigDecimal("12.34"),
                        "order_sn", "202608230001")));
        when(gateway.refund("after-sale-42", "202608230001", new BigDecimal("12.34"))).thenReturn("alipay-trade-7");
        when(jdbcTemplate.update(startsWith("UPDATE oms_payment_refund SET status='SUCCEEDED'"),
                any(), any(), any())).thenReturn(1);
        when(jdbcTemplate.update(startsWith("INSERT INTO oms_outbox_event"),
                any(), any(), any(), any(), any(), any())).thenReturn(1);

        new RefundDispatcher(jdbcTemplate, gateway, transactionManager).dispatch();

        verify(gateway).refund("after-sale-42", "202608230001", new BigDecimal("12.34"));
        verify(jdbcTemplate).update(startsWith("UPDATE oms_payment_refund SET status='SUCCEEDED'"),
                any(), any(), any());
        verify(jdbcTemplate).update(startsWith("INSERT INTO oms_outbox_event"),
                any(), any(), any(), any(), any(), any());
    }
}
