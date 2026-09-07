package com.macro.mall.portal.service;

import com.macro.mall.mapper.OmsOrderMapper;
import com.macro.mall.model.OmsOrder;
import com.macro.mall.portal.config.AlipayConfig;
import com.macro.mall.portal.service.impl.AlipayServiceImpl;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PaymentCallbackValidationTest {
    @Test
    void callbackMustMatchConfiguredAppAndOrderAmount() {
        OmsOrderMapper orderMapper = mock(OmsOrderMapper.class);
        OmsOrder order = new OmsOrder();
        order.setPayAmount(new BigDecimal("19.99"));
        when(orderMapper.selectByExample(any())).thenReturn(List.of(order));

        AlipayConfig config = new AlipayConfig();
        config.setAppId("app-1");
        AlipayServiceImpl service = new AlipayServiceImpl();
        ReflectionTestUtils.setField(service, "orderMapper", orderMapper);
        ReflectionTestUtils.setField(service, "alipayConfig", config);

        assertTrue(ReflectionTestUtils.invokeMethod(
                service, "callbackMatchesOrder", Map.of("app_id", "app-1", "out_trade_no", "SN1", "total_amount", "19.990")));
        assertFalse(ReflectionTestUtils.invokeMethod(
                service, "callbackMatchesOrder", Map.of("app_id", "app-1", "out_trade_no", "SN1", "total_amount", "0.01")));
        assertFalse(ReflectionTestUtils.invokeMethod(
                service, "callbackMatchesOrder", Map.of("app_id", "other-app", "out_trade_no", "SN1", "total_amount", "19.99")));
    }
}
