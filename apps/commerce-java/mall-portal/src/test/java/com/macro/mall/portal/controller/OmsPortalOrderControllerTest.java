package com.macro.mall.portal.controller;

import com.macro.mall.common.api.CommonResult;
import com.macro.mall.portal.service.OmsPortalOrderService;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class OmsPortalOrderControllerTest {
    @Test
    void directPaymentSuccessEndpointIsRejected() {
        OmsPortalOrderController controller = new OmsPortalOrderController();

        CommonResult result = controller.paySuccess(9L, 1);

        assertEquals(500L, result.getCode());
    }

    @Test
    void browserCannotInvokeDirectCancellationEndpoints() {
        OmsPortalOrderController controller = new OmsPortalOrderController();
        OmsPortalOrderService service = mock(OmsPortalOrderService.class);
        ReflectionTestUtils.setField(controller, "portalOrderService", service);

        CommonResult delayed = controller.cancelOrder(9L);
        CommonResult immediate = controller.cancelUserOrder(10L);

        assertEquals(500L, delayed.getCode());
        assertEquals(500L, immediate.getCode());
        verify(service, never()).sendDelayMessageCancelOrder(9L);
        verify(service, never()).cancelOrderForCurrentMember(10L);
    }

    @Test
    void browserCannotInvokeBulkTimeoutCancellation() {
        OmsPortalOrderController controller = new OmsPortalOrderController();
        OmsPortalOrderService service = mock(OmsPortalOrderService.class);
        ReflectionTestUtils.setField(controller, "portalOrderService", service);

        CommonResult result = controller.cancelTimeOutOrder();

        assertEquals(500L, result.getCode());
        verify(service, never()).cancelTimeOutOrder();
    }
}
