package com.macro.mall.portal.controller;

import com.macro.mall.common.api.CommonPage;
import com.macro.mall.common.api.CommonResult;
import com.macro.mall.portal.domain.AgentRefundAction;
import com.macro.mall.portal.domain.AgentCommerceRead;
import com.macro.mall.portal.domain.OmsOrderDetail;
import com.macro.mall.portal.service.AgentRefundActionService;
import com.macro.mall.portal.service.AgentCommerceReadService;
import com.macro.mall.portal.service.OmsPortalOrderService;
import com.macro.mall.portal.service.PmsPortalProductService;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentCommerceControllerTest {
    private final OmsPortalOrderService orderService = mock(OmsPortalOrderService.class);
    private final AgentRefundActionService refundActionService = mock(AgentRefundActionService.class);
    private final AgentCommerceReadService readService = mock(AgentCommerceReadService.class);
    private final PmsPortalProductService productService = mock(PmsPortalProductService.class);
    private final AgentCommerceController controller = new AgentCommerceController(orderService, refundActionService, readService, productService);

    @Test
    void listOrdersBoundsPaginationBeforeCallingService() {
        when(orderService.list(-1, 1, 50)).thenReturn(new CommonPage<>());

        CommonResult<CommonPage<OmsOrderDetail>> result = controller.listOrders(-1, 0, 999);

        assertEquals(200L, result.getCode());
        verify(orderService).list(-1, 1, 50);
    }

    @Test
    void previewDelegatesWithoutChangingOrderState() {
        AgentRefundAction.PreviewRequest request = new AgentRefundAction.PreviewRequest(1L, 2L, 1, "damaged", "a3e2394e-159c-4b60-88b1-d2a8a78a8d72");
        AgentRefundAction.PreviewResponse preview = new AgentRefundAction.PreviewResponse("token", "a".repeat(64), new BigDecimal("12.34"), Instant.now());
        when(refundActionService.preview(any())).thenReturn(preview);

        CommonResult<AgentRefundAction.PreviewResponse> result = controller.previewRefund(request, request.agentTraceId());

        assertEquals(preview, result.getData());
        verify(refundActionService).preview(request);
    }

    @Test
    void logisticsDelegatesToTheOwnershipCheckedReadService() {
        AgentCommerceRead.LogisticsResponse logistics = new AgentCommerceRead.LogisticsResponse(1L, 2, "DHL", "TRACK-1", null, null);
        when(readService.logistics(1L)).thenReturn(logistics);

        CommonResult<AgentCommerceRead.LogisticsResponse> result = controller.logistics(1L);

        assertEquals(logistics, result.getData());
        verify(readService).logistics(1L);
    }

    @Test
    void productSearchBoundsPageAndSortBeforeCallingService() {
        when(productService.search("shoe", null, null, 1, 20, 4)).thenReturn(List.of());

        CommonResult<?> result = controller.searchProducts("shoe", null, null, 0, 999, 99);

        assertEquals(200L, result.getCode());
        verify(productService).search("shoe", null, null, 1, 20, 4);
    }
}
