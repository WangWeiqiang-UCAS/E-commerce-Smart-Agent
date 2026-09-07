package com.macro.mall.portal.controller;

import com.macro.mall.common.api.CommonPage;
import com.macro.mall.common.api.CommonResult;
import com.macro.mall.portal.domain.AgentRefundAction;
import com.macro.mall.portal.domain.AgentOrderCancellationAction;
import com.macro.mall.portal.domain.AgentCommerceRead;
import com.macro.mall.portal.domain.OmsOrderDetail;
import com.macro.mall.portal.service.AgentRefundActionService;
import com.macro.mall.portal.service.AgentOrderCancellationActionService;
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
    private final AgentOrderCancellationActionService cancellationActionService = mock(AgentOrderCancellationActionService.class);
    private final AgentCommerceReadService readService = mock(AgentCommerceReadService.class);
    private final PmsPortalProductService productService = mock(PmsPortalProductService.class);
    private final AgentCommerceController controller = new AgentCommerceController(orderService, refundActionService, cancellationActionService, readService, productService);

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
    void orderCancellationPreviewAndConfirmDelegateToTheActionLedger() {
        String traceId = "a3e2394e-159c-4b60-88b1-d2a8a78a8d72";
        AgentOrderCancellationAction.PreviewRequest previewRequest = new AgentOrderCancellationAction.PreviewRequest(1L, traceId);
        AgentOrderCancellationAction.PreviewResponse preview = new AgentOrderCancellationAction.PreviewResponse("token", "a".repeat(64), Instant.now());
        AgentOrderCancellationAction.ConfirmRequest confirmRequest = new AgentOrderCancellationAction.ConfirmRequest("token", "d0f994ad-7ac1-46ed-9dab-c0d8ee8a7d14", "a".repeat(64), traceId, Instant.now());
        AgentOrderCancellationAction.ConfirmResponse confirmed = new AgentOrderCancellationAction.ConfirmResponse(3L, 1L, "CANCELLED");
        when(cancellationActionService.preview(previewRequest)).thenReturn(preview);
        when(cancellationActionService.confirm(confirmRequest)).thenReturn(confirmed);

        assertEquals(preview, controller.previewOrderCancellation(previewRequest, traceId).getData());
        assertEquals(confirmed, controller.confirmOrderCancellation(confirmRequest, traceId).getData());
        verify(cancellationActionService).preview(previewRequest);
        verify(cancellationActionService).confirm(confirmRequest);
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
    void identityIsDerivedByCommerceSecurityContext() {
        AgentCommerceRead.MemberIdentity identity = new AgentCommerceRead.MemberIdentity(7L, "member");
        when(readService.currentMemberIdentity()).thenReturn(identity);

        CommonResult<AgentCommerceRead.MemberIdentity> result = controller.identity();

        assertEquals(identity, result.getData());
        verify(readService).currentMemberIdentity();
    }

    @Test
    void productSearchBoundsPageAndSortBeforeCallingService() {
        when(productService.search("shoe", null, null, 1, 20, 4)).thenReturn(List.of());

        CommonResult<?> result = controller.searchProducts("shoe", null, null, 0, 999, 99);

        assertEquals(200L, result.getCode());
        verify(productService).search("shoe", null, null, 1, 20, 4);
    }
}
