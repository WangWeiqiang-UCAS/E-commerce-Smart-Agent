package com.macro.mall.portal.controller;

import com.macro.mall.common.api.CommonPage;
import com.macro.mall.common.api.CommonResult;
import com.macro.mall.common.exception.Asserts;
import com.macro.mall.portal.domain.AgentRefundAction;
import com.macro.mall.portal.domain.AgentCommerceRead;
import com.macro.mall.portal.domain.AgentProduct;
import com.macro.mall.portal.domain.OmsOrderDetail;
import com.macro.mall.portal.domain.PmsPortalProductDetail;
import com.macro.mall.portal.service.AgentCommerceReadService;
import com.macro.mall.portal.service.AgentRefundActionService;
import com.macro.mall.portal.service.OmsPortalOrderService;
import com.macro.mall.portal.service.PmsPortalProductService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

import jakarta.validation.Valid;
import java.util.List;

/**
 * Narrow, authenticated read API for the agent runtime.
 *
 * <p>Do not expose internal portal controllers to an LLM. The underlying order
 * service derives the member from Spring Security, so a caller cannot select a
 * customer by supplying an ID.</p>
 */
@RestController
@RequestMapping("/agent/commerce")
@Tag(name = "AgentCommerceController", description = "Allow-listed commerce tools for the agent")
public class AgentCommerceController {

    private final OmsPortalOrderService orderService;
    private final AgentRefundActionService refundActionService;
    private final AgentCommerceReadService readService;
    private final PmsPortalProductService productService;

    public AgentCommerceController(OmsPortalOrderService orderService, AgentRefundActionService refundActionService,
                                   AgentCommerceReadService readService, PmsPortalProductService productService) {
        this.orderService = orderService;
        this.refundActionService = refundActionService;
        this.readService = readService;
        this.productService = productService;
    }

    @GetMapping("/orders")
    @Operation(summary = "List orders belonging to the authenticated member")
    public CommonResult<CommonPage<OmsOrderDetail>> listOrders(
            @RequestParam(defaultValue = "-1") Integer status,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "20") Integer pageSize) {
        int boundedPageSize = Math.min(Math.max(pageSize, 1), 50);
        return CommonResult.success(orderService.list(status, Math.max(pageNum, 1), boundedPageSize));
    }

    @GetMapping("/orders/{orderId}")
    @Operation(summary = "Get one order belonging to the authenticated member")
    public CommonResult<OmsOrderDetail> getOrder(@PathVariable Long orderId) {
        return CommonResult.success(orderService.detail(orderId));
    }

    @GetMapping("/orders/{orderId}/logistics")
    @Operation(summary = "Get logistics projection for an order owned by the authenticated member")
    public CommonResult<AgentCommerceRead.LogisticsResponse> logistics(@PathVariable Long orderId) {
        return CommonResult.success(readService.logistics(orderId));
    }

    @GetMapping("/after-sales/{afterSaleId}")
    @Operation(summary = "Get after-sales status after validating underlying order ownership")
    public CommonResult<AgentCommerceRead.AfterSaleStatusResponse> afterSaleStatus(@PathVariable Long afterSaleId) {
        return CommonResult.success(readService.afterSaleStatus(afterSaleId));
    }

    @GetMapping("/products/search")
    @Operation(summary = "Search public, published products for guided selling")
    public CommonResult<List<AgentProduct.Summary>> searchProducts(
            @RequestParam(required = false) String query,
            @RequestParam(required = false) Long brandId,
            @RequestParam(required = false) Long categoryId,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(defaultValue = "0") Integer sort) {
        if (query != null && query.trim().length() > 200) {
            Asserts.fail("Product search query is too long");
        }
        int boundedPageSize = Math.min(Math.max(pageSize, 1), 20);
        List<AgentProduct.Summary> products = productService.search(query == null ? null : query.trim(), brandId, categoryId,
                        Math.max(pageNum, 1), boundedPageSize, Math.min(Math.max(sort, 0), 4))
                .stream().map(AgentProduct.Summary::from).toList();
        return CommonResult.success(products);
    }

    @GetMapping("/products/{productId}")
    @Operation(summary = "Get a public product and its current SKU stock projection")
    public CommonResult<AgentProduct.Detail> productDetail(@PathVariable Long productId) {
        PmsPortalProductDetail detail = productService.detail(productId);
        if (detail == null || detail.getProduct() == null) {
            Asserts.fail("Product not found");
        }
        return CommonResult.success(new AgentProduct.Detail(AgentProduct.Summary.from(detail.getProduct()),
                (detail.getSkuStockList() == null ? List.<com.macro.mall.model.PmsSkuStock>of() : detail.getSkuStockList())
                        .stream().map(AgentProduct.Sku::from).toList()));
    }

    @PostMapping("/actions/refund/preview")
    @Operation(summary = "Create a short-lived refund preview; no commerce state is changed")
    public CommonResult<AgentRefundAction.PreviewResponse> previewRefund(
            @Valid @RequestBody AgentRefundAction.PreviewRequest request,
            @RequestHeader(value = "X-Trace-Id", required = false) String traceId) {
        requireMatchingTrace(traceId, request.agentTraceId());
        return CommonResult.success(refundActionService.preview(request));
    }

    @PostMapping("/actions/refund/confirm")
    @Operation(summary = "Confirm a preview and create an after-sales application; never refund funds")
    public CommonResult<AgentRefundAction.ConfirmResponse> confirmRefund(
            @Valid @RequestBody AgentRefundAction.ConfirmRequest request,
            @RequestHeader(value = "X-Trace-Id", required = false) String traceId) {
        requireMatchingTrace(traceId, request.agentTraceId());
        return CommonResult.success(refundActionService.confirm(request));
    }

    private void requireMatchingTrace(String headerTraceId, String bodyTraceId) {
        if (headerTraceId != null && !headerTraceId.equalsIgnoreCase(bodyTraceId)) {
            Asserts.fail("Trace ID must match the confirmed action");
        }
    }
}
