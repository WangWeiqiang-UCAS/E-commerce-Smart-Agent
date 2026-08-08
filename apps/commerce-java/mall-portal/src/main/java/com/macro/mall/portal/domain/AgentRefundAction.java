package com.macro.mall.portal.domain;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

import java.math.BigDecimal;
import java.time.Instant;

public final class AgentRefundAction {
    private AgentRefundAction() { }

    public record PreviewRequest(
            @NotNull Long orderId,
            @NotNull Long productId,
            @NotNull @Min(1) @Max(999) Integer quantity,
            @NotBlank String reason,
            @NotBlank @Pattern(regexp = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$") String agentTraceId) { }

    public record PreviewResponse(String actionToken, String previewHash, BigDecimal amount, Instant expiresAt) { }

    public record ConfirmRequest(
            @NotBlank String actionToken,
            @NotBlank String idempotencyKey,
            @NotBlank String previewHash,
            @NotBlank @Pattern(regexp = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$") String agentTraceId,
            @NotNull Instant confirmedAt) { }

    public record ConfirmResponse(Long actionId, Long afterSaleId, String status) { }
}
