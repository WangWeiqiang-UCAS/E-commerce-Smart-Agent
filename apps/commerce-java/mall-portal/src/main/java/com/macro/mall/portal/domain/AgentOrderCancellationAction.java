package com.macro.mall.portal.domain;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

import java.time.Instant;

/** Short-lived, customer-confirmed cancellation action payloads for the Agent boundary. */
public final class AgentOrderCancellationAction {
    private AgentOrderCancellationAction() { }

    public record PreviewRequest(
            @NotNull Long orderId,
            @NotBlank @Pattern(regexp = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$") String agentTraceId) { }

    public record PreviewResponse(String actionToken, String previewHash, Instant expiresAt) { }

    public record ConfirmRequest(
            @NotBlank String actionToken,
            @NotBlank @Pattern(regexp = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$") String idempotencyKey,
            @NotBlank String previewHash,
            @NotBlank @Pattern(regexp = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$") String agentTraceId,
            @NotNull Instant confirmedAt) { }

    public record ConfirmResponse(Long actionId, Long orderId, String status) { }
}
