package com.macro.mall.portal.service.impl;

import com.macro.mall.common.exception.Asserts;
import com.macro.mall.mapper.OmsOrderMapper;
import com.macro.mall.model.OmsOrder;
import com.macro.mall.portal.domain.AgentOrderCancellationAction;
import com.macro.mall.portal.service.AgentOrderCancellationActionService;
import com.macro.mall.portal.service.OmsPortalOrderService;
import com.macro.mall.portal.service.UmsMemberService;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Implements preview-confirm-execute without exposing a direct browser cancel operation. */
@Service
public class AgentOrderCancellationActionServiceImpl implements AgentOrderCancellationActionService {
    private static final SecureRandom RANDOM = new SecureRandom();
    private final JdbcTemplate jdbcTemplate;
    private final UmsMemberService memberService;
    private final OmsOrderMapper orderMapper;
    private final OmsPortalOrderService orderService;

    public AgentOrderCancellationActionServiceImpl(JdbcTemplate jdbcTemplate, UmsMemberService memberService,
                                                   OmsOrderMapper orderMapper, OmsPortalOrderService orderService) {
        this.jdbcTemplate = jdbcTemplate;
        this.memberService = memberService;
        this.orderMapper = orderMapper;
        this.orderService = orderService;
    }

    @Override
    @Transactional
    public AgentOrderCancellationAction.PreviewResponse preview(AgentOrderCancellationAction.PreviewRequest request) {
        Long memberId = memberService.getCurrentMember().getId();
        OmsOrder order = cancellableOwnedOrder(memberId, request.orderId());
        Instant expiresAt = Instant.now().plus(10, ChronoUnit.MINUTES);
        String token = randomToken();
        String previewHash = sha256(memberId + ":" + order.getId() + ":" + order.getStatus() + ":" + request.agentTraceId());
        jdbcTemplate.update("INSERT INTO oms_agent_action (action_token_hash,member_id,action_type,order_id,agent_trace_id,preview_hash,status,created_at,expires_at,order_status_at_preview) VALUES (?,?,?,?,?,?,?,NOW(),?,?)",
                sha256(token), memberId, "ORDER_CANCEL", order.getId(), request.agentTraceId(), previewHash,
                "PENDING", java.sql.Timestamp.from(expiresAt), order.getStatus());
        return new AgentOrderCancellationAction.PreviewResponse(token, previewHash, expiresAt);
    }

    @Override
    @Transactional(isolation = Isolation.SERIALIZABLE)
    public AgentOrderCancellationAction.ConfirmResponse confirm(AgentOrderCancellationAction.ConfirmRequest request) {
        Long memberId = memberService.getCurrentMember().getId();
        List<Map<String, Object>> actions = jdbcTemplate.queryForList(
                "SELECT * FROM oms_agent_action WHERE action_token_hash=? AND member_id=? AND action_type='ORDER_CANCEL' FOR UPDATE",
                sha256(request.actionToken()), memberId);
        if (actions.isEmpty()) Asserts.fail("Order cancellation preview not found");
        Map<String, Object> action = actions.get(0);
        Long actionId = ((Number) action.get("id")).longValue();
        Long orderId = ((Number) action.get("order_id")).longValue();
        if ("CONSUMED".equals(action.get("status")) && request.idempotencyKey().equals(action.get("idempotency_key"))) {
            return new AgentOrderCancellationAction.ConfirmResponse(actionId, orderId, "CANCELLED");
        }
        if (!"PENDING".equals(action.get("status")) || !request.previewHash().equals(action.get("preview_hash"))
                || !request.agentTraceId().equals(action.get("agent_trace_id"))
                || ((java.sql.Timestamp) action.get("expires_at")).toInstant().isBefore(Instant.now())) {
            Asserts.fail("Order cancellation preview has expired or was already used");
        }
        int claimed = jdbcTemplate.update("UPDATE oms_agent_action SET idempotency_key=?, confirmed_at=? WHERE id=? AND idempotency_key IS NULL",
                request.idempotencyKey(), java.sql.Timestamp.from(request.confirmedAt()), actionId);
        if (claimed != 1) Asserts.fail("Order cancellation is already in progress");
        OmsOrder order = cancellableOwnedOrder(memberId, orderId);
        if (!order.getStatus().equals(((Number) action.get("order_status_at_preview")).intValue())
                || !orderService.cancelOrderForCurrentMember(orderId)) {
            Asserts.fail("Order is no longer eligible for cancellation");
        }
        jdbcTemplate.update("UPDATE oms_agent_action SET status='CONSUMED', consumed_at=NOW() WHERE id=?", actionId);
        String eventId = UUID.randomUUID().toString();
        String payload = "{\"eventId\":\"" + eventId + "\",\"eventType\":\"order.status_changed\",\"schemaVersion\":1,\"aggregateId\":\""
                + orderId + "\",\"orderStatus\":4,\"traceId\":\"" + request.agentTraceId() + "\",\"occurredAt\":\"" + Instant.now() + "\"}";
        jdbcTemplate.update("INSERT INTO oms_outbox_event (event_id,event_type,schema_version,aggregate_id,trace_id,payload,occurred_at) VALUES (?,?,?,?,?,?,NOW())",
                eventId, "order.status_changed", 1, orderId.toString(), request.agentTraceId(), payload);
        return new AgentOrderCancellationAction.ConfirmResponse(actionId, orderId, "CANCELLED");
    }

    private OmsOrder cancellableOwnedOrder(Long memberId, Long orderId) {
        OmsOrder order = orderMapper.selectByPrimaryKey(orderId);
        if (order == null || !memberId.equals(order.getMemberId())) Asserts.fail("Order not found");
        if (!Integer.valueOf(0).equals(order.getStatus()) || !Integer.valueOf(0).equals(order.getDeleteStatus())) {
            Asserts.fail("Only pending orders can be cancelled");
        }
        return order;
    }

    private String randomToken() { byte[] bytes = new byte[32]; RANDOM.nextBytes(bytes); return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes); }
    private String sha256(String value) { try { byte[] hash = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)); StringBuilder result = new StringBuilder(); for (byte b : hash) result.append(String.format("%02x", b)); return result.toString(); } catch (Exception e) { throw new IllegalStateException(e); } }
}
