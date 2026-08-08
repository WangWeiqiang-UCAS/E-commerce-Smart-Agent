package com.macro.mall.portal.service.impl;

import com.macro.mall.common.exception.Asserts;
import com.macro.mall.mapper.OmsOrderItemMapper;
import com.macro.mall.mapper.OmsOrderMapper;
import com.macro.mall.model.OmsOrder;
import com.macro.mall.model.OmsOrderItem;
import com.macro.mall.model.OmsOrderItemExample;
import com.macro.mall.portal.domain.AgentRefundAction;
import com.macro.mall.portal.domain.OmsOrderReturnApplyParam;
import com.macro.mall.portal.service.AgentRefundActionService;
import com.macro.mall.portal.service.OmsPortalOrderReturnApplyService;
import com.macro.mall.portal.service.UmsMemberService;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class AgentRefundActionServiceImpl implements AgentRefundActionService {
    private static final SecureRandom RANDOM = new SecureRandom();
    private final JdbcTemplate jdbcTemplate;
    private final UmsMemberService memberService;
    private final OmsOrderMapper orderMapper;
    private final OmsOrderItemMapper orderItemMapper;
    private final OmsPortalOrderReturnApplyService returnApplyService;

    public AgentRefundActionServiceImpl(JdbcTemplate jdbcTemplate, UmsMemberService memberService,
            OmsOrderMapper orderMapper, OmsOrderItemMapper orderItemMapper,
            OmsPortalOrderReturnApplyService returnApplyService) {
        this.jdbcTemplate = jdbcTemplate;
        this.memberService = memberService;
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.returnApplyService = returnApplyService;
    }

    @Override
    @Transactional
    public AgentRefundAction.PreviewResponse preview(AgentRefundAction.PreviewRequest request) {
        Long memberId = memberService.getCurrentMember().getId();
        OmsOrderItem item = ownedOrderItem(memberId, request.orderId(), request.productId(), request.quantity());
        BigDecimal amount = item.getRealAmount()
                .divide(BigDecimal.valueOf(item.getProductQuantity()), 2, RoundingMode.DOWN)
                .multiply(BigDecimal.valueOf(request.quantity()));
        Instant expiresAt = Instant.now().plus(10, ChronoUnit.MINUTES);
        String token = randomToken();
        String previewHash = sha256(memberId + ":" + request.orderId() + ":" + request.productId() + ":"
                + request.quantity() + ":" + amount + ":" + request.reason());
        jdbcTemplate.update("INSERT INTO oms_agent_action (action_token_hash,member_id,action_type,order_id,product_id,quantity,reason,agent_trace_id,preview_hash,preview_amount,status,created_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,NOW(),?)",
                sha256(token), memberId, "REFUND", request.orderId(), request.productId(), request.quantity(),
                request.reason(), request.agentTraceId(), previewHash, amount, "PENDING",
                java.sql.Timestamp.from(expiresAt));
        return new AgentRefundAction.PreviewResponse(token, previewHash, amount, expiresAt);
    }

    @Override
    @Transactional
    public AgentRefundAction.ConfirmResponse confirm(AgentRefundAction.ConfirmRequest request) {
        Long memberId = memberService.getCurrentMember().getId();
        List<Map<String, Object>> actions = jdbcTemplate.queryForList("SELECT * FROM oms_agent_action WHERE action_token_hash=? AND member_id=? AND action_type='REFUND' FOR UPDATE", sha256(request.actionToken()), memberId);
        if (actions.isEmpty()) Asserts.fail("Refund preview not found");
        Map<String, Object> action = actions.get(0);
        if ("CONSUMED".equals(action.get("status")) && request.idempotencyKey().equals(action.get("idempotency_key"))) {
            return new AgentRefundAction.ConfirmResponse(((Number) action.get("id")).longValue(),
                    ((Number) action.get("after_sale_id")).longValue(), "APPLICATION_CREATED");
        }
        if (!"PENDING".equals(action.get("status")) || !request.previewHash().equals(action.get("preview_hash"))
                || !request.agentTraceId().equals(action.get("agent_trace_id"))
                || ((java.sql.Timestamp) action.get("expires_at")).toInstant().isBefore(Instant.now())) {
            Asserts.fail("Refund preview has expired or was already used");
        }
        int claimed = jdbcTemplate.update("UPDATE oms_agent_action SET idempotency_key=?, confirmed_at=NOW() WHERE id=? AND idempotency_key IS NULL", request.idempotencyKey(), action.get("id"));
        if (claimed != 1) Asserts.fail("Refund confirmation is already in progress");
        OmsOrderReturnApplyParam apply = new OmsOrderReturnApplyParam();
        apply.setOrderId(((Number) action.get("order_id")).longValue());
        apply.setProductId(((Number) action.get("product_id")).longValue());
        apply.setProductCount(((Number) action.get("quantity")).intValue());
        apply.setReason((String) action.get("reason"));
        Long afterSaleId = returnApplyService.createAndReturnId(apply);
        jdbcTemplate.update("UPDATE oms_agent_action SET status='CONSUMED', after_sale_id=?, consumed_at=NOW() WHERE id=?", afterSaleId, action.get("id"));
        Long actionId = ((Number) action.get("id")).longValue();
        String eventId = UUID.randomUUID().toString();
        String payload = "{\"eventId\":\"" + eventId + "\",\"eventType\":\"after_sale.created\",\"schemaVersion\":1,\"aggregateId\":\""
                + afterSaleId + "\",\"traceId\":\"" + request.agentTraceId() + "\",\"occurredAt\":\"" + Instant.now() + "\"}";
        jdbcTemplate.update("INSERT INTO oms_outbox_event (event_id,event_type,schema_version,aggregate_id,trace_id,payload,occurred_at) VALUES (?,?,?,?,?,?,NOW())",
                eventId, "after_sale.created", 1, afterSaleId.toString(), request.agentTraceId(), payload);
        return new AgentRefundAction.ConfirmResponse(actionId, afterSaleId, "APPLICATION_CREATED");
    }

    private OmsOrderItem ownedOrderItem(Long memberId, Long orderId, Long productId, Integer quantity) {
        OmsOrder order = orderMapper.selectByPrimaryKey(orderId);
        if (order == null || !memberId.equals(order.getMemberId())) Asserts.fail("Order not found");
        OmsOrderItemExample example = new OmsOrderItemExample();
        example.createCriteria().andOrderIdEqualTo(orderId).andProductIdEqualTo(productId);
        List<OmsOrderItem> items = orderItemMapper.selectByExample(example);
        if (items.size() != 1 || quantity > items.get(0).getProductQuantity()) Asserts.fail("Order item not found");
        return items.get(0);
    }

    private String randomToken() { byte[] bytes = new byte[32]; RANDOM.nextBytes(bytes); return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes); }
    private String sha256(String value) { try { byte[] hash = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)); StringBuilder result = new StringBuilder(); for (byte b : hash) result.append(String.format("%02x", b)); return result.toString(); } catch (Exception e) { throw new IllegalStateException(e); } }
}
