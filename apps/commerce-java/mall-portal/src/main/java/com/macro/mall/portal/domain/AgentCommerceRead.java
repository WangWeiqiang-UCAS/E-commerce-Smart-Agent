package com.macro.mall.portal.domain;

import java.math.BigDecimal;
import java.util.Date;

/** Deliberately small, non-PII projections exposed to the agent boundary. */
public final class AgentCommerceRead {
    private AgentCommerceRead() { }

    public record LogisticsResponse(Long orderId, Integer orderStatus, String deliveryCompany,
                                   String deliverySn, Date deliveryTime, Date receiveTime) { }

    public record AfterSaleStatusResponse(Long afterSaleId, Long orderId, String orderSn,
                                          Long productId, Integer status, BigDecimal returnAmount,
                                          Date createdAt, Date handledAt) { }
}
