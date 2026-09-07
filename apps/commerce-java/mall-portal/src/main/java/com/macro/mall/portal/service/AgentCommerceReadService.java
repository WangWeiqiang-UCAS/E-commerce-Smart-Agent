package com.macro.mall.portal.service;

import com.macro.mall.portal.domain.AgentCommerceRead;

/** Authenticated, allow-listed read projections for the agent. */
public interface AgentCommerceReadService {
    AgentCommerceRead.MemberIdentity currentMemberIdentity();

    AgentCommerceRead.LogisticsResponse logistics(Long orderId);

    AgentCommerceRead.AfterSaleStatusResponse afterSaleStatus(Long afterSaleId);
}
