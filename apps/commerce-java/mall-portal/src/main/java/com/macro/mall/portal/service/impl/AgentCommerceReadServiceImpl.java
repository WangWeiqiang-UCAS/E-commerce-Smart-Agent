package com.macro.mall.portal.service.impl;

import com.macro.mall.common.exception.Asserts;
import com.macro.mall.mapper.OmsOrderMapper;
import com.macro.mall.mapper.OmsOrderReturnApplyMapper;
import com.macro.mall.model.OmsOrder;
import com.macro.mall.model.OmsOrderReturnApply;
import com.macro.mall.portal.domain.AgentCommerceRead;
import com.macro.mall.portal.service.AgentCommerceReadService;
import com.macro.mall.portal.service.UmsMemberService;
import org.springframework.stereotype.Service;

@Service
public class AgentCommerceReadServiceImpl implements AgentCommerceReadService {
    private final OmsOrderMapper orderMapper;
    private final OmsOrderReturnApplyMapper returnApplyMapper;
    private final UmsMemberService memberService;

    public AgentCommerceReadServiceImpl(OmsOrderMapper orderMapper,
                                        OmsOrderReturnApplyMapper returnApplyMapper,
                                        UmsMemberService memberService) {
        this.orderMapper = orderMapper;
        this.returnApplyMapper = returnApplyMapper;
        this.memberService = memberService;
    }

    @Override
    public AgentCommerceRead.MemberIdentity currentMemberIdentity() {
        var member = memberService.getCurrentMember();
        return new AgentCommerceRead.MemberIdentity(member.getId(), member.getUsername());
    }

    @Override
    public AgentCommerceRead.LogisticsResponse logistics(Long orderId) {
        OmsOrder order = ownedOrder(orderId);
        return new AgentCommerceRead.LogisticsResponse(order.getId(), order.getStatus(),
                order.getDeliveryCompany(), order.getDeliverySn(), order.getDeliveryTime(), order.getReceiveTime());
    }

    @Override
    public AgentCommerceRead.AfterSaleStatusResponse afterSaleStatus(Long afterSaleId) {
        OmsOrderReturnApply application = returnApplyMapper.selectByPrimaryKey(afterSaleId);
        if (application == null) {
            Asserts.fail("After-sales application not found");
        }
        ownedOrder(application.getOrderId());
        return new AgentCommerceRead.AfterSaleStatusResponse(application.getId(), application.getOrderId(),
                application.getOrderSn(), application.getProductId(), application.getStatus(),
                application.getReturnAmount(), application.getCreateTime(), application.getHandleTime());
    }

    private OmsOrder ownedOrder(Long orderId) {
        OmsOrder order = orderMapper.selectByPrimaryKey(orderId);
        if (order == null || !memberService.getCurrentMember().getId().equals(order.getMemberId())) {
            Asserts.fail("Order not found");
        }
        return order;
    }
}
