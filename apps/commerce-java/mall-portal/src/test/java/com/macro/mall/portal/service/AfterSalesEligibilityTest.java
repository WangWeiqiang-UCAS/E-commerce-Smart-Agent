package com.macro.mall.portal.service;

import com.macro.mall.common.exception.ApiException;
import com.macro.mall.mapper.OmsOrderItemMapper;
import com.macro.mall.mapper.OmsOrderMapper;
import com.macro.mall.mapper.OmsOrderReturnApplyMapper;
import com.macro.mall.model.OmsOrder;
import com.macro.mall.model.OmsOrderItem;
import com.macro.mall.model.OmsOrderReturnApply;
import com.macro.mall.model.UmsMember;
import com.macro.mall.portal.domain.OmsOrderReturnApplyParam;
import com.macro.mall.portal.service.impl.OmsPortalOrderReturnApplyServiceImpl;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AfterSalesEligibilityTest {
    @Test
    void rejectsAfterSalesForAnUnpaidOrder() {
        OmsOrder order = order(0);
        OmsPortalOrderReturnApplyServiceImpl service = serviceFor(order);

        assertThrows(ApiException.class, () -> service.createAndReturnId(request(1)));
    }

    @Test
    void rejectsQuantityAlreadyCoveredByOpenAfterSales() {
        OmsOrder order = order(1);
        OmsOrderItem item = new OmsOrderItem();
        item.setProductQuantity(2);
        OmsOrderReturnApply existing = new OmsOrderReturnApply();
        existing.setProductCount(2);
        existing.setStatus(0);

        OmsPortalOrderReturnApplyServiceImpl service = serviceFor(order);
        OmsOrderItemMapper itemMapper = mock(OmsOrderItemMapper.class);
        OmsOrderReturnApplyMapper returnMapper = mock(OmsOrderReturnApplyMapper.class);
        when(itemMapper.selectByExample(any())).thenReturn(List.of(item));
        when(returnMapper.selectByExample(any())).thenReturn(List.of(existing));
        ReflectionTestUtils.setField(service, "orderItemMapper", itemMapper);
        ReflectionTestUtils.setField(service, "returnApplyMapper", returnMapper);

        assertThrows(ApiException.class, () -> service.createAndReturnId(request(1)));
    }

    private OmsPortalOrderReturnApplyServiceImpl serviceFor(OmsOrder order) {
        OmsOrderMapper orderMapper = mock(OmsOrderMapper.class);
        UmsMemberService memberService = mock(UmsMemberService.class);
        UmsMember member = new UmsMember();
        member.setId(9L);
        when(orderMapper.selectByPrimaryKey(1L)).thenReturn(order);
        when(memberService.getCurrentMember()).thenReturn(member);
        OmsPortalOrderReturnApplyServiceImpl service = new OmsPortalOrderReturnApplyServiceImpl();
        ReflectionTestUtils.setField(service, "orderMapper", orderMapper);
        ReflectionTestUtils.setField(service, "memberService", memberService);
        return service;
    }

    private OmsOrder order(int status) {
        OmsOrder order = new OmsOrder();
        order.setId(1L);
        order.setMemberId(9L);
        order.setStatus(status);
        return order;
    }

    private OmsOrderReturnApplyParam request(int quantity) {
        OmsOrderReturnApplyParam request = new OmsOrderReturnApplyParam();
        request.setOrderId(1L);
        request.setProductId(2L);
        request.setProductCount(quantity);
        return request;
    }
}
