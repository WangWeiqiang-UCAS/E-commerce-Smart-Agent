package com.macro.mall.portal.service.impl;

import com.macro.mall.common.exception.Asserts;
import com.macro.mall.mapper.OmsOrderItemMapper;
import com.macro.mall.mapper.OmsOrderMapper;
import com.macro.mall.mapper.OmsOrderReturnApplyMapper;
import com.macro.mall.model.OmsOrder;
import com.macro.mall.model.OmsOrderItem;
import com.macro.mall.model.OmsOrderItemExample;
import com.macro.mall.model.OmsOrderReturnApply;
import com.macro.mall.model.OmsOrderReturnApplyExample;
import com.macro.mall.portal.domain.OmsOrderReturnApplyParam;
import com.macro.mall.portal.service.OmsPortalOrderReturnApplyService;
import com.macro.mall.portal.service.UmsMemberService;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.annotation.Isolation;

import java.math.RoundingMode;
import java.util.Date;
import java.util.List;

/**
 * 订单退货管理Service实现类
 * Created by macro on 2018/10/17.
 */
@Service
public class OmsPortalOrderReturnApplyServiceImpl implements OmsPortalOrderReturnApplyService {
    @Autowired
    private OmsOrderReturnApplyMapper returnApplyMapper;

    @Autowired
    private OmsOrderMapper orderMapper;

    @Autowired
    private OmsOrderItemMapper orderItemMapper;

    @Autowired
    private UmsMemberService memberService;

    @Override
    @Transactional(isolation = Isolation.SERIALIZABLE)
    public int create(OmsOrderReturnApplyParam returnApply) {
        createAndReturnId(returnApply);
        return 1;
    }

    @Override
    @Transactional(isolation = Isolation.SERIALIZABLE)
    public Long createAndReturnId(OmsOrderReturnApplyParam returnApply) {
        if (returnApply.getOrderId() == null || returnApply.getProductId() == null
                || returnApply.getProductCount() == null || returnApply.getProductCount() < 1) {
            Asserts.fail("Invalid return request");
        }
        OmsOrder order = orderMapper.selectByPrimaryKey(returnApply.getOrderId());
        if (order == null || !memberService.getCurrentMember().getId().equals(order.getMemberId())) {
            Asserts.fail("Order not found");
        }
        if (order.getStatus() == null || order.getStatus() < 1 || order.getStatus() > 3) {
            Asserts.fail("Only paid orders can enter after-sales processing");
        }
        OmsOrderItemExample itemExample = new OmsOrderItemExample();
        itemExample.createCriteria()
                .andOrderIdEqualTo(order.getId())
                .andProductIdEqualTo(returnApply.getProductId());
        List<OmsOrderItem> orderItems = orderItemMapper.selectByExample(itemExample);
        if (orderItems.size() != 1) {
            Asserts.fail("Order item not found");
        }
        OmsOrderItem orderItem = orderItems.get(0);
        if (returnApply.getProductCount() > orderItem.getProductQuantity()) {
            Asserts.fail("Return quantity exceeds purchased quantity");
        }
        OmsOrderReturnApplyExample applyExample = new OmsOrderReturnApplyExample();
        applyExample.createCriteria()
                .andOrderIdEqualTo(order.getId())
                .andProductIdEqualTo(returnApply.getProductId());
        int alreadyApplied = returnApplyMapper.selectByExample(applyExample).stream()
                .filter(application -> application.getStatus() == null || application.getStatus() != 3)
                .mapToInt(application -> application.getProductCount() == null ? 0 : application.getProductCount())
                .sum();
        if (alreadyApplied + returnApply.getProductCount() > orderItem.getProductQuantity()) {
            Asserts.fail("Return quantity exceeds remaining eligible quantity");
        }

        OmsOrderReturnApply realApply = new OmsOrderReturnApply();
        BeanUtils.copyProperties(returnApply,realApply);
        // Client URLs are not evidence. Attachment object keys are created by
        // the controlled object-storage flow and linked after the application
        // has a durable ID.
        realApply.setProofPics(null);
        // Authoritative order and product data always comes from the commerce database.
        realApply.setOrderSn(order.getOrderSn());
        realApply.setMemberUsername(memberService.getCurrentMember().getUsername());
        realApply.setProductPic(orderItem.getProductPic());
        realApply.setProductName(orderItem.getProductName());
        realApply.setProductBrand(orderItem.getProductBrand());
        realApply.setProductAttr(orderItem.getProductAttr());
        realApply.setProductPrice(orderItem.getProductPrice());
        realApply.setProductRealPrice(orderItem.getRealAmount()
                .divide(java.math.BigDecimal.valueOf(orderItem.getProductQuantity()), 2, RoundingMode.DOWN));
        realApply.setReturnAmount(realApply.getProductRealPrice()
                .multiply(java.math.BigDecimal.valueOf(returnApply.getProductCount())));
        realApply.setCreateTime(new Date());
        realApply.setStatus(0);
        returnApplyMapper.insert(realApply);
        return realApply.getId();
    }
}
