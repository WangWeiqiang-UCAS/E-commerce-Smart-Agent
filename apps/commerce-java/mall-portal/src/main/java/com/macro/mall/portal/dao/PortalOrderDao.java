package com.macro.mall.portal.dao;

import com.macro.mall.model.OmsOrderItem;
import com.macro.mall.portal.domain.OmsOrderDetail;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * 前台订单管理自定义Dao
 * Created by macro on 2018/9/4.
 */
public interface PortalOrderDao {
    /**
     * 获取订单及下单商品详情
     */
    OmsOrderDetail getDetail(@Param("orderId") Long orderId);

    /**
     * 修改 pms_sku_stock表的锁定库存及真实库存
     */
    int updateSkuStock(@Param("itemList") List<OmsOrderItem> orderItemList);

    /** Reserve stock only when every requested SKU still has available units. */
    int lockSkuStock(@Param("itemList") List<OmsOrderItem> orderItemList);

    /** Atomically transition a pending order once; protects duplicate provider callbacks. */
    int markOrderPaidIfPending(@Param("orderId") Long orderId, @Param("payType") Integer payType);

    /**
     * 获取超时订单
     * @param minute 超时时间（分）
     */
    List<OmsOrderDetail> getTimeOutOrders(@Param("minute") Integer minute);

    /**
     * 批量修改订单状态
     */
    int updateOrderStatus(@Param("ids") List<Long> ids,@Param("status") Integer status);

    /** Transition a pending order to cancelled exactly once. */
    int cancelPendingOrder(@Param("orderId") Long orderId);

    /** Transition a shipped order to received exactly once. */
    int confirmReceivedOrder(@Param("orderId") Long orderId);

    /** Hide an owned terminal order exactly once without changing its business status. */
    int hideTerminalOrderForMember(@Param("orderId") Long orderId, @Param("memberId") Long memberId);

    /**
     * 解除取消订单的库存锁定
     */
    int releaseSkuStockLock(@Param("itemList") List<OmsOrderItem> orderItemList);

}
