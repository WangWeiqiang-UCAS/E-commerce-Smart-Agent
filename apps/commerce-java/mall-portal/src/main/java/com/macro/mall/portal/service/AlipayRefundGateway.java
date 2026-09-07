package com.macro.mall.portal.service;

import java.math.BigDecimal;

/** Payment-channel boundary used only by the durable refund dispatcher. */
public interface AlipayRefundGateway {
    String refund(String refundRequestNo, String orderSn, BigDecimal refundAmount);
}
