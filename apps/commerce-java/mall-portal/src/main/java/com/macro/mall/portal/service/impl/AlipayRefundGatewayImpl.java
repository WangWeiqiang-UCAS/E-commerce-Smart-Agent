package com.macro.mall.portal.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.alipay.api.AlipayApiException;
import com.alipay.api.AlipayClient;
import com.alipay.api.request.AlipayTradeRefundRequest;
import com.alipay.api.response.AlipayTradeRefundResponse;
import com.macro.mall.portal.service.AlipayRefundGateway;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;

/**
 * Alipay accepts out_request_no as its merchant-side idempotency key. The
 * dispatcher deliberately reuses it after a process crash or a timeout.
 */
@Service
public class AlipayRefundGatewayImpl implements AlipayRefundGateway {
    private final AlipayClient alipayClient;

    public AlipayRefundGatewayImpl(AlipayClient alipayClient) {
        this.alipayClient = alipayClient;
    }

    @Override
    public String refund(String refundRequestNo, String orderSn, BigDecimal refundAmount) {
        AlipayTradeRefundRequest request = new AlipayTradeRefundRequest();
        JSONObject content = new JSONObject();
        content.put("out_trade_no", orderSn);
        content.put("refund_amount", refundAmount.toPlainString());
        content.put("out_request_no", refundRequestNo);
        request.setBizContent(content.toString());
        try {
            AlipayTradeRefundResponse response = alipayClient.execute(request);
            if (response == null || !response.isSuccess()) {
                String detail = response == null ? "empty Alipay refund response"
                        : response.getSubCode() + ":" + response.getSubMsg();
                throw new IllegalStateException("Alipay refund was not accepted: " + detail);
            }
            return response.getTradeNo();
        } catch (AlipayApiException error) {
            throw new IllegalStateException("Alipay refund request failed", error);
        }
    }
}
