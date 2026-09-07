package com.macro.mall.service.impl;

import com.github.pagehelper.PageHelper;
import com.macro.mall.dao.OmsOrderReturnApplyDao;
import com.macro.mall.common.exception.Asserts;
import com.macro.mall.dto.OmsOrderReturnApplyResult;
import com.macro.mall.dto.OmsReturnApplyQueryParam;
import com.macro.mall.dto.OmsUpdateStatusParam;
import com.macro.mall.mapper.OmsOrderReturnApplyMapper;
import com.macro.mall.mapper.OmsOrderMapper;
import com.macro.mall.model.OmsOrder;
import com.macro.mall.model.OmsOrderReturnApply;
import com.macro.mall.model.OmsOrderReturnApplyExample;
import com.macro.mall.service.OmsOrderReturnApplyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.jdbc.core.JdbcTemplate;

import java.math.BigDecimal;
import java.util.Date;
import java.util.List;
import java.time.Instant;
import java.util.UUID;

/**
 * 订单退货管理Service实现类
 * Created by macro on 2018/10/18.
 */
@Service
public class OmsOrderReturnApplyServiceImpl implements OmsOrderReturnApplyService {
    @Autowired
    private OmsOrderReturnApplyDao returnApplyDao;
    @Autowired
    private OmsOrderReturnApplyMapper returnApplyMapper;
    @Autowired
    private OmsOrderMapper orderMapper;
    @Autowired
    private JdbcTemplate jdbcTemplate;
    @Override
    public List<OmsOrderReturnApply> list(OmsReturnApplyQueryParam queryParam, Integer pageSize, Integer pageNum) {
        PageHelper.startPage(pageNum,pageSize);
        return returnApplyDao.getList(queryParam);
    }

    @Override
    public int delete(List<Long> ids) {
        OmsOrderReturnApplyExample example = new OmsOrderReturnApplyExample();
        example.createCriteria().andIdIn(ids).andStatusEqualTo(3);
        return returnApplyMapper.deleteByExample(example);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public int updateStatus(Long id, OmsUpdateStatusParam statusParam) {
        Integer status = statusParam.getStatus();
        if (status == null) {
            return 0;
        }
        OmsOrderReturnApply current = returnApplyMapper.selectByPrimaryKey(id);
        if (current == null) {
            return 0;
        }
        Integer expectedStatus;
        OmsOrder refundOrder = null;
        OmsOrderReturnApply returnApply = new OmsOrderReturnApply();
        if(status.equals(1)){
            expectedStatus = 0;
            BigDecimal approvedAmount = statusParam.getReturnAmount();
            if (approvedAmount == null || approvedAmount.signum() < 0
                    || current.getReturnAmount() == null || approvedAmount.compareTo(current.getReturnAmount()) > 0) {
                Asserts.fail("Approved return amount is invalid");
            }
            //确认退货
            returnApply.setId(id);
            returnApply.setStatus(1);
            returnApply.setReturnAmount(statusParam.getReturnAmount());
            returnApply.setCompanyAddressId(statusParam.getCompanyAddressId());
            returnApply.setHandleTime(new Date());
            returnApply.setHandleMan(statusParam.getHandleMan());
            returnApply.setHandleNote(statusParam.getHandleNote());
        }else if(status.equals(2)){
            expectedStatus = 1;
            if (current.getReturnAmount() == null || current.getReturnAmount().signum() <= 0) {
                Asserts.fail("Refund amount is invalid");
            }
            refundOrder = orderMapper.selectByPrimaryKey(current.getOrderId());
            if (refundOrder == null) {
                Asserts.fail("Order for after-sales application was not found");
            }
            //完成退货
            returnApply.setId(id);
            returnApply.setStatus(2);
            returnApply.setReceiveTime(new Date());
            returnApply.setReceiveMan(statusParam.getReceiveMan());
            returnApply.setReceiveNote(statusParam.getReceiveNote());
        }else if(status.equals(3)){
            expectedStatus = 0;
            //拒绝退货
            returnApply.setId(id);
            returnApply.setStatus(3);
            returnApply.setHandleTime(new Date());
            returnApply.setHandleMan(statusParam.getHandleMan());
            returnApply.setHandleNote(statusParam.getHandleNote());
        }else{
            return 0;
        }
        int updated = returnApplyDao.transitionStatus(returnApply, expectedStatus);
        if (updated == 1) {
            if (status.equals(2)) {
                String paymentChannel = Integer.valueOf(1).equals(refundOrder.getPayType()) ? "ALIPAY" : "UNSUPPORTED";
                jdbcTemplate.update("INSERT INTO oms_payment_refund (after_sale_id,order_id,payment_channel,refund_request_no,"
                                + "refund_amount,status,attempts,created_at,updated_at) VALUES (?,?,?,?,?,'PENDING',0,NOW(),NOW()) "
                                + "ON DUPLICATE KEY UPDATE after_sale_id=VALUES(after_sale_id)",
                        id, refundOrder.getId(), paymentChannel, "after-sale-" + id, current.getReturnAmount());
            }
            String eventId = UUID.randomUUID().toString();
            String occurredAt = Instant.now().toString();
            String payload = "{\"eventId\":\"" + eventId + "\",\"eventType\":\"after_sale.status_changed\",\"schemaVersion\":1,\"aggregateId\":\""
                    + id + "\",\"afterSaleId\":\"" + id + "\",\"status\":" + status + ",\"occurredAt\":\"" + occurredAt + "\"}";
            jdbcTemplate.update("INSERT INTO oms_outbox_event (event_id,event_type,schema_version,aggregate_id,trace_id,payload,occurred_at) VALUES (?,?,?,?,?,?,NOW())",
                    eventId, "after_sale.status_changed", 1, id.toString(), null, payload);
        }
        return updated;
    }

    @Override
    public OmsOrderReturnApplyResult getItem(Long id) {
        return returnApplyDao.getDetail(id);
    }
}
