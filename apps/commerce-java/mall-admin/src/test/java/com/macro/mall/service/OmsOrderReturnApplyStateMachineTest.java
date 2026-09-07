package com.macro.mall.service;

import com.macro.mall.common.exception.ApiException;
import com.macro.mall.dao.OmsOrderReturnApplyDao;
import com.macro.mall.dto.OmsUpdateStatusParam;
import com.macro.mall.mapper.OmsOrderReturnApplyMapper;
import com.macro.mall.model.OmsOrderReturnApply;
import com.macro.mall.service.impl.OmsOrderReturnApplyServiceImpl;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.jdbc.core.JdbcTemplate;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OmsOrderReturnApplyStateMachineTest {
    @Test
    void approvalCannotExceedTheOriginalApplicationAmount() {
        OmsOrderReturnApplyDao dao = mock(OmsOrderReturnApplyDao.class);
        OmsOrderReturnApplyMapper mapper = mock(OmsOrderReturnApplyMapper.class);
        OmsOrderReturnApply current = new OmsOrderReturnApply();
        current.setStatus(0);
        current.setReturnAmount(new BigDecimal("10.00"));
        when(mapper.selectByPrimaryKey(1L)).thenReturn(current);

        OmsUpdateStatusParam request = new OmsUpdateStatusParam();
        request.setStatus(1);
        request.setReturnAmount(new BigDecimal("10.01"));

        assertThrows(ApiException.class, () -> service(dao, mapper).updateStatus(1L, request));
        verify(dao, never()).transitionStatus(any(), any());
    }

    @Test
    void completionUsesTheReturningStateAsItsOnlyPredecessor() {
        OmsOrderReturnApplyDao dao = mock(OmsOrderReturnApplyDao.class);
        OmsOrderReturnApplyMapper mapper = mock(OmsOrderReturnApplyMapper.class);
        OmsOrderReturnApply current = new OmsOrderReturnApply();
        current.setStatus(1);
        when(mapper.selectByPrimaryKey(1L)).thenReturn(current);
        when(dao.transitionStatus(any(), eq(1))).thenReturn(1);

        OmsUpdateStatusParam request = new OmsUpdateStatusParam();
        request.setStatus(2);

        assertEquals(1, service(dao, mapper).updateStatus(1L, request));
        verify(dao).transitionStatus(any(), eq(1));
    }

    private OmsOrderReturnApplyServiceImpl service(OmsOrderReturnApplyDao dao, OmsOrderReturnApplyMapper mapper) {
        OmsOrderReturnApplyServiceImpl service = new OmsOrderReturnApplyServiceImpl();
        ReflectionTestUtils.setField(service, "returnApplyDao", dao);
        ReflectionTestUtils.setField(service, "returnApplyMapper", mapper);
        ReflectionTestUtils.setField(service, "jdbcTemplate", mock(JdbcTemplate.class));
        return service;
    }
}
