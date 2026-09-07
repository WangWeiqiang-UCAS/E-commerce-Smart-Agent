package com.macro.mall.portal.service;

import com.macro.mall.portal.dao.PortalOrderDao;
import com.macro.mall.portal.service.impl.OmsPortalOrderServiceImpl;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PaymentCallbackIdempotencyTest {
    @Test
    void replayedPaymentCallbackDoesNotConsumeInventoryAgain() {
        PortalOrderDao orderDao = mock(PortalOrderDao.class);
        when(orderDao.markOrderPaidIfPending(9L, 1)).thenReturn(0);
        OmsPortalOrderServiceImpl service = new OmsPortalOrderServiceImpl();
        ReflectionTestUtils.setField(service, "portalOrderDao", orderDao);

        assertEquals(0, service.paySuccess(9L, 1));

        verify(orderDao).markOrderPaidIfPending(9L, 1);
        verify(orderDao, never()).getDetail(9L);
        verify(orderDao, never()).updateSkuStock(org.mockito.ArgumentMatchers.anyList());
    }
}
