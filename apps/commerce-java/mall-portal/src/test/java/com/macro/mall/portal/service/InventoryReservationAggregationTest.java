package com.macro.mall.portal.service;

import com.macro.mall.model.OmsOrderItem;
import com.macro.mall.portal.service.impl.OmsPortalOrderServiceImpl;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class InventoryReservationAggregationTest {
    @Test
    @SuppressWarnings("unchecked")
    void aggregatesRepeatedSkuBeforeConditionalStockUpdate() {
        OmsOrderItem first = new OmsOrderItem();
        first.setProductSkuId(7L);
        first.setProductQuantity(1);
        OmsOrderItem second = new OmsOrderItem();
        second.setProductSkuId(7L);
        second.setProductQuantity(2);

        List<OmsOrderItem> result = ReflectionTestUtils.invokeMethod(
                new OmsPortalOrderServiceImpl(), "stockItems", List.of(first, second));

        assertEquals(1, result.size());
        assertEquals(3, result.get(0).getProductQuantity());
    }
}
