package com.macro.mall.portal.domain;

import com.macro.mall.model.PmsProduct;
import com.macro.mall.model.PmsSkuStock;

import java.math.BigDecimal;
import java.util.List;

/** Non-sensitive product projections permitted at the agent boundary. */
public final class AgentProduct {
    private AgentProduct() { }

    public record Summary(Long id, String name, String productSn, String pic, BigDecimal price,
                          BigDecimal promotionPrice, Integer stock, String brandName,
                          String categoryName, String description) {
        public static Summary from(PmsProduct product) {
            return new Summary(product.getId(), product.getName(), product.getProductSn(), product.getPic(),
                    product.getPrice(), product.getPromotionPrice(), product.getStock(), product.getBrandName(),
                    product.getProductCategoryName(), product.getDescription());
        }
    }

    public record Sku(Long id, String skuCode, BigDecimal price, BigDecimal promotionPrice,
                      Integer stock, String pic, String attributes) {
        public static Sku from(PmsSkuStock sku) {
            return new Sku(sku.getId(), sku.getSkuCode(), sku.getPrice(), sku.getPromotionPrice(),
                    sku.getStock(), sku.getPic(), sku.getSpData());
        }
    }

    public record Detail(Summary product, List<Sku> skus) { }
}
