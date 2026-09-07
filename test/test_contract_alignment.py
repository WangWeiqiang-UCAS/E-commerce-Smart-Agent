import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_order_list_contract_matches_java_pagination_and_tool_schema():
    openapi = yaml.safe_load((ROOT / "packages/api-contracts/agent-commerce.openapi.yaml").read_text(encoding="utf-8"))
    tool_schema = json.loads((ROOT / "packages/tool-schemas/commerce-tools.schema.json").read_text(encoding="utf-8"))

    parameters = {
        item["name"]: item["schema"]
        for item in openapi["paths"]["/agent/commerce/orders"]["get"]["parameters"]
    }
    assert set(parameters) == {"status", "pageNum", "pageSize"}
    assert parameters["status"]["type"] == "integer"
    assert parameters["pageSize"]["maximum"] == 50
    assert tool_schema["tools"]["get_user_orders"]["additionalProperties"] is False


def test_every_model_callable_tool_rejects_unknown_arguments():
    tool_schema = json.loads((ROOT / "packages/tool-schemas/commerce-tools.schema.json").read_text(encoding="utf-8"))
    assert all(schema.get("additionalProperties") is False for schema in tool_schema["tools"].values())


def test_commerce_inventory_sql_uses_conditional_reservations_and_releases():
    source = (
        ROOT / "apps/commerce-java/mall-portal/src/main/resources/dao/PortalOrderDao.xml"
    ).read_text(encoding="utf-8")

    assert '<update id="lockSkuStock">' in source
    assert "stock - lock_stock >= CASE id" in source
    assert "lock_stock = lock_stock + CASE id" in source
    assert "lock_stock >= CASE id" in source
    assert '<update id="cancelPendingOrder">' in source
    assert "AND status = 0" in source
    assert '<update id="confirmReceivedOrder">' in source
    assert "AND status = 2" in source


def test_receipt_confirmation_uses_a_conditional_idempotent_order_transition():
    service = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/OmsPortalOrderServiceImpl.java"
    ).read_text(encoding="utf-8")

    assert "portalOrderDao.confirmReceivedOrder(orderId)" in service
    assert "recordOrderStatusEvent(orderId, 3)" in service
    assert "Order status changed before receipt confirmation" in service


def test_order_deletion_is_owned_terminal_only_and_idempotent():
    dao = (ROOT / "apps/commerce-java/mall-portal/src/main/resources/dao/PortalOrderDao.xml").read_text(encoding="utf-8")
    service = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/OmsPortalOrderServiceImpl.java"
    ).read_text(encoding="utf-8")

    assert '<update id="hideTerminalOrderForMember">' in dao
    assert "member_id = #{memberId}" in dao
    assert "status IN (3, 4)" in dao
    assert "portalOrderDao.hideTerminalOrderForMember(orderId, order.getMemberId())" in service
    assert "Order status changed before deletion" in service


def test_payment_callbacks_bind_channel_data_to_the_order_amount():
    source = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/AlipayServiceImpl.java"
    ).read_text(encoding="utf-8")

    assert "pendingOrderForCurrentMember(aliPayParam.getOutTradeNo())" in source
    assert "bizContent.put(\"total_amount\", order.getPayAmount().toPlainString())" in source
    assert "callbackMatchesOrder(params)" in source
    assert "matchesAmount(order, response.getTotalAmount())" in source
    assert "aliPayParam.getTotalAmount()" not in source


def test_after_sales_creation_requires_paid_order_and_remaining_quantity():
    source = (
        ROOT
        / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/OmsPortalOrderReturnApplyServiceImpl.java"
    ).read_text(encoding="utf-8")

    assert "Only paid orders can enter after-sales processing" in source
    assert "Return quantity exceeds remaining eligible quantity" in source
    assert "@Transactional(isolation = Isolation.SERIALIZABLE)" in source
    assert ".andOrderIdEqualTo(order.getId())" in source
    assert ".andProductIdEqualTo(returnApply.getProductId())" in source
    assert "realApply.setProofPics(null)" in source


def test_admin_after_sales_status_updates_are_conditional_state_transitions():
    service = (
        ROOT / "apps/commerce-java/mall-admin/src/main/java/com/macro/mall/service/impl/OmsOrderReturnApplyServiceImpl.java"
    ).read_text(encoding="utf-8")
    mapper = (ROOT / "apps/commerce-java/mall-admin/src/main/resources/dao/OmsOrderReturnApplyDao.xml").read_text(
        encoding="utf-8"
    )

    assert "returnApplyDao.transitionStatus(returnApply, expectedStatus)" in service
    assert "Approved return amount is invalid" in service
    assert "AND status = #{expectedStatus}" in mapper


def test_after_sales_status_event_is_described_and_consumable():
    asyncapi = yaml.safe_load((ROOT / "packages/api-contracts/commerce-events.asyncapi.yaml").read_text(encoding="utf-8"))
    consumer = (ROOT / "app/events/commerce_consumer.py").read_text(encoding="utf-8")

    assert "afterSaleStatusChanged" in asyncapi["channels"]["commerceEvents"]["messages"]
    assert "AfterSaleStatusChangedEvent" in asyncapi["components"]["schemas"]
    assert "after_sale.status_changed" in consumer


def test_outbox_publisher_uses_expiring_leases_instead_of_network_transactions():
    publisher = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/component/OutboxPublisher.java"
    ).read_text(encoding="utf-8")
    migration = (
        ROOT / "apps/commerce-java/document/sql/migrations/V20260820_01__outbox_publish_leases.sql"
    ).read_text(encoding="utf-8")

    assert "claimNextEvent" in publisher
    assert "publish_lock_owner" in publisher
    assert "INTERVAL 60 SECOND" in publisher
    assert "@Transactional" not in publisher
    assert "publish_lock_owner" in migration
    assert "idx_outbox_publish_lease" in migration


def test_customer_cancellation_routes_bind_orders_to_the_authenticated_member():
    controller = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/controller/OmsPortalOrderController.java"
    ).read_text(encoding="utf-8")
    service = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/OmsPortalOrderServiceImpl.java"
    ).read_text(encoding="utf-8")

    assert "Direct cancellation is disabled" in controller
    assert "sendDelayMessageCancelOrderForCurrentMember(orderId)" not in controller
    assert "portalOrderService.cancelOrderForCurrentMember(orderId);" not in controller
    assert "ownedOrderForCurrentMember(orderId)" in service
    assert "memberService.getCurrentMember()" in service
    assert 'Asserts.fail("Order not found")' in service
    assert "portalOrderService.cancelTimeOutOrder();" not in controller


def test_order_cancellation_is_an_explicit_preview_confirm_action():
    openapi = yaml.safe_load((ROOT / "packages/api-contracts/agent-commerce.openapi.yaml").read_text(encoding="utf-8"))
    controller = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/controller/AgentCommerceController.java"
    ).read_text(encoding="utf-8")
    action_service = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/AgentOrderCancellationActionServiceImpl.java"
    ).read_text(encoding="utf-8")
    migration = (
        ROOT / "apps/commerce-java/document/sql/migrations/V20260823_01__agent_order_cancellation_actions.sql"
    ).read_text(encoding="utf-8")

    assert "/agent/commerce/actions/order-cancellation/preview" in openapi["paths"]
    assert "/agent/commerce/actions/order-cancellation/confirm" in openapi["paths"]
    assert "previewOrderCancellation" in controller
    assert "confirmOrderCancellation" in controller
    assert "ORDER_CANCEL" in action_service
    assert "FOR UPDATE" in action_service
    assert "cancelOrderForCurrentMember(orderId)" in action_service
    assert "order_status_at_preview" in action_service
    assert "order_status_at_preview" in migration
    assert "VALUES (?,?,?,?,?,?,?,NOW(),?,?)" in action_service
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "V20260823_01__agent_order_cancellation_actions.sql" in compose


def test_widget_reuses_idempotency_keys_for_action_confirmation_retries():
    widget = (ROOT / "apps/customer-widget-ts/src/widget.tsx").read_text(encoding="utf-8")

    assert "const [refundIdempotencyKey" in widget
    assert "idempotencyKey: refundIdempotencyKey" in widget
    assert "const [cancellationIdempotencyKey" in widget
    assert "idempotencyKey: cancellationIdempotencyKey" in widget


def test_order_cancellation_event_is_contractually_described_and_consumed():
    asyncapi = yaml.safe_load((ROOT / "packages/api-contracts/commerce-events.asyncapi.yaml").read_text(encoding="utf-8"))
    consumer = (ROOT / "app/events/commerce_consumer.py").read_text(encoding="utf-8")
    producer = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/AgentOrderCancellationActionServiceImpl.java"
    ).read_text(encoding="utf-8")

    assert "orderStatusChanged" in asyncapi["channels"]["commerceEvents"]["messages"]
    assert "OrderStatusChangedEvent" in asyncapi["components"]["schemas"]
    assert '"orderStatus\\":4' in producer
    assert "order.status_changed" in consumer


def test_received_after_sales_creates_a_durable_idempotent_payment_refund_dispatch():
    admin_service = (
        ROOT / "apps/commerce-java/mall-admin/src/main/java/com/macro/mall/service/impl/OmsOrderReturnApplyServiceImpl.java"
    ).read_text(encoding="utf-8")
    dispatcher = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/component/RefundDispatcher.java"
    ).read_text(encoding="utf-8")
    gateway = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/AlipayRefundGatewayImpl.java"
    ).read_text(encoding="utf-8")
    migration = (
        ROOT / "apps/commerce-java/document/sql/migrations/V20260823_02__payment_refund_dispatch.sql"
    ).read_text(encoding="utf-8")
    asyncapi = yaml.safe_load((ROOT / "packages/api-contracts/commerce-events.asyncapi.yaml").read_text(encoding="utf-8"))

    assert "INSERT INTO oms_payment_refund" in admin_service
    assert '"after-sale-" + id' in admin_service
    assert "ON DUPLICATE KEY UPDATE" in admin_service
    assert "out_request_no" in gateway
    assert "refundRequestNo" in gateway
    assert "refund.completed" in dispatcher
    assert "TransactionTemplate" in dispatcher
    assert "uk_payment_refund_after_sale" in migration
    assert "uk_payment_refund_request_no" in migration
    assert "refundCompleted" in asyncapi["channels"]["commerceEvents"]["messages"]
    java_ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "RefundDispatcherTest" in java_ci


def test_after_sales_evidence_uses_private_object_keys_and_member_scoped_attachment_records():
    service = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/service/impl/AfterSaleEvidenceServiceImpl.java"
    ).read_text(encoding="utf-8")
    controller = (
        ROOT / "apps/commerce-java/mall-portal/src/main/java/com/macro/mall/portal/controller/AfterSaleEvidenceController.java"
    ).read_text(encoding="utf-8")
    migration = (
        ROOT / "apps/commerce-java/document/sql/migrations/V20260823_03__after_sale_evidence_attachments.sql"
    ).read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    bff_upload = (ROOT / "apps/agent-bff-ts/src/app/api/after-sales/evidence/upload/route.ts").read_text(encoding="utf-8")
    bff_attach = (ROOT / "apps/agent-bff-ts/src/app/api/after-sales/evidence/[afterSaleId]/route.ts").read_text(encoding="utf-8")

    assert "after-sales/" in service
    assert "MAX_FILE_SIZE" in service
    assert "Only JPEG, PNG, and PDF evidence files are accepted" in service
    assert "getPresignedObjectUrl" in service
    assert "member_id" in migration
    assert "uk_after_sale_attachment_object_key" in migration
    assert "MultipartFile" in controller
    assert 'assertSameOrigin(request)' in bff_upload
    assert "FormData" in bff_upload
    assert "objectKeys" in bff_attach
    assert "MINIO_SECRET_KEY:?Set MINIO_SECRET_KEY" in compose
    assert "V20260823_03__after_sale_evidence_attachments.sql" in compose
    widget = (ROOT / "apps/customer-widget-ts/src/widget.tsx").read_text(encoding="utf-8")
    assert "/api/after-sales/evidence/upload" in widget
    assert "setEvidenceObjectKeys" in widget
    assert "afterSaleIdForEvidence" in widget
