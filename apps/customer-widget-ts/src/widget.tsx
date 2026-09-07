"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

export type RefundSelection = { orderId: number; productId: number; quantity: number };
export type OrderCancellationSelection = { orderId: number; orderSn?: string };
type Preview = { actionToken: string; previewHash: string; amount: number; expiresAt: string };
type CancellationPreview = { actionToken: string; previewHash: string; expiresAt: string };
type ConfirmedAfterSale = { actionId: number; afterSaleId: number; status: "APPLICATION_CREATED" };
type Product = { id: number; name: string; price: number; promotionPrice?: number; stock: number; brandName?: string; description?: string };
type Sku = { id: number; skuCode: string; price: number; promotionPrice?: number; stock: number };
type ProductDetail = { product: Product; skus: Sku[] };
type InterruptPayload = {
  kind: "refund_preview_confirmation";
  thread_id?: string;
  order_sn?: string;
  draft: { orderId: number; orderSn: string; productId: number; productName: string; quantity: number; reason: string };
};
type HistoryMessage = {
  id: number;
  message_type: string;
  sender_type: string;
  content: { message?: string; handoff?: unknown; interrupt?: unknown };
  created_at: string;
};
type HandoffStatus = { mode: "agent" | "human" };

export function CustomerSupportWidget({ threadId, selection, cancellationSelection }: {
  threadId: string; selection?: RefundSelection; cancellationSelection?: OrderCancellationSelection
}) {
  const [question, setQuestion] = useState("");
  const [productQuery, setProductQuery] = useState("");
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<ProductDetail | null>(null);
  const [loadingProductId, setLoadingProductId] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  const [handoffMode, setHandoffMode] = useState<HandoffStatus["mode"]>("agent");
  const [streaming, setStreaming] = useState(false);
  const [pendingInterrupt, setPendingInterrupt] = useState<InterruptPayload | null>(null);
  const [reason, setReason] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [refundIdempotencyKey, setRefundIdempotencyKey] = useState<string | null>(null);
  const [cancellationPreview, setCancellationPreview] = useState<CancellationPreview | null>(null);
  const [cancellationTraceId, setCancellationTraceId] = useState<string | null>(null);
  const [cancellationIdempotencyKey, setCancellationIdempotencyKey] = useState<string | null>(null);
  const [agentTraceId, setAgentTraceId] = useState<string | null>(null);
  const [afterSaleIdForEvidence, setAfterSaleIdForEvidence] = useState<number | null>(null);
  const [evidenceFiles, setEvidenceFiles] = useState<File[]>([]);
  const [evidenceObjectKeys, setEvidenceObjectKeys] = useState<string[]>([]);
  const [uploadingEvidence, setUploadingEvidence] = useState(false);
  const [error, setError] = useState("");

  const loadHistory = useCallback(async () => {
    const response = await fetch(`/api/chat/history/${encodeURIComponent(threadId)}`, { cache: "no-store" });
    if (!response.ok) return;
    setHistory(await response.json() as HistoryMessage[]);
  }, [threadId]);

  const loadHandoffStatus = useCallback(async () => {
    const response = await fetch(`/api/chat/handoff-status/${encodeURIComponent(threadId)}`, { cache: "no-store" });
    if (!response.ok) return;
    const status = await response.json() as HandoffStatus;
    setHandoffMode(status.mode);
  }, [threadId]);

  useEffect(() => {
    void loadHistory();
    void loadHandoffStatus();
    const interval = window.setInterval(() => { void loadHistory(); void loadHandoffStatus(); }, 10_000);
    return () => window.clearInterval(interval);
  }, [loadHistory, loadHandoffStatus]);

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || streaming || handoffMode === "human") return;
    setError("");
    setStreaming(true);
    setPendingInterrupt(null);
    setHistory(previous => [...previous, {
      id: Date.now(), message_type: "text", sender_type: "user",
      content: { message: question }, created_at: new Date().toISOString()
    }]);
    setQuestion("");

    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId })
    });
    if (!response.ok || !response.body) {
      setError(response.status === 409 ? "A human support agent is handling this conversation." : "Chat is temporarily unavailable.");
      setStreaming(false);
      return;
    }

    let assistant = "";
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const result = await reader.read();
      if (result.done) break;
      buffer += decoder.decode(result.value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const eventText of events) {
        const line = eventText.split("\n").find(value => value.startsWith("data: "));
        if (!line) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        try {
          const payload = JSON.parse(data);
          if (payload.token) {
            assistant += payload.token;
            setHistory(previous => {
              const copy = [...previous];
              copy[copy.length - 1] = { id: Date.now(), message_type: "text", sender_type: "agent", content: { message: assistant }, created_at: new Date().toISOString() };
              return copy;
            });
          } else if (payload.interrupt) {
            setPendingInterrupt(payload.interrupt as InterruptPayload);
          }
        } catch { /* ignore malformed stream event */ }
      }
    }
    setStreaming(false);
    await loadHistory();
  }

  async function searchProducts(event: FormEvent) {
    event.preventDefault();
    if (!productQuery.trim()) return;
    setError("");
    setSelectedProduct(null);
    const response = await fetch(`/api/products/search?${new URLSearchParams({ query: productQuery.trim() })}`, { cache: "no-store" });
    const body: { data?: Product[] } | null = await response.json().catch(() => null);
    if (!response.ok || !Array.isArray(body?.data)) return setError("Unable to search products.");
    setProducts(body.data);
  }

  async function showProductDetail(productId: number) {
    if (loadingProductId !== null) return;
    setError("");
    setLoadingProductId(productId);
    try {
      const response = await fetch(`/api/products/${encodeURIComponent(String(productId))}`, { cache: "no-store" });
      const body: { data?: ProductDetail } | null = await response.json().catch(() => null);
      if (!response.ok || !body?.data || !Array.isArray(body.data.skus)) {
        setError("Unable to load current product options.");
        return;
      }
      setSelectedProduct(body.data);
    } finally {
      setLoadingProductId(null);
    }
  }

  async function decide(decision: "confirmed" | "cancelled") {
    if (!pendingInterrupt) return;
    const response = await fetch("/api/chat/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: threadId,
        decision,
        confirmed_at: decision === "confirmed" ? new Date().toISOString() : undefined
      })
    });
    if (!response.ok) return setError("Unable to submit your decision.");
    setPendingInterrupt(null);
    await loadHistory();
  }

  async function createPreview() {
    if (!selection || !reason.trim()) return setError("Select an order item and enter a reason first.");
    const nextTraceId = crypto.randomUUID();
    const response = await fetch("/api/refund/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...selection, reason, agentTraceId: nextTraceId })
    });
    const body = await response.json();
    if (!response.ok || !body.data) return setError("Unable to create refund preview.");
    setPreview(body.data);
    setAgentTraceId(nextTraceId);
    setRefundIdempotencyKey(crypto.randomUUID());
    setError("");
  }

  async function confirmPreview() {
    if (!preview || !agentTraceId || !refundIdempotencyKey) return;
    const response = await fetch("/api/refund/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actionToken: preview.actionToken,
        previewHash: preview.previewHash,
        agentTraceId,
        idempotencyKey: refundIdempotencyKey,
        confirmedAt: new Date().toISOString()
      })
    });
    const body: { data?: ConfirmedAfterSale } = await response.json();
    if (!response.ok || !body.data) return setError("Unable to submit after-sales request.");
    const confirmed = body.data;
    setPreview(null);
    setAgentTraceId(null);
    setRefundIdempotencyKey(null);
    setAfterSaleIdForEvidence(confirmed.afterSaleId);
    setEvidenceObjectKeys([]);
    setHistory(previous => [...previous, {
      id: Date.now(), message_type: "refund_card", sender_type: "system",
      content: { message: `Your after-sales request #${confirmed.afterSaleId} has been created.` },
      created_at: new Date().toISOString()
    }]);
  }

  async function submitEvidence() {
    if (!afterSaleIdForEvidence || evidenceFiles.length === 0 || uploadingEvidence) return;
    setUploadingEvidence(true);
    setError("");
    let objectKeys = evidenceObjectKeys;
    try {
      if (objectKeys.length === 0) {
        const uploaded: string[] = [];
        for (const file of evidenceFiles) {
          const form = new FormData();
          form.append("file", file);
          const response = await fetch("/api/after-sales/evidence/upload", { method: "POST", body: form });
          const body: { data?: { objectKey?: string } } = await response.json().catch(() => ({}));
          if (!response.ok || !body.data?.objectKey) {
            setEvidenceObjectKeys(uploaded);
            return setError("Unable to upload one of the evidence files.");
          }
          uploaded.push(body.data.objectKey);
          setEvidenceObjectKeys([...uploaded]);
        }
        objectKeys = uploaded;
      }
      const response = await fetch(`/api/after-sales/evidence/${encodeURIComponent(String(afterSaleIdForEvidence))}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ objectKeys })
      });
      if (!response.ok) return setError("Unable to attach evidence to the after-sales request.");
      setEvidenceFiles([]);
      setHistory(previous => [...previous, {
        id: Date.now(), message_type: "refund_card", sender_type: "system",
        content: { message: `Evidence was attached to after-sales request #${afterSaleIdForEvidence}.` },
        created_at: new Date().toISOString()
      }]);
    } finally {
      setUploadingEvidence(false);
    }
  }

  async function createCancellationPreview() {
    if (!cancellationSelection) return setError("Select a pending order first.");
    const nextTraceId = crypto.randomUUID();
    const response = await fetch("/api/order-cancellation/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ orderId: cancellationSelection.orderId, agentTraceId: nextTraceId })
    });
    const body: { data?: CancellationPreview } = await response.json().catch(() => ({}));
    if (!response.ok || !body.data) return setError("This order can no longer be cancelled.");
    setCancellationPreview(body.data);
    setCancellationTraceId(nextTraceId);
    setCancellationIdempotencyKey(crypto.randomUUID());
    setError("");
  }

  async function confirmCancellationPreview() {
    if (!cancellationPreview || !cancellationTraceId || !cancellationIdempotencyKey) return;
    const response = await fetch("/api/order-cancellation/confirm", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actionToken: cancellationPreview.actionToken, previewHash: cancellationPreview.previewHash,
        agentTraceId: cancellationTraceId, idempotencyKey: cancellationIdempotencyKey, confirmedAt: new Date().toISOString()
      })
    });
    const body: { data?: { orderId: number; status: "CANCELLED" } } = await response.json().catch(() => ({}));
    if (!response.ok || !body.data) return setError("Unable to cancel this order; its status may have changed.");
    setCancellationPreview(null);
    setCancellationTraceId(null);
    setCancellationIdempotencyKey(null);
    setHistory(previous => [...previous, {
      id: Date.now(), message_type: "order_card", sender_type: "system",
      content: { message: `Order ${body.data?.orderId} has been cancelled.` }, created_at: new Date().toISOString()
    }]);
  }

  return <section aria-label="Customer support">
    <div aria-live="polite">
      {history.map(message => (
        <article key={message.id}>
          <strong>{message.sender_type}</strong>
          <p>{message.content.message}</p>
          {message.message_type === "refund_card" && <p role="status">Refund handoff — final approval is owned by the commerce after-sales state machine.</p>}
        </article>
      ))}
    </div>
    {handoffMode === "human" && <p role="status">A human support agent is handling this conversation.</p>}
    <form onSubmit={ask}>
      <label>Ask customer support<input value={question} onChange={event => setQuestion(event.target.value)} required disabled={streaming || handoffMode === "human"} /></label>
      <button type="submit" disabled={streaming}>{streaming ? "Sending…" : "Send"}</button>
    </form>
    <section aria-label="Product search">
      <form onSubmit={searchProducts}>
        <label>Find products<input value={productQuery} onChange={event => setProductQuery(event.target.value)} maxLength={200} /></label>
        <button type="submit">Search products</button>
      </form>
      {products.map(product => <article key={product.id} aria-label={`Product ${product.name}`}>
        <strong>{product.name}</strong>
        <p>{product.promotionPrice ?? product.price} · {product.stock > 0 ? `${product.stock} in stock` : "Out of stock"}</p>
        {product.brandName && <p>{product.brandName}</p>}
        {product.description && <p>{product.description}</p>}
        <button type="button" onClick={() => void showProductDetail(product.id)} disabled={loadingProductId !== null}>
          {loadingProductId === product.id ? "Loading options..." : "View options"}
        </button>
      </article>)}
      {selectedProduct && <section aria-label={`Product options for ${selectedProduct.product.name}`}>
        <h3>{selectedProduct.product.name} options</h3>
        {selectedProduct.skus.length === 0 ? <p>No purchasable options are currently available.</p> : (
          <ul>{selectedProduct.skus.map(sku => <li key={sku.id}>
            {sku.skuCode}: {sku.promotionPrice ?? sku.price} · {sku.stock > 0 ? `${sku.stock} in stock` : "Out of stock"}
          </li>)}</ul>
        )}
      </section>}
    </section>
    {pendingInterrupt && (
      <section aria-label="Confirm refund request" role="alertdialog">
        <p>Order {pendingInterrupt.order_sn}: confirm a refund draft for {pendingInterrupt.draft.productName} x{pendingInterrupt.draft.quantity}?</p>
        <button type="button" onClick={() => void decide("confirmed")}>Confirm refund request</button>
        <button type="button" onClick={() => void decide("cancelled")}>Cancel</button>
      </section>
    )}
    <section aria-label="Refund request">
      <label>Refund reason<input value={reason} onChange={event => setReason(event.target.value)} /></label>
      <button type="button" onClick={() => void createPreview()}>Preview refund</button>
      {preview && (
        <div>
          <p>Refund amount: {preview.amount}</p>
          <p>Expires: {new Date(preview.expiresAt).toLocaleString()}</p>
          <button type="button" onClick={() => void confirmPreview()}>Confirm refund request</button>
        </div>
      )}
    </section>
    {afterSaleIdForEvidence && <section aria-label="After-sales evidence">
      <p>Attach up to three JPEG, PNG, or PDF files to after-sales request #{afterSaleIdForEvidence}.</p>
      <input type="file" accept="image/jpeg,image/png,application/pdf" multiple onChange={event => {
        const files = Array.from(event.target.files ?? []).slice(0, 3);
        if (files.some(file => file.size === 0 || file.size > 5 * 1024 * 1024)) {
          setError("Evidence files must be between 1 byte and 5 MiB.");
          return;
        }
        setEvidenceFiles(files);
        setEvidenceObjectKeys([]);
      }} />
      <button type="button" onClick={() => void submitEvidence()} disabled={uploadingEvidence || evidenceFiles.length === 0}>
        {uploadingEvidence ? "Uploading evidence..." : "Attach evidence"}
      </button>
    </section>}
    {cancellationSelection && <section aria-label="Order cancellation">
      <p>Order {cancellationSelection.orderSn ?? cancellationSelection.orderId}: cancellation is available only while payment is pending.</p>
      <button type="button" onClick={() => void createCancellationPreview()}>Preview cancellation</button>
      {cancellationPreview && <div>
        <p>Cancellation preview expires: {new Date(cancellationPreview.expiresAt).toLocaleString()}</p>
        <button type="button" onClick={() => void confirmCancellationPreview()}>Confirm order cancellation</button>
      </div>}
    </section>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
