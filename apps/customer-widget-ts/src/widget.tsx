"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

export type RefundSelection = { orderId: number; productId: number; quantity: number };
type Preview = { actionToken: string; previewHash: string; amount: number; expiresAt: string };
type ConfirmedAfterSale = { actionId: number; afterSaleId: number; status: "APPLICATION_CREATED" };
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

export function CustomerSupportWidget({ threadId, selection }: { threadId: string; selection?: RefundSelection }) {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [pendingInterrupt, setPendingInterrupt] = useState<InterruptPayload | null>(null);
  const [reason, setReason] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [agentTraceId, setAgentTraceId] = useState<string | null>(null);
  const [error, setError] = useState("");

  const loadHistory = useCallback(async () => {
    const response = await fetch(`/api/chat/history/${encodeURIComponent(threadId)}`, { cache: "no-store" });
    if (!response.ok) return;
    setHistory(await response.json() as HistoryMessage[]);
  }, [threadId]);

  useEffect(() => { void loadHistory(); }, [loadHistory]);

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || streaming) return;
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
      setError("Chat is temporarily unavailable.");
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
    setError("");
  }

  async function confirmPreview() {
    if (!preview || !agentTraceId) return;
    const response = await fetch("/api/refund/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actionToken: preview.actionToken,
        previewHash: preview.previewHash,
        agentTraceId,
        idempotencyKey: crypto.randomUUID(),
        confirmedAt: new Date().toISOString()
      })
    });
    const body: { data?: ConfirmedAfterSale } = await response.json();
    if (!response.ok || !body.data) return setError("Unable to submit after-sales request.");
    const confirmed = body.data;
    setPreview(null);
    setAgentTraceId(null);
    setHistory(previous => [...previous, {
      id: Date.now(), message_type: "refund_card", sender_type: "system",
      content: { message: `Your after-sales request #${confirmed.afterSaleId} has been created.` },
      created_at: new Date().toISOString()
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
    <form onSubmit={ask}>
      <label>Ask customer support<input value={question} onChange={event => setQuestion(event.target.value)} required disabled={streaming} /></label>
      <button type="submit" disabled={streaming}>{streaming ? "Sending…" : "Send"}</button>
    </form>
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
    {error && <p role="alert">{error}</p>}
  </section>;
}
