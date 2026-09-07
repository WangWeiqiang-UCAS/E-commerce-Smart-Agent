"use client";
import { useEffect, useState } from "react";

type ThreadSummary = { thread_id: string; message_count: number; last_message_at: string; last_sender_type?: string | null; last_content?: { message?: string } | null; mode: "agent" | "human"; assigned_admin_id?: number | null };
type ThreadMessage = { id: number; thread_id: string; message_type: string; sender_type: string; content: { message?: string; handoff?: unknown; interrupt?: unknown }; created_at: string };
type SupportOperator = { id: number; username: string; full_name: string };

export default function AgentDesk() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [selectedThread, setSelectedThread] = useState<string | null>(null);
  const [messages, setMessages] = useState<ThreadMessage[]>([]);
  const [reply, setReply] = useState("");
  const [note, setNote] = useState("");
  const [operators, setOperators] = useState<SupportOperator[]>([]);
  const [transferTarget, setTransferTarget] = useState("");
  const [transferReason, setTransferReason] = useState("");
  const [error, setError] = useState("");

  async function loadThreads(): Promise<boolean> {
    const response = await fetch("/api/threads", { cache: "no-store" });
    if (response.status === 401) { setAuthenticated(false); return false; }
    if (!response.ok) { setError("Unable to load conversations."); return false; }
    setThreads(await response.json() as ThreadSummary[]);
    setError("");
    return true;
  }
  async function loadOperators() {
    const response = await fetch("/api/support-operators", { cache: "no-store" });
    if (response.ok) setOperators(await response.json() as SupportOperator[]);
  }
  async function login(event: React.FormEvent) {
    event.preventDefault();
    const response = await fetch("/api/session/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }) });
    if (!response.ok) return setError("Login failed or the account is not an administrator.");
    setAuthenticated(true);
    await loadThreads();
    await loadOperators();
  }
  async function logout() {
    await fetch("/api/session/logout", { method: "POST" });
    window.history.replaceState(null, "", window.location.pathname);
    setAuthenticated(false); setThreads([]); setMessages([]); setSelectedThread(null); setOperators([]);
  }
  async function openThread(threadId: string) {
    setSelectedThread(threadId);
    window.history.replaceState(null, "", `#thread=${encodeURIComponent(threadId)}`);
    const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`, { cache: "no-store" });
    if (!response.ok) return setError("Unable to load messages.");
    setMessages(await response.json() as ThreadMessage[]);
    setError("");
  }
  const selectedSummary = threads.find(thread => thread.thread_id === selectedThread);
  async function takeOver() {
    if (!selectedThread) return;
    const response = await fetch(`/api/threads/${encodeURIComponent(selectedThread)}/takeover`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (!response.ok) return setError("Unable to take over this conversation.");
    await loadThreads(); await openThread(selectedThread);
  }
  async function returnToAgent() {
    if (!selectedThread) return;
    const response = await fetch(`/api/threads/${encodeURIComponent(selectedThread)}/return-to-agent`, { method: "POST" });
    if (!response.ok) return setError("Only the assigned administrator can return this conversation.");
    await loadThreads(); await openThread(selectedThread);
  }
  async function sendReply(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedThread || !reply.trim()) return;
    const response = await fetch(`/api/threads/${encodeURIComponent(selectedThread)}/messages`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: reply.trim() }) });
    if (!response.ok) return setError("Take over the conversation before sending a reply.");
    setReply("");
    await openThread(selectedThread);
  }
  async function addNote(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedThread || !note.trim()) return;
    const response = await fetch(`/api/threads/${encodeURIComponent(selectedThread)}/notes`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: note.trim() }) });
    if (!response.ok) return setError("Only the current conversation owner can add an internal note.");
    setNote("");
    await openThread(selectedThread);
  }
  async function transferThread(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedThread || !transferTarget) return;
    const response = await fetch(`/api/threads/${encodeURIComponent(selectedThread)}/transfer`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target_admin_id: Number(transferTarget), reason: transferReason || undefined }) });
    if (!response.ok) return setError("The conversation could not be transferred. Refresh and try again.");
    setTransferReason(""); setTransferTarget("");
    await loadThreads(); await openThread(selectedThread);
  }
  useEffect(() => {
    void (async () => {
      const restored = await loadThreads();
      if (!restored) return;
      setAuthenticated(true);
      await loadOperators();
      const match = /^#thread=(.+)$/.exec(window.location.hash);
      if (match) await openThread(decodeURIComponent(match[1]));
    })();
  }, []);

  return <main>
    <h1>Agent support desk</h1>
    {!authenticated ? <form onSubmit={login}>
      <label>Administrator username<input value={username} onChange={event => setUsername(event.target.value)} required /></label>
      <label>Password<input type="password" value={password} onChange={event => setPassword(event.target.value)} required /></label>
      <button type="submit">Sign in</button>
    </form> : <>
      <section><h2>Conversation queue</h2>
        <button onClick={() => void loadThreads()}>Refresh</button><button onClick={() => void logout()}>Sign out</button>
        {threads.length === 0 && <p role="status">No conversations yet.</p>}
        <ul>{threads.map(thread => <li key={thread.thread_id}><button onClick={() => void openThread(thread.thread_id)}>
          {thread.thread_id} — {thread.message_count} messages · {thread.mode === "human" ? "human handling" : "Agent handling"} · {thread.last_content?.message?.slice(0, 60) ?? "no text"}
        </button></li>)}</ul>
      </section>
      {selectedThread && <section aria-label="Conversation transcript">
        <h2>{selectedThread}</h2>
        <p role="status">{selectedSummary?.mode === "human" ? "Human support currently owns this conversation." : "The Agent currently owns this conversation."}</p>
        {selectedSummary?.mode === "human" ? <button onClick={() => void returnToAgent()}>Return to Agent</button> : <button onClick={() => void takeOver()}>Take over conversation</button>}
        {messages.map(message => <article key={message.id}><strong>{message.sender_type === "internal" ? "internal note" : message.sender_type}</strong> <time>{new Date(message.created_at).toLocaleString()}</time><p>{message.content.message}</p>
          {message.sender_type === "agent" && message.content.message && <button type="button" onClick={() => setReply(message.content.message ?? "")}>
            Use as editable reply
          </button>}
          {message.message_type === "refund_card" && <p role="status">Refund handoff recorded — final approval is owned by the commerce after-sales state machine.</p>}
          {message.message_type === "audit_card" && <p role="status">Awaiting explicit customer confirmation (interrupt).</p>}
        </article>)}
        {selectedSummary?.mode === "human" && <>
          <form onSubmit={sendReply}><label>Reply to customer<textarea value={reply} onChange={event => setReply(event.target.value)} required maxLength={4000} /></label><p role="status">Agent suggestions are drafts. Review and submit this reply yourself.</p><button type="submit">Send reply</button></form>
          <form onSubmit={addNote}><label>Internal note<textarea value={note} onChange={event => setNote(event.target.value)} required maxLength={4000} /></label><p role="status">Internal notes are retained for support audit and never shown in the customer chat history.</p><button type="submit">Add internal note</button></form>
          <form onSubmit={transferThread}><label>Transfer to<select value={transferTarget} onChange={event => setTransferTarget(event.target.value)} required><option value="">Choose an administrator</option>{operators.map(operator => <option value={operator.id} key={operator.id}>{operator.full_name} ({operator.username})</option>)}</select></label><label>Transfer reason (optional)<input value={transferReason} onChange={event => setTransferReason(event.target.value)} maxLength={500} /></label><button type="submit">Transfer conversation</button></form>
        </>}
      </section>}
    </>}
    {error && <p role="alert">{error}</p>}
  </main>;
}
