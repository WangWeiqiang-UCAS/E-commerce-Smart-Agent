"use client";
import { useEffect, useState } from "react";

type ThreadSummary = {
  thread_id: string;
  message_count: number;
  last_message_at: string;
  last_sender_type?: string | null;
  last_content?: { message?: string } | null;
};

type ThreadMessage = {
  id: number;
  thread_id: string;
  message_type: string;
  sender_type: string;
  content: { message?: string; handoff?: unknown; interrupt?: unknown };
  created_at: string;
};

export default function AgentDesk() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [selectedThread, setSelectedThread] = useState<string | null>(null);
  const [messages, setMessages] = useState<ThreadMessage[]>([]);
  const [error, setError] = useState("");

  async function loadThreads() {
    const response = await fetch("/api/threads", { cache: "no-store" });
    if (response.status === 401) return setAuthenticated(false);
    if (!response.ok) return setError("Unable to load conversations.");
    const body = await response.json() as ThreadSummary[];
    setThreads(body);
    setError("");
  }

  async function login(event: React.FormEvent) {
    event.preventDefault();
    const response = await fetch("/api/session/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    if (!response.ok) return setError("Login failed or the account is not an administrator.");
    setError("");
    setAuthenticated(true);
    await loadThreads();
  }

  async function logout() {
    await fetch("/api/session/logout", { method: "POST" });
    setAuthenticated(false);
    setThreads([]);
    setMessages([]);
    setSelectedThread(null);
  }

  async function openThread(threadId: string) {
    setSelectedThread(threadId);
    const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`, { cache: "no-store" });
    if (!response.ok) return setError("Unable to load messages.");
    setMessages(await response.json() as ThreadMessage[]);
    setError("");
  }

  useEffect(() => { void loadThreads(); }, []);

  return <main>
    <h1>Agent support desk</h1>
    {!authenticated ? (
      <form onSubmit={login}>
        <label>Administrator username<input value={username} onChange={event => setUsername(event.target.value)} required /></label>
        <label>Password<input type="password" value={password} onChange={event => setPassword(event.target.value)} required /></label>
        <button type="submit">Sign in</button>
      </form>
    ) : (
      <>
        <section>
          <h2>Conversation queue</h2>
          <button onClick={() => void loadThreads()}>Refresh</button>
          <button onClick={() => void logout()}>Sign out</button>
          {threads.length === 0 && <p role="status">No conversations yet.</p>}
          <ul>
            {threads.map(thread => (
              <li key={thread.thread_id}>
                <button onClick={() => void openThread(thread.thread_id)}>
                  {thread.thread_id} — {thread.message_count} messages · {thread.last_content?.message?.slice(0, 60) ?? "no text"}
                </button>
              </li>
            ))}
          </ul>
        </section>
        {selectedThread && (
          <section aria-label="Conversation transcript">
            <h2>{selectedThread}</h2>
            {messages.map(message => (
              <article key={message.id}>
                <strong>{message.sender_type}</strong> <time>{new Date(message.created_at).toLocaleString()}</time>
                <p>{message.content.message}</p>
                {message.message_type === "refund_card" && <p role="status">Refund handoff recorded — final approval is owned by the commerce after-sales state machine.</p>}
                {message.message_type === "audit_card" && <p role="status">Awaiting explicit customer confirmation (interrupt).</p>}
              </article>
            ))}
          </section>
        )}
      </>
    )}
    {error && <p role="alert">{error}</p>}
  </main>;
}
