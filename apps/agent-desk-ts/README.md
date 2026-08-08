# Agent desk

Human-support workspace for pending agent actions. It uses only an HttpOnly
administrator session cookie and proxies queue reads/decisions through server
routes (`/api/session/login`, `/api/threads`, `/api/threads/[threadId]`).
State-changing decisions require same-origin browser requests.

The current adapter lists conversation threads and transcripts from the Agent
admin read API (`/api/v1/threads`), which is guarded by an administrator JWT.
Refund decisions remain in the commerce after-sales state machine; the desk
records audit cards and handoffs without an approval transition.
