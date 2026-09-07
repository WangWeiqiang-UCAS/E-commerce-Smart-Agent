# Agent desk

Human-support workspace for pending agent actions. It uses only an HttpOnly
administrator session cookie and proxies queue reads/decisions through server
routes (`/api/session/login`, `/api/threads`, `/api/threads/[threadId]`).
State-changing decisions require same-origin browser requests.

The current adapter lists conversation threads and transcripts from the Agent
admin read API (`/api/v1/threads`), which is guarded by an administrator JWT.
Refund decisions remain in the commerce after-sales state machine; the desk
records audit cards and handoffs without an approval transition. Administrators
can take over a support conversation, send customer-facing support replies, and
return it to the Agent. Conversation ownership is persisted in the Agent
database and is intentionally separate from commerce order or after-sales data.

While owning a human handoff, an administrator can add an internal note or
transfer it to another active administrator. Both operations append a durable
conversation event. Internal notes have `sender_type=internal` and are filtered
from the customer history endpoint, so they never reach the customer widget.
Transfers are guarded against concurrent ownership changes and cannot target a
disabled or non-administrator account.

The desk restores an existing HttpOnly session on reload and reopens the last
selected thread from the URL hash. Thread IDs are restricted to opaque
alphanumeric, underscore, and hyphen identifiers; access tokens and customer
PII must never be used as thread IDs.

Agent messages can be copied into the reply editor as editable drafts. A human
must take over the thread and explicitly submit the final reply; the desk never
auto-sends an AI suggestion.
