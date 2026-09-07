# Agent BFF

The BFF is the browser-facing boundary for the customer widget. It reads the
Java commerce session token from a secure, HttpOnly cookie; browser code never
receives that token. The Agent runtime resolves the authenticated Java member
through its narrow commerce identity projection.

It currently proxies:

- streaming chat to the Python Agent runtime;
- chat history and interrupt resume to the Agent runtime;
- refund preview and confirmation to Java commerce.
- pending-order cancellation preview and confirmation to Java commerce.
- authenticated private after-sales evidence upload, attachment, and download
  URL retrieval; the browser never receives object-storage credentials.

`POST /api/session/login` authenticates submitted credentials only against
Java commerce and stores its token in an HttpOnly cookie. `POST
/api/session/logout` clears that cookie. The browser never receives the token
in a response body.

All state-changing routes enforce the configured same-origin check and validate
request bodies with Zod. Set the values in `.env.example` through the deployment
secret store, then run `npm ci && npm run build`.
