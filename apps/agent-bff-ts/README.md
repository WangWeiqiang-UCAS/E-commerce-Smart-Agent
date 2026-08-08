# Agent BFF

The BFF is the browser-facing boundary for the customer widget and the support
desk. It reads separate secure, HttpOnly session cookies for the Agent runtime
and Java commerce system. Browser code never receives either token.

It currently proxies:

- streaming chat to the Python Agent runtime;
- chat history and interrupt resume to the Agent runtime;
- refund preview and confirmation to Java commerce.

`POST /api/session/login` authenticates the submitted credentials against both
the Python Agent runtime and Java commerce, then stores their raw tokens only
in separate HttpOnly cookies. `POST /api/session/logout` clears both cookies.
The browser never receives either token in a response body.

All state-changing routes enforce the configured same-origin check and validate
request bodies with Zod. Set the values in `.env.example` through the deployment
secret store, then run `npm ci && npm run build`.
