# API documentation

- `packages/api-contracts/agent-commerce.openapi.yaml` — the Java commerce
  anti-corruption boundary (read tools + refund preview/confirm).
- FastAPI runtime: `/docs` on the Agent service for chat, resume, history,
  auth, status, and admin thread APIs.
- BFF routes are thin proxies documented in `apps/agent-bff-ts/README.md`.

Contract changes must update the OpenAPI file and pass an OpenAPI diff review
before merging.
