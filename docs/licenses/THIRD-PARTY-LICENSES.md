# Third-party license posture

| Component | License | Notes |
| --- | --- | --- |
| macrozheng/mall (Spring Boot 3.5) | Apache-2.0 | retained in `apps/commerce-java/LICENSE` |
| FastAPI / Starlette / Uvicorn | MIT / BSD-3-Clause | Agent runtime |
| LangGraph / LangChain | MIT | Agent workflow |
| SQLModel / SQLAlchemy | MIT | Agent persistence |
| pgvector | MIT | vector extension |
| Next.js / React | MIT | BFF, widget, desk |
| assistant-ui (planned) | MIT | chat UI when adopted |
| RabbitMQ client (kombu) | BSD-3-Clause | event consumer |
| OpenTelemetry | Apache-2.0 | observability |
| Langfuse OSS core | MIT | ee/ separately licensed |

Regenerate `third_party/SBOM` from the lock files before each release; the table
above is a summary, not a complete audit.
