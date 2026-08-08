# Tool schemas

JSON Schema for every tool exposed to the Agent, plus the write-action
preview/confirm payloads. The Agent model can only invoke tools listed here;
names, parameters, call budgets, timeouts, and retries are allow-listed by
`app/services/commerce_client.py` and the Java `AgentCommerceController`.

Source of truth: `../api-contracts/agent-commerce.openapi.yaml`.
