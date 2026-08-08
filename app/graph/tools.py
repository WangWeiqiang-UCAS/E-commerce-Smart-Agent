"""Tool registry.

Refund preview/confirmation is intentionally rendered by the BFF and submitted
to Java Commerce after explicit user confirmation. No commerce write tool is
available to an LLM.
"""

refund_tools: list = []
