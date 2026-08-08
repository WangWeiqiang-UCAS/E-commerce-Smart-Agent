# Agent evaluation

Evaluation is a first-class deliverable, not a post-hoc report. The three
directories are:

- `datasets/` — fixed, versioned question/answer/evidence cases used for
  offline regression (orders, policy citations, refund eligibility, and
  prompt-injection / over-permission attempts).
- `graders/` — deterministic and LLM-assisted graders for intent accuracy,
  citation correctness, parameter accuracy, refusal rate, and PII leakage.
- `regression/` — scripts that run the dataset against a built graph and fail
  CI when gates regress.

## Gates (target.md section 10)

- Cited-answer correctness >= 95%.
- Key tool parameter accuracy >= 98%.
- Unsupported-question refusal rate >= 98%.
- P95 first token < 2.5 s; P95 read-only customer request < 6 s.
- Unconfirmed write executions == 0.
- Cross-user order/refund access == 0 (see `test/test_security_boundaries.py`).
