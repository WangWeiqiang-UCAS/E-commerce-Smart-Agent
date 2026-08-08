# Graders

Grader contract: `grade(case, response) -> {"pass": bool, "reasons": [str]}`.

Deterministic graders:

- `intent_grader` — structured intent equals the dataset label.
- `citation_grader` — every required `source/version/chunk_id` appears in the
  answer when the case has `citations`.
- `tool_grader` — tool names and parameters match the expected allow-list
  exactly (bounding, ordering, and defaults included).
- `refusal_grader` — `refused` cases must not fabricate evidence or perform
  write actions.
- `pii_grader` — answers must not echo phone/email/ID/address/payment values.

LLM-assisted graders (documented, reviewed, and versioned in prompts/) score
answer helpfulness and citation correctness on sampled subsets.
