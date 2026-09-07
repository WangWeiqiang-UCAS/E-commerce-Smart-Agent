# Regression runner

Run deterministic grading against a versioned dataset and recorded Agent
results:

```powershell
poetry run python -m evals.regression.run --dataset evals/datasets/agent_eval_sample.jsonl --results evals/regression/baseline_results.jsonl
```

The runner validates intent, tool names/parameters, required citations, refusal
behavior, unconfirmed write execution, and P95 `first_token_ms` / `read_only_ms`
against the target latency limits. It fails when a gate falls below the target.
`baseline_results.jsonl` is a checked-in fixture for grader regression;
it is not evidence of live model quality. Production evaluation must generate
new results from a pinned model/prompt/tool environment before grading. The CI
job `agent-evals` runs the fixture gate on every push.
