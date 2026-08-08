# Regression runner

Run offline against a compiled graph with an in-memory checkpointer and a
stubbed commerce client:

```powershell
poetry run python -m evals.regression.run --dataset evals/datasets/agent_eval_sample.jsonl
```

The runner fails when any gate threshold drops below the target. The CI job
`agent-evals` (add when the dataset grows) runs this on every push.
