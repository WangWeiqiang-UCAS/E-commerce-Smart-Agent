# Datasets

Each row is JSONL with a stable `id`, `question`, `intent`, expected `citations`
or `tool_calls`, and optional `refused` flag. Dataset versions are immutable;
changes create a new file (`policy-v1.jsonl`, `policy-v2.jsonl`).

`agent_eval_sample.jsonl` is a small seed set. Expand it before measuring the
production gates.
