"""Grade recorded Agent results against versioned evaluation cases.

The runner deliberately accepts recorded results rather than silently claiming
to execute an LLM. Production evaluation jobs must generate the result JSONL
from a pinned model/prompt/tool environment before invoking this command.
"""
import argparse
import asyncio
import json
from pathlib import Path

from evals.graders.deterministic import gate_summary, grade_case, percentile_95

GATES = {"citation_correctness": 0.95, "tool_parameter_accuracy": 0.98, "refusal_rate": 0.98}
MAX_P95_FIRST_TOKEN_MS = 2500
MAX_P95_READ_ONLY_MS = 6000


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


async def run(dataset_path: Path, results_path: Path) -> int:
    expected_rows = load_jsonl(dataset_path)
    result_rows = load_jsonl(results_path)
    results_by_id = {row.get("id"): row for row in result_rows}
    cases: list[tuple[dict, list[str]]] = []
    for expected in expected_rows:
        case_id = expected.get("id")
        actual = results_by_id.get(case_id)
        errors = ["missing recorded result"] if actual is None else grade_case(expected, actual)
        cases.append((expected, errors))
        print(f"{case_id}: {'PASS' if not errors else 'FAIL - ' + '; '.join(errors)}")
    extra = set(results_by_id) - {row.get("id") for row in expected_rows}
    if extra:
        print(f"FAIL - unexpected result IDs: {sorted(extra)}")
        return 1
    summary = gate_summary(cases)
    print("Gates: " + ", ".join(f"{name}={value:.1%}" for name, value in summary.items()))
    first_token = [float(row["first_token_ms"]) for row in result_rows if "first_token_ms" in row]
    read_only = [float(row["read_only_ms"]) for row in result_rows if "read_only_ms" in row]
    if len(first_token) != len(result_rows) or len(read_only) != len(result_rows):
        print("FAIL - every recorded result must include first_token_ms and read_only_ms")
        return 1
    p95_first_token = percentile_95(first_token)
    p95_read_only = percentile_95(read_only)
    print(f"Latency: p95_first_token_ms={p95_first_token:.0f}, p95_read_only_ms={p95_read_only:.0f}")
    gate_failed = any(summary[name] < threshold for name, threshold in GATES.items())
    latency_failed = p95_first_token >= MAX_P95_FIRST_TOKEN_MS or p95_read_only >= MAX_P95_READ_ONLY_MS
    return 0 if not gate_failed and not latency_failed and all(not errors for _, errors in cases) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade recorded Agent regression results")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.dataset, args.results)))


if __name__ == "__main__":
    main()
