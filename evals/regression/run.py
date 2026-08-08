"""Offline regression runner for the Agent graph.

Placeholder contract for the evaluation pipeline. It loads a JSONL dataset and
reports gate results; deterministic graders are added under evals/graders/.
"""
import argparse
import asyncio
import json
from pathlib import Path


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def grade_row(row: dict) -> dict:
    """Deterministic grading stub; wire to evals.graders implementations."""
    passed = isinstance(row.get("id"), str) and isinstance(row.get("intent"), str)
    return {"id": row.get("id"), "pass": passed, "reasons": ["row shape validated"] if passed else ["missing id/intent"]}


async def run(dataset_path: Path) -> int:
    rows = load_dataset(dataset_path)
    results = [grade_row(row) for row in rows]
    passed = sum(1 for result in results if result["pass"])
    total = len(results)
    print(f"Evaluated {total} cases; {passed} passed")
    return 0 if total == 0 or passed == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Agent regression runner")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.dataset)))


if __name__ == "__main__":
    main()
