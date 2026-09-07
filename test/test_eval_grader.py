from evals.graders.deterministic import grade_case, percentile_95


def test_eval_grader_rejects_unconfirmed_write_execution():
    errors = grade_case(
        {"id": "refund", "intent": "REFUND", "requires_confirmation": True},
        {"id": "refund", "intent": "REFUND", "write_executed": True},
    )

    assert "write executed without explicit confirmation" in errors


def test_eval_grader_requires_expected_tool_arguments():
    errors = grade_case(
        {"id": "product", "intent": "GUIDE", "tool_calls": [{"tool": "search_products", "args": {"query": "shoe"}}]},
        {"id": "product", "intent": "GUIDE", "tool_calls": [{"tool": "search_products", "args": {"query": "bag"}}]},
    )

    assert "tool calls or parameters mismatch" in errors


def test_percentile_95_uses_the_conservative_upper_sample():
    assert percentile_95([100, 200, 300, 400, 500]) == 500
