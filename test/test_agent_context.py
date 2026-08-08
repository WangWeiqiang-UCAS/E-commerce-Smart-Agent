from app.core.agent_context import (
    commerce_access_token,
    reset_request_context,
    set_request_context,
    trace_id,
)


def test_commerce_token_is_request_scoped_and_resettable():
    assert commerce_access_token() is None
    assert trace_id() is None

    tokens = set_request_context("secret-commerce-token", "f9a9f7bc-ef2b-4ef0-9000-2f77142c3b6c")
    assert commerce_access_token() == "secret-commerce-token"
    assert trace_id() == "f9a9f7bc-ef2b-4ef0-9000-2f77142c3b6c"

    reset_request_context(tokens)
    assert commerce_access_token() is None
    assert trace_id() is None
