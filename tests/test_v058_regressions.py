"""Regression tests for v0.5.8: malformed numbers must parse-fail, not raise.

Issue: check()/acheck() raised on model responses whose numbers are
syntactically valid JSON but unrepresentable — int literals past
sys.get_int_max_str_digits() (ValueError from the json scanner), ints too
large for float() (OverflowError), and nesting deep enough to exhaust the
recursion stack (RecursionError). BOOTH's contract is that bad model
output becomes UNCERTAIN with method == "parse_failure" and never raises.
"""

import asyncio
import json

import pytest

import booth

BIG_CONFIDENCE_400 = '{"answer": "x", "confidence": ' + "9" * 400 + "}"
BIG_CONFIDENCE_5000 = '{"answer": "x", "confidence": ' + "9" * 5000 + "}"
BIG_ANSWER_5000 = '{"answer": ' + "1" * 5000 + ', "confidence": 0.9}'
DEEP_NEST_50000 = '{"a":' * 50000 + "1" + "}" * 50000

RAISING_PAYLOADS = [
    BIG_CONFIDENCE_400,
    BIG_CONFIDENCE_5000,
    BIG_ANSWER_5000,
    DEEP_NEST_50000,
]


def _assert_parse_failure(result, payload):
    assert result.status == booth.UNCERTAIN
    assert result.method == "parse_failure"
    assert result.call_failed is False
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.parse_ok is False
    assert attempt.call_failed is False
    assert attempt.error
    # The raw response is preserved for debugging either way.
    assert payload[:20] in attempt.raw_text


@pytest.mark.parametrize("payload", RAISING_PAYLOADS)
def test_check_malformed_numbers_return_parse_failure(payload):
    result = booth.check(
        lambda prompt: payload,
        "q",
        max_retries=0,
    )
    _assert_parse_failure(result, payload)


@pytest.mark.parametrize("payload", RAISING_PAYLOADS)
def test_acheck_malformed_numbers_return_parse_failure(payload):
    async def call_fn(prompt):
        return payload

    result = asyncio.run(booth.acheck(call_fn, "q", max_retries=0))
    _assert_parse_failure(result, payload)


@pytest.mark.parametrize("payload", RAISING_PAYLOADS)
def test_raising_parse_fn_returns_parse_failure(payload):
    """The same payloads through a parse_fn crash inside parse_fn itself."""
    result = booth.check(
        lambda prompt: "provider-native-payload",
        "q",
        parse_fn=lambda raw: json.loads(raw),
        max_retries=0,
    )
    _assert_parse_failure(result, "provider-native-payload")
    assert result.attempts[0].error.startswith("parse_fn raised ")


def test_parse_fn_huge_int_answer_string_coercion_parse_failure():
    """str() of an int over the digit limit raises ValueError on the
    parse_fn path, where the payload never went through json.loads."""
    # Built from bytes so the int exists before str() is attempted —
    # int("1"*5000) would raise inside parse_fn instead.
    huge_answer = int.from_bytes(b"\xff" * 2000, "big")  # ~4816 decimal digits

    def parse_fn(raw):
        return {"answer": huge_answer, "confidence": 0.9}

    result = booth.check(
        lambda prompt: "provider-native-payload",
        "q",
        parse_fn=parse_fn,
        max_retries=0,
    )
    _assert_parse_failure(result, "provider-native-payload")
    assert "'answer' could not be converted to a string" in result.attempts[0].error


def test_parse_fn_huge_int_confidence_overflow_parse_failure():
    """float() of an int past the double range raises OverflowError."""
    def parse_fn(raw):
        return {"answer": "x", "confidence": int("9" * 400)}

    result = booth.check(
        lambda prompt: "provider-native-payload",
        "q",
        parse_fn=parse_fn,
        max_retries=0,
    )
    _assert_parse_failure(result, "provider-native-payload")
    assert "'confidence' could not be converted to a float" in result.attempts[0].error


def test_check_huge_int_confidence_reports_float_error():
    """Within the digit limit json.loads succeeds, so the failure is the
    float conversion itself and the error names it."""
    result = booth.check(
        lambda prompt: BIG_CONFIDENCE_400,
        "q",
        max_retries=0,
    )
    _assert_parse_failure(result, BIG_CONFIDENCE_400)
    assert "'confidence' could not be converted to a float" in result.attempts[0].error


def test_retry_after_malformed_numbers_uses_parse_failure_prompt():
    prompts = []

    def call_fn(prompt):
        prompts.append(prompt)
        return BIG_CONFIDENCE_400

    result = booth.check(call_fn, "q", max_retries=1)

    assert len(prompts) == 2
    assert result.status == booth.UNCERTAIN
    assert result.method == "parse_failure"
    assert len(result.attempts) == 2
    assert all(not a.parse_ok for a in result.attempts)
    # The retry prompt is the existing parse-failure one, not the plain retry.
    assert "could not be parsed" in prompts[1]
