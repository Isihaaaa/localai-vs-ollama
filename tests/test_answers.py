from decimal import Decimal

import pytest

from harness.answers import extract_answer, is_correct


def test_reads_the_explicit_answer_line() -> None:
    response = "She has 3 + 4 = 7 apples, then buys 11 more.\nAnswer: 18"
    assert extract_answer(response) == Decimal(18)


def test_ignores_answer_lines_inside_the_thinking_block() -> None:
    response = (
        "<think>Maybe it is 12?\nAnswer: 12\nNo, I forgot the 6 eggs.</think>\n"
        "16 - 3 - 4 = 9 eggs, 9 * 2 = 18 dollars.\nAnswer: 18"
    )
    assert extract_answer(response) == Decimal(18)


def test_falls_back_to_the_last_number_when_the_answer_line_is_missing() -> None:
    response = "<think>It could be 99.</think>She makes 9 * 2 = 18 dollars every day."
    assert extract_answer(response) == Decimal(18)


def test_returns_none_when_the_response_has_no_number() -> None:
    assert extract_answer("<think>Hmm, 42?</think>I cannot solve this.") is None


def test_ignores_reasoning_when_the_template_already_opened_the_thinking_block() -> None:
    # Qwen3-family chat templates can put "<think>" into the prompt, so the
    # response starts inside the thinking block and only closes it.
    response = "First guess 12, then 99 is wrong too.</think>\nShe makes 18 dollars."
    assert extract_answer(response) == Decimal(18)


def test_ignores_an_unclosed_thinking_block_from_a_truncated_response() -> None:
    assert extract_answer("<think>Step 1: 5 * 3 = 15, then") is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Answer: 1,080", Decimal(1080)),
        ("Answer: $18", Decimal(18)),
        ("Answer: -4", Decimal(-4)),
        ("**Answer:** 2.5", Decimal("2.5")),
        ("answer: 7.", Decimal(7)),
    ],
)
def test_normalizes_how_the_number_is_written(response: str, expected: Decimal) -> None:
    assert extract_answer(response) == expected


def test_the_same_value_written_differently_is_correct() -> None:
    assert is_correct(extract_answer("Answer: 18.00"), Decimal(18))


def test_a_different_value_is_incorrect() -> None:
    assert not is_correct(extract_answer("Answer: 17"), Decimal(18))


def test_no_extractable_answer_is_incorrect() -> None:
    assert not is_correct(None, Decimal(18))
