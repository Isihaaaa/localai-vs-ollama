"""Generative answer extraction and scoring for numeric Benchmarks (ADR-0001)."""

import re
from decimal import Decimal, InvalidOperation

# A closed thinking block, or an unclosed one running to the end (truncated output).
_THINKING = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_NUMBER = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"
_ANSWER_LINE = re.compile(rf"Answer:\**\s*({_NUMBER})", re.IGNORECASE)
_ANY_NUMBER = re.compile(_NUMBER)


def extract_answer(response: str) -> Decimal | None:
    """Return the model's final numeric answer, or None if it gave none.

    Prefers the last explicit ``Answer: <number>`` line the prompt asks for and
    falls back to the last number in the response. Text inside the Thinking
    mode's ``<think>`` block is never considered.
    """
    # A lone closing tag means the chat template opened the block in the prompt.
    visible = _THINKING.sub("", response).rsplit("</think>", 1)[-1]
    answer_lines = _ANSWER_LINE.findall(visible)
    if answer_lines:
        return _to_decimal(answer_lines[-1])
    numbers = _ANY_NUMBER.findall(visible)
    return _to_decimal(numbers[-1]) if numbers else None


def is_correct(extracted: Decimal | None, reference: Decimal) -> bool:
    return extracted is not None and extracted == reference


def _to_decimal(text: str) -> Decimal | None:
    cleaned = re.sub(r"[\s$,]", "", text)
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None
