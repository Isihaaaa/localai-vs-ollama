import json
from decimal import Decimal
from pathlib import Path

import pytest

from harness.gsm8k import load_problem


@pytest.fixture
def test_split(tmp_path: Path) -> Path:
    rows = [
        {
            "question": "Janet has 3 apples and buys 4 more. How many?",
            "answer": "3 + 4 = <<3+4=7>>7\n#### 7",
        },
        {
            "question": "A shop sells 1,000 pens and 80 pencils. How many items?",
            "answer": "1000 + 80 = <<1000+80=1080>>1080\n#### 1,080",
        },
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_loads_the_question_and_reference_answer_by_index(test_split: Path) -> None:
    problem = load_problem(test_split, index=1)
    assert problem.index == 1
    assert problem.question == "A shop sells 1,000 pens and 80 pencils. How many items?"
    assert problem.reference == Decimal(1080)


def test_out_of_range_index_is_a_clear_error(test_split: Path) -> None:
    with pytest.raises(IndexError, match="index 5.*2 problems"):
        load_problem(test_split, index=5)
