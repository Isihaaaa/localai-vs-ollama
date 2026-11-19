import json
from decimal import Decimal
from pathlib import Path

import pytest

from harness.gsm8k import load_problem, select_problems, subset_indices


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


def test_a_subset_is_the_same_on_every_machine_and_spread_over_the_split() -> None:
    subset = subset_indices(total=1319, size=100)

    assert subset == subset_indices(total=1319, size=100)
    assert subset == sorted(set(subset)) and len(subset) == 100
    assert subset != list(range(100))  # not just the first problems
    assert all(0 <= i < 1319 for i in subset)


def test_a_smaller_subset_is_contained_in_a_larger_one() -> None:
    assert set(subset_indices(total=1319, size=50)) <= set(subset_indices(total=1319, size=500))


def test_a_subset_cannot_be_larger_than_the_split() -> None:
    with pytest.raises(ValueError, match="1319"):
        subset_indices(total=1319, size=2000)


def test_without_options_the_full_split_is_selected() -> None:
    assert select_problems(total=1319) == (list(range(1319)), "full")


def test_a_subset_is_selected_by_its_size() -> None:
    assert select_problems(total=1319, subset=100) == (subset_indices(1319, 100), "subset-100")


def test_a_range_is_selected_by_start_and_count() -> None:
    assert select_problems(total=1319, start=10, count=3) == ([10, 11, 12], "range-10-12")
    assert select_problems(total=1319, start=1300) == (list(range(1300, 1319)), "range-1300-1318")


def test_a_subset_and_a_range_cannot_be_combined() -> None:
    with pytest.raises(ValueError, match="subset"):
        select_problems(total=1319, subset=100, start=5)
