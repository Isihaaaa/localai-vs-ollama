"""GSM8K Benchmark: the official test split (1,319 problems)."""

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import httpx

TEST_SPLIT_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

# Fixed for good: changing it changes every subset, and old results could no longer be compared.
SUBSET_SEED = 20261116

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    "End your response with a final line in exactly this format:\n"
    "Answer: <number>\n\n"
    "Problem: {question}"
)


@dataclass(frozen=True)
class Problem:
    index: int
    question: str
    reference: Decimal


def load_problem(path: Path, index: int) -> Problem:
    return load_problems(path, [index])[0]


def load_problems(path: Path, indices: Sequence[int]) -> list[Problem]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = []
    for index in indices:
        if not 0 <= index < len(lines):
            raise IndexError(f"GSM8K index {index} is out of range: the file has {len(lines)} problems")
        row = json.loads(lines[index])
        reference_text = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        problems.append(Problem(index=index, question=row["question"], reference=Decimal(reference_text)))
    return problems


def count_problems(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def subset_indices(total: int, size: int) -> list[int]:
    """A fixed, reproducible subset of ``size`` problems out of ``total``, in index order.

    The split is shuffled once with a fixed seed and the first ``size``
    problems are taken, so every machine gets the same subset, and a smaller
    subset is always part of a larger one.
    """
    if not 0 < size <= total:
        raise ValueError(f"A subset needs between 1 and {total} problems, not {size}")
    order = list(range(total))
    random.Random(SUBSET_SEED).shuffle(order)
    return sorted(order[:size])


def select_problems(
    total: int, *, subset: int | None = None, start: int | None = None, count: int | None = None
) -> tuple[list[int], str]:
    """The problem indices to run and a label naming the selection for the result file.

    No options: the full split. ``subset``: a fixed subset of that size.
    ``start``/``count``: a contiguous range (``count`` defaults to the rest).
    """
    if subset is not None:
        if start is not None or count is not None:
            raise ValueError("Pick either a subset or a range (start/count), not both")
        return subset_indices(total, subset), f"subset-{subset}"
    if start is None and count is None:
        return list(range(total)), "full"
    first = start or 0
    end = total if count is None else min(first + count, total)
    if not 0 <= first < end:
        raise ValueError(f"No problems to run: the range starts at {first} in a {total}-problem split")
    return list(range(first, end)), f"range-{first}-{end - 1}"


def build_prompt(problem: Problem) -> str:
    return PROMPT_TEMPLATE.format(question=problem.question)


def ensure_test_split(path: Path) -> Path:
    """Download the GSM8K test split to ``path`` unless it is already there."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(TEST_SPLIT_URL, follow_redirects=True, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path
