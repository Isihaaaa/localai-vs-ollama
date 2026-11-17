"""GSM8K Benchmark: the official test split (1,319 problems)."""

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import httpx

TEST_SPLIT_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

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
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= index < len(lines):
        raise IndexError(f"GSM8K index {index} is out of range: the file has {len(lines)} problems")
    row = json.loads(lines[index])
    reference_text = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
    return Problem(index=index, question=row["question"], reference=Decimal(reference_text))


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
