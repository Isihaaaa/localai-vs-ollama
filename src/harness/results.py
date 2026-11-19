"""What every kind of run shares when it scores answers and writes a result file."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from harness.answers import extract_answer, is_correct
from harness.machine import MachineDescription
from harness.model_file import ModelFile
from harness.runtimes import message_content

STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def stamp(moment: datetime) -> str:
    """The time prefix of result and raw-response file names."""
    return moment.strftime(STAMP_FORMAT)


@dataclass(frozen=True)
class ScoredResponse:
    """One Runtime response: its raw bytes as stored locally, and how it scored."""

    raw_bytes: bytes
    extracted: str | None
    correct: bool

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()


def score_response(response: dict[str, Any], reference: Decimal) -> ScoredResponse:
    """Serialize a response for local storage and score its visible answer (ADR-0001)."""
    raw_bytes = json.dumps(response, indent=2, ensure_ascii=False).encode("utf-8")
    extracted = extract_answer(message_content(response))
    return ScoredResponse(
        raw_bytes=raw_bytes,
        extracted=None if extracted is None else str(extracted),
        correct=is_correct(extracted, reference),
    )


def model_fields(model: str, model_file: ModelFile) -> dict[str, Any]:
    """The result-file fields naming what was run."""
    return {
        "model": model,
        "model_file": model_file.to_dict(),
        # Proper Thinking mode handling arrives with the full GSM8K run (#4).
        "thinking_mode": "model-default",
    }


def machine_field(machine: MachineDescription, runtime_versions: dict[str, str | None]) -> dict[str, Any]:
    return {**machine.to_dict(), "runtime_versions": runtime_versions}


def tokens_per_second(completion_tokens: int | None, seconds: float) -> float | None:
    """Throughput: completion tokens (reasoning included) over wall time, prompt processing included."""
    if completion_tokens and seconds > 0:
        return round(completion_tokens / seconds, 2)
    return None


def write_result(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
