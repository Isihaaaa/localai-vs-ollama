"""Walking skeleton: one GSM8K problem through every Runtime, scored and recorded."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.answers import extract_answer, is_correct
from harness.gsm8k import Problem, build_prompt
from harness.machine import MachineDescription
from harness.runtimes import Runtime, message_content

RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str | None


def run_skeleton(
    *,
    runtimes: Sequence[Runtime],
    model: str,
    model_file: ModelFile,
    problem: Problem,
    machine: MachineDescription,
    results_dir: Path,
    raw_dir: Path,
    now: datetime,
) -> Path:
    """Ask each Runtime the problem in turn and write one result file.

    Runtimes are queried strictly one after the other, and each unloads the
    model before the next starts, so two copies never share the VRAM. Raw
    responses go to ``raw_dir`` (local only); the result file keeps only
    their checksum. Nothing is written to ``results_dir`` unless every
    Runtime answered.
    """
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    prompt = build_prompt(problem)
    versions: dict[str, str | None] = {}
    outcomes: list[dict[str, Any]] = []

    for runtime in runtimes:
        versions[runtime.name] = runtime.version()
        response = runtime.chat(model, prompt)
        runtime.unload(model)

        raw_path = raw_dir / machine.machine_id / f"{stamp}-gsm8k-{problem.index}-{runtime.name}.json"
        raw_bytes = json.dumps(response, indent=2, ensure_ascii=False).encode("utf-8")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw_bytes)

        extracted = extract_answer(message_content(response))
        outcomes.append(
            {
                "runtime": runtime.name,
                "extracted": None if extracted is None else str(extracted),
                "correct": is_correct(extracted, problem.reference),
                "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "walking-skeleton",
        "started_at": now.isoformat(),
        "model": model,
        "model_file": {"name": model_file.name, "sha256": model_file.sha256},
        # Proper Thinking mode handling arrives with the full GSM8K run (#4).
        "thinking_mode": "model-default",
        "problem": {
            "benchmark": "gsm8k",
            "index": problem.index,
            "reference": str(problem.reference),
        },
        "machine": {**machine.to_dict(), "runtime_versions": versions},
        "results": outcomes,
    }
    path = results_dir / machine.machine_id / f"{stamp}-gsm8k-skeleton.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
