"""Walking skeleton: one GSM8K problem through every Runtime, scored and recorded."""

import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.gsm8k import Problem, build_prompt
from harness.machine import MachineDescription
from harness.model_file import ModelFile
from harness.results import (
    machine_field,
    model_fields,
    score_response,
    stamp,
    tokens_per_second,
    write_result,
)
from harness.runtimes import Runtime, ThinkingMode
from harness.sampling import VENDOR

# The skeleton only shows that the pipeline works, so it uses the defaults of a real run.
THINKING_MODE = ThinkingMode.ON
SAMPLING = VENDOR

# 2: per-Runtime performance (load and generation time, token counts, throughput).
RESULT_SCHEMA_VERSION = 2


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
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Ask each Runtime the problem in turn and write one result file.

    Runtimes are queried strictly one after the other, and each unloads the
    model before the next starts, so two copies never share the VRAM. Raw
    responses go to ``raw_dir`` (local only); the result file keeps only
    their checksum. Nothing is written to ``results_dir`` unless every
    Runtime answered.

    Each Runtime first loads the model with a one-token warm-up request, so
    the timed request measures generation only. ``clock`` is a monotonic
    seconds counter, injectable for tests.
    """
    started = stamp(now)
    prompt = build_prompt(problem)
    versions: dict[str, str | None] = {}
    outcomes: list[dict[str, Any]] = []

    for runtime in runtimes:
        versions[runtime.name] = runtime.version()
        load_started = clock()
        runtime.load(model)
        load_seconds = clock() - load_started
        generation_started = clock()
        response = runtime.chat(model, prompt, THINKING_MODE, SAMPLING)
        generation_seconds = clock() - generation_started
        runtime.unload(model)

        scored = score_response(response, problem.reference)
        raw_path = raw_dir / machine.machine_id / f"{started}-gsm8k-{problem.index}-{runtime.name}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(scored.raw_bytes)
        outcomes.append(
            {
                "runtime": runtime.name,
                "extracted": scored.extracted,
                "correct": scored.correct,
                "raw_response_sha256": scored.sha256,
                "performance": _performance(response, load_seconds, generation_seconds),
            }
        )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "walking-skeleton",
        "started_at": now.isoformat(),
        **model_fields(model, model_file, THINKING_MODE, SAMPLING),
        "problem": {
            "benchmark": "gsm8k",
            "index": problem.index,
            "reference": str(problem.reference),
        },
        "machine": machine_field(machine, versions),
        "results": outcomes,
    }
    return write_result(results_dir / machine.machine_id / f"{started}-gsm8k-skeleton.json", result)


def _performance(response: dict[str, Any], load_seconds: float, generation_seconds: float) -> dict[str, Any]:
    """Timing and token counts of one Runtime's answer.

    ``completion_tokens`` includes the Thinking mode's reasoning tokens.
    """
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return {
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second(completion_tokens, generation_seconds),
    }
