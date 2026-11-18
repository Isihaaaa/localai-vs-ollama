"""Resumable, fault-tolerant Benchmark runs.

A run keeps its progress in a journal under ``raw_dir`` (local only, ignored
by git), next to the raw model responses. Starting the same run again reads
the journal and asks only the questions that have no answer yet, so a run
killed in the middle of the night continues where it stopped.

Transient errors (connection drops, timeouts, HTTP 408/429/5xx) are retried
with backoff. A question that still fails is recorded as failed and the run
goes on; a Runtime that stays unreachable stops the run, keeping the journal
for the next attempt. What goes to ``results_dir`` is aggregates and a
checksum of the raw responses, written once every question has an outcome.
"""

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from harness.answers import extract_answer, is_correct
from harness.gsm8k import Problem, build_prompt
from harness.machine import MachineDescription
from harness.runtimes import (
    MAX_TOKENS,
    Runtime,
    RuntimeRequestFailed,
    RuntimeTimeout,
    RuntimeUnavailable,
    message_content,
)
from harness.skeleton import ModelFile

RUN_SCHEMA_VERSION = 1
BENCHMARK = "gsm8k"

# Waits before each retry: a restarting container usually answers within two minutes.
DEFAULT_BACKOFF_SECONDS = (10.0, 30.0, 90.0)
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

T = TypeVar("T")


class RunInterrupted(RuntimeError):
    """A Runtime stayed unreachable; the progress so far is kept for resuming."""


def run_benchmark(
    *,
    runtimes: Sequence[Runtime],
    model: str,
    model_file: ModelFile,
    problems: Sequence[Problem],
    machine: MachineDescription,
    results_dir: Path,
    raw_dir: Path,
    now: datetime,
    backoff_seconds: Sequence[float] = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Ask every Runtime every problem, resuming a previous attempt if there is one.

    The run is identified by its Run configuration (model, Model file,
    Runtimes and the exact prompts), so rerunning the same command resumes
    it. Runtimes are queried one after the other and each unloads the model
    before the next starts. Returns the path of the result file.
    """
    run_dir = raw_dir / machine.machine_id / _run_id(model, model_file, runtimes, problems)
    manifest = _Manifest.open(run_dir / "manifest.json", now)
    journal = _Journal(run_dir / "journal.jsonl")
    retrying = _Retrying(backoff_seconds, sleep)

    for runtime in runtimes:
        pending = [p for p in problems if journal.status(runtime.name, p.index) != "answered"]
        if not pending:
            continue
        try:
            manifest.record_version(runtime.name, retrying(runtime.version))
            load_started = clock()
            retrying(lambda: runtime.load(model))
            journal.append({"type": "load", "runtime": runtime.name, "seconds": round(clock() - load_started, 3)})
            for problem in pending:
                _ask(runtime, model, problem, run_dir, journal, retrying, clock)
            retrying(lambda: runtime.unload(model))
        except RuntimeUnavailable as error:
            raise RunInterrupted(
                f"{error} The run stopped; its progress is kept in {run_dir}. "
                "Run the same command again to continue where it stopped."
            ) from error

    result = _result(
        manifest, journal, runtimes, problems, model, model_file, machine, run_dir
    )
    stamp = manifest.started_at.strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / machine.machine_id / f"{stamp}-{BENCHMARK}-{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _ask(
    runtime: Runtime,
    model: str,
    problem: Problem,
    run_dir: Path,
    journal: "_Journal",
    retrying: "_Retrying",
    clock: Callable[[], float],
) -> None:
    prompt = build_prompt(problem)

    def timed_chat() -> tuple[dict[str, Any], float]:
        # Timed per attempt, so failed attempts and backoff waits do not count as generation time.
        started = clock()
        response = runtime.chat(model, prompt)
        return response, clock() - started

    try:
        response, generation_seconds = retrying(timed_chat)
    except (RuntimeTimeout, RuntimeRequestFailed) as error:
        # The Runtime is up, but this question keeps failing: record it and go on.
        journal.append(
            {
                "type": "question",
                "runtime": runtime.name,
                "index": problem.index,
                "status": "failed",
                "error": str(error),
                "attempts": retrying.attempts,
            }
        )
        return

    raw_bytes = json.dumps(response, indent=2, ensure_ascii=False).encode("utf-8")
    _write_atomically(run_dir / runtime.name / f"{problem.index}.json", raw_bytes)
    extracted = extract_answer(message_content(response))
    usage = response.get("usage") or {}
    journal.append(
        {
            "type": "question",
            "runtime": runtime.name,
            "index": problem.index,
            "status": "answered",
            "extracted": None if extracted is None else str(extracted),
            "correct": is_correct(extracted, problem.reference),
            "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "generation_seconds": round(generation_seconds, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    )


class _Retrying:
    """Calls a Runtime, retrying transient errors after each backoff wait."""

    def __init__(self, backoff_seconds: Sequence[float], sleep: Callable[[float], None]) -> None:
        self._backoff = list(backoff_seconds)
        self._sleep = sleep
        self.attempts = 0
        """How many attempts the latest call made, the failed ones included."""

    def __call__(self, call: Callable[[], T]) -> T:
        self.attempts = 0
        for wait in [*self._backoff, None]:
            self.attempts += 1
            try:
                return call()
            except (RuntimeUnavailable, RuntimeRequestFailed) as error:
                if wait is None or not _is_transient(error):
                    raise
            self._sleep(wait)
        raise AssertionError("unreachable: the last attempt returns or raises")


def _is_transient(error: Exception) -> bool:
    if isinstance(error, RuntimeRequestFailed):
        return error.status_code in TRANSIENT_STATUS_CODES
    return isinstance(error, RuntimeUnavailable)


class _Journal:
    """Append-only record of a run's progress, one JSON object per line.

    Every line is flushed to disk before the next question starts. A line
    cut short by a kill is dropped when the journal is read back.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self.records: list[dict[str, Any]] = []
        if path.exists():
            self._read()

    def _read(self) -> None:
        text = self._path.read_text(encoding="utf-8")
        lines = text.split("\n")
        complete = lines[:-1]  # the part after the last newline is empty or cut short
        self.records = [json.loads(line) for line in complete if line.strip()]
        if lines[-1]:
            _write_atomically(self._path, "".join(line + "\n" for line in complete).encode("utf-8"))

    def append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        self.records.append(record)

    def outcomes(self, runtime: str) -> dict[int, dict[str, Any]]:
        """The latest outcome of every question this Runtime was asked."""
        return {
            r["index"]: r for r in self.records if r["type"] == "question" and r["runtime"] == runtime
        }

    def status(self, runtime: str, index: int) -> str | None:
        outcome = self.outcomes(runtime).get(index)
        return None if outcome is None else str(outcome["status"])

    def last_load_seconds(self, runtime: str) -> float | None:
        loads = [r["seconds"] for r in self.records if r["type"] == "load" and r["runtime"] == runtime]
        return loads[-1] if loads else None


class _Manifest:
    """When the run first started, and the Runtime versions it saw."""

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self._path = path
        self._data = data

    @classmethod
    def open(cls, path: Path, now: datetime) -> "_Manifest":
        if path.exists():
            return cls(path, json.loads(path.read_text(encoding="utf-8")))
        manifest = cls(path, {"started_at": now.isoformat(), "runtime_versions": {}})
        manifest._save()
        return manifest

    @property
    def started_at(self) -> datetime:
        return datetime.fromisoformat(self._data["started_at"])

    @property
    def runtime_versions(self) -> dict[str, str | None]:
        versions: dict[str, str | None] = self._data["runtime_versions"]
        return versions

    def record_version(self, runtime: str, version: str | None) -> None:
        self._data["runtime_versions"][runtime] = version
        self._save()

    def _save(self) -> None:
        _write_atomically(self._path, (json.dumps(self._data, indent=2) + "\n").encode("utf-8"))


def _write_atomically(path: Path, content: bytes) -> None:
    """Write ``path`` so that a kill or power loss leaves either the old or the new content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def _run_id(model: str, model_file: ModelFile, runtimes: Sequence[Runtime], problems: Sequence[Problem]) -> str:
    configuration = {
        "benchmark": BENCHMARK,
        "model": model,
        "model_file": [model_file.name, model_file.sha256],
        "runtimes": [r.name for r in runtimes],
        "max_tokens": MAX_TOKENS,
        "prompts": [(p.index, build_prompt(p)) for p in problems],
    }
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:12]
    return f"{BENCHMARK}-{model}-{digest}"


def _result(
    manifest: _Manifest,
    journal: _Journal,
    runtimes: Sequence[Runtime],
    problems: Sequence[Problem],
    model: str,
    model_file: ModelFile,
    machine: MachineDescription,
    run_dir: Path,
) -> dict[str, Any]:
    indices = [p.index for p in problems]
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": "benchmark-run",
        "run_id": run_dir.name,
        "started_at": manifest.started_at.isoformat(),
        "model": model,
        "model_file": {"name": model_file.name, "sha256": model_file.sha256},
        # Proper Thinking mode handling arrives with the full GSM8K run (#4).
        "thinking_mode": "model-default",
        "benchmark": {"name": BENCHMARK, "questions": len(problems)},
        "machine": {**machine.to_dict(), "runtime_versions": manifest.runtime_versions},
        "results": [_aggregate(journal, runtime.name, indices) for runtime in runtimes],
    }


def _aggregate(journal: _Journal, runtime: str, indices: Sequence[int]) -> dict[str, Any]:
    all_outcomes = journal.outcomes(runtime)
    outcomes = [all_outcomes[i] for i in sorted(indices)]
    answered = [o for o in outcomes if o["status"] == "answered"]
    failed = [o for o in outcomes if o["status"] == "failed"]
    correct = sum(1 for o in answered if o["correct"])
    checksum_lines = "".join(f"{o['index']} {o['raw_response_sha256']}\n" for o in answered)
    return {
        "runtime": runtime,
        "answered": len(answered),
        "correct": correct,
        "failed": len(failed),
        "accuracy": round(correct / len(outcomes), 4) if outcomes else None,
        "failures": [{"index": o["index"], "error": o["error"], "attempts": o["attempts"]} for o in failed],
        "raw_responses_sha256": hashlib.sha256(checksum_lines.encode("utf-8")).hexdigest(),
        "performance": _performance(answered, journal.last_load_seconds(runtime)),
    }


def _performance(answered: Sequence[dict[str, Any]], load_seconds: float | None) -> dict[str, Any]:
    """Totals over the answered questions; throughput is total completion tokens over total wall time."""
    generation_seconds = sum(o["generation_seconds"] for o in answered)
    counted = [o for o in answered if o["completion_tokens"] is not None]
    completion_tokens = sum(o["completion_tokens"] for o in counted) if counted else None
    counted_seconds = sum(o["generation_seconds"] for o in counted)
    return {
        "load_seconds": load_seconds,
        "generation_seconds": round(generation_seconds, 3),
        "prompt_tokens": sum(o["prompt_tokens"] or 0 for o in answered) if counted else None,
        "completion_tokens": completion_tokens,
        "tokens_per_second": (
            round(completion_tokens / counted_seconds, 2)
            if completion_tokens and counted_seconds > 0
            else None
        ),
    }
