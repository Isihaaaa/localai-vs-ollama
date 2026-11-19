"""Command line entry point: ``uv run harness doctor`` / ``skeleton`` / ``gsm8k``."""

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from harness.benchmark_run import RunInterrupted, run_benchmark
from harness.doctor import format_checklist, run_checks
from harness.gsm8k import count_problems, ensure_test_split, load_problem, load_problems, select_problems
from harness.machine import REPO_ROOT, LocalHost, MachineDescription, describe_machine
from harness.model_file import ModelFile
from harness.runtimes import (
    LocalAI,
    Ollama,
    Runtime,
    RuntimeRequestFailed,
    RuntimeUnavailable,
    ThinkingMode,
)
from harness.skeleton import run_skeleton

CORE_MODEL = "qwen3.5-9b"
CORE_MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
# Generous: a 9B model with Thinking mode on a laptop GPU, or CPU-only, can take minutes.
REQUEST_TIMEOUT_S = 1800
# The doctor only asks for versions and model lists.
DOCTOR_TIMEOUT_S = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check the prerequisites of a run")
    _add_runtime_urls(doctor)
    skeleton = commands.add_parser("skeleton", help="One GSM8K problem through both Runtimes")
    skeleton.add_argument("--index", type=int, default=0, help="GSM8K test problem index")
    _add_run_arguments(skeleton)
    gsm8k = commands.add_parser(
        "gsm8k", help="GSM8K through both Runtimes; rerun the same command to resume an interrupted run"
    )
    gsm8k.add_argument(
        "--thinking",
        choices=[mode.value for mode in ThinkingMode],
        default=ThinkingMode.ON.value,
        help="Thinking mode: let the model reason before answering, or not (default: on)",
    )
    gsm8k.add_argument(
        "--subset", type=int, help="A fixed, reproducible subset of this many problems (default: full split)"
    )
    gsm8k.add_argument("--start", type=int, help="First GSM8K test problem index of a range")
    gsm8k.add_argument("--count", type=int, help="Number of problems in the range (default: all from --start)")
    gsm8k.add_argument(
        "--runtime",
        action="append",
        choices=[LocalAI.name, Ollama.name],
        help="Ask only this Runtime; repeat for several (default: all)",
    )
    _add_run_arguments(gsm8k)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "gsm8k":
        return _gsm8k(args)
    return _skeleton(args)


def _add_runtime_urls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--localai-url", default="http://localhost:8080")
    parser.add_argument("--ollama-url", default="http://localhost:11434")


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of every command that sends questions to the Runtimes."""
    _add_runtime_urls(parser)
    parser.add_argument("--machine-id", help="Override the derived machine id")
    parser.add_argument(
        "--wait", type=int, default=600, help="Seconds to wait for the Runtimes to list the model"
    )


def _runtimes(http: httpx.Client, args: argparse.Namespace) -> list[Runtime]:
    return [LocalAI(http, args.localai_url), Ollama(http, args.ollama_url)]


def _gsm8k_split() -> Path:
    return ensure_test_split(REPO_ROOT / "data" / "gsm8k" / "test.jsonl")


def _core_model_file() -> ModelFile:
    return ModelFile(name=CORE_MODEL_FILE, sha256=_recorded_sha256(CORE_MODEL_FILE))


def _machine(args: argparse.Namespace) -> MachineDescription:
    return describe_machine(LocalHost(), machine_id=args.machine_id)


def _doctor(args: argparse.Namespace) -> int:
    host = LocalHost()
    with httpx.Client(timeout=DOCTOR_TIMEOUT_S) as http:
        checks = run_checks(
            host,
            describe_machine(host),
            _runtimes(http, args),
            CORE_MODEL,
            disk_path=str(REPO_ROOT),
        )
    print(format_checklist(checks))
    if all(check.passed for check in checks):
        print("Ready to run.")
        return 0
    print("Some prerequisites are missing; fix the failed checks above.", file=sys.stderr)
    return 1


def _skeleton(args: argparse.Namespace) -> int:
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as http:
        runtimes = _runtimes(http, args)
        try:
            _wait_until_ready(runtimes, args.wait)
            problem = load_problem(_gsm8k_split(), args.index)
            path = run_skeleton(
                runtimes=runtimes,
                model=CORE_MODEL,
                model_file=_core_model_file(),
                problem=problem,
                machine=_machine(args),
                results_dir=REPO_ROOT / "results",
                raw_dir=REPO_ROOT / "runs",
                now=datetime.now(UTC),
            )
        except (RuntimeUnavailable, RuntimeRequestFailed) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

    print(f"Result written to {path.relative_to(REPO_ROOT)}")
    print(path.read_text(encoding="utf-8"))
    return 0


def _gsm8k(args: argparse.Namespace) -> int:
    split = _gsm8k_split()
    try:
        indices, selection = select_problems(
            count_problems(split), subset=args.subset, start=args.start, count=args.count
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    problems = load_problems(split, indices)
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as http:
        runtimes = [r for r in _runtimes(http, args) if not args.runtime or r.name in args.runtime]
        try:
            _wait_until_ready(runtimes, args.wait)
            path = run_benchmark(
                runtimes=runtimes,
                model=CORE_MODEL,
                model_file=_core_model_file(),
                thinking_mode=ThinkingMode(args.thinking),
                problems=problems,
                selection=selection,
                machine=_machine(args),
                results_dir=REPO_ROOT / "results",
                raw_dir=REPO_ROOT / "runs",
                now=datetime.now(UTC),
                progress=_print_progress,
            )
        except (RuntimeUnavailable, RuntimeRequestFailed, RunInterrupted) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Interrupted. Run the same command again to continue where it stopped.", file=sys.stderr)
            return 130

    print(f"Result written to {path.relative_to(REPO_ROOT)}")
    return 0


def _print_progress(line: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {line}", file=sys.stderr, flush=True)


def _wait_until_ready(runtimes: list[Runtime], timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    for runtime in runtimes:
        while True:
            try:
                if runtime.has_model(CORE_MODEL):
                    break
            except RuntimeUnavailable:
                if time.monotonic() > deadline:
                    raise
            if time.monotonic() > deadline:
                raise RuntimeUnavailable(
                    f"{runtime.name} is up but does not list the model '{CORE_MODEL}'. "
                    "Check `docker compose logs model-fetch ollama-model localai`."
                )
            print(f"Waiting for {runtime.name} to list '{CORE_MODEL}'...", file=sys.stderr)
            time.sleep(5)


def _recorded_sha256(model_file: str) -> str | None:
    """The checksum `model-fetch` recorded when it downloaded the Model file."""
    path = REPO_ROOT / ".state" / f"{model_file}.sha256"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


if __name__ == "__main__":
    sys.exit(main())
