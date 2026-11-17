"""Command line entry point: ``uv run harness skeleton``."""

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from harness.gsm8k import ensure_test_split, load_problem
from harness.machine import REPO_ROOT, LocalHost, describe_machine
from harness.runtimes import LocalAI, Ollama, Runtime, RuntimeRequestFailed, RuntimeUnavailable
from harness.skeleton import ModelFile, run_skeleton

CORE_MODEL = "qwen3.5-9b"
CORE_MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
# Generous: a 9B model with Thinking mode on a laptop GPU, or CPU-only, can take minutes.
REQUEST_TIMEOUT_S = 1800


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    commands = parser.add_subparsers(dest="command", required=True)
    skeleton = commands.add_parser("skeleton", help="One GSM8K problem through both Runtimes")
    skeleton.add_argument("--index", type=int, default=0, help="GSM8K test problem index")
    skeleton.add_argument("--localai-url", default="http://localhost:8080")
    skeleton.add_argument("--ollama-url", default="http://localhost:11434")
    skeleton.add_argument("--machine-id", help="Override the derived machine id")
    skeleton.add_argument(
        "--wait", type=int, default=600, help="Seconds to wait for the Runtimes to list the model"
    )
    args = parser.parse_args(argv)

    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as http:
        runtimes: list[Runtime] = [LocalAI(http, args.localai_url), Ollama(http, args.ollama_url)]
        try:
            _wait_until_ready(runtimes, args.wait)
            problem = load_problem(ensure_test_split(REPO_ROOT / "data" / "gsm8k" / "test.jsonl"), args.index)
            path = run_skeleton(
                runtimes=runtimes,
                model=CORE_MODEL,
                model_file=ModelFile(name=CORE_MODEL_FILE, sha256=_recorded_sha256(CORE_MODEL_FILE)),
                problem=problem,
                machine=describe_machine(LocalHost(), machine_id=args.machine_id),
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
