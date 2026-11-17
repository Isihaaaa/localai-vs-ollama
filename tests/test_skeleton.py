import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from harness.gsm8k import Problem
from harness.machine import Gpu, MachineDescription
from harness.runtimes import LocalAI, Ollama, RuntimeRequestFailed, RuntimeUnavailable
from harness.skeleton import ModelFile, run_skeleton

LOCALAI_URL = "http://localai:8080"
OLLAMA_URL = "http://ollama:11434"
MODEL = "qwen3.5-9b"

PROBLEM = Problem(
    index=0,
    question="Janet sells 9 eggs at $2 each. How much does she make?",
    reference=Decimal(18),
)
MACHINE = MachineDescription(
    machine_id="rtx-3080-test",
    os="Windows 11",
    cpu="i9",
    memory_gb=64,
    gpus=[Gpu("NVIDIA GeForce RTX 3080 Laptop GPU", 16384, "581.57")],
    script_version="0.1.0",
    script_commit="abc1234",
)
MODEL_FILE = ModelFile(name="Qwen3.5-9B-Q4_K_M.gguf", sha256="03b747")
NOW = datetime(2026, 11, 17, 15, 0, tzinfo=UTC)


def chat_reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeRuntimes:
    """Plays LocalAI and Ollama over HTTP and records every request in order."""

    def __init__(self, localai_answer: str, ollama_answer: str) -> None:
        self.answers = {"localai": localai_answer, "ollama": ollama_answer}
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.down: set[str] = set()
        self.failing: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        runtime = request.url.host
        if runtime in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        body = json.loads(request.content) if request.content else {}
        self.requests.append((runtime, request.url.path, body))
        routes: dict[tuple[str, str], Callable[[], httpx.Response]] = {
            ("localai", "/version"): lambda: httpx.Response(200, json={"version": "v4.12.1"}),
            ("ollama", "/api/version"): lambda: httpx.Response(200, json={"version": "0.40.16"}),
            ("localai", "/backend/shutdown"): lambda: httpx.Response(200, json={}),
            ("ollama", "/api/generate"): lambda: httpx.Response(200, json={"done": True}),
        }
        if request.url.path == "/v1/chat/completions" and runtime in self.failing:
            return httpx.Response(500, json={"error": "model failed to load"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=chat_reply(self.answers[runtime]))
        return routes[(runtime, request.url.path)]()

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def run(fake: FakeRuntimes, tmp_path: Path) -> Path:
    http = fake.client()
    return run_skeleton(
        runtimes=[LocalAI(http, LOCALAI_URL), Ollama(http, OLLAMA_URL)],
        model=MODEL,
        model_file=MODEL_FILE,
        problem=PROBLEM,
        machine=MACHINE,
        results_dir=tmp_path / "results",
        raw_dir=tmp_path / "runs",
        now=NOW,
    )


def test_records_correct_and_incorrect_per_runtime(tmp_path: Path) -> None:
    fake = FakeRuntimes(localai_answer="9 * 2 = 18\nAnswer: 18", ollama_answer="Answer: 17")

    result = json.loads(run(fake, tmp_path).read_text())

    outcomes = {r["runtime"]: (r["extracted"], r["correct"]) for r in result["results"]}
    assert outcomes == {"localai": ("18", True), "ollama": ("17", False)}
    assert result["problem"] == {"benchmark": "gsm8k", "index": 0, "reference": "18"}


def test_result_file_lands_under_the_machine_id_with_the_machine_description(
    tmp_path: Path,
) -> None:
    fake = FakeRuntimes(localai_answer="Answer: 18", ollama_answer="Answer: 18")

    path = run(fake, tmp_path)
    result = json.loads(path.read_text())

    assert path.parent == tmp_path / "results" / "rtx-3080-test"
    assert result["machine"]["cpu"] == "i9"
    assert result["machine"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 3080 Laptop GPU"
    assert result["machine"]["runtime_versions"] == {"localai": "v4.12.1", "ollama": "0.40.16"}
    assert result["model_file"] == {"name": "Qwen3.5-9B-Q4_K_M.gguf", "sha256": "03b747"}


def test_both_runtimes_get_the_same_model_and_prompt_at_temperature_zero(
    tmp_path: Path,
) -> None:
    fake = FakeRuntimes(localai_answer="Answer: 18", ollama_answer="Answer: 18")

    run(fake, tmp_path)

    chats = [body for _, path, body in fake.requests if path == "/v1/chat/completions"]
    assert len(chats) == 2
    assert chats[0]["model"] == chats[1]["model"] == MODEL
    assert chats[0]["messages"] == chats[1]["messages"]
    assert PROBLEM.question in chats[0]["messages"][0]["content"]
    assert chats[0]["temperature"] == chats[1]["temperature"] == 0


def test_runtimes_are_queried_one_after_the_other_and_unloaded_in_between(
    tmp_path: Path,
) -> None:
    fake = FakeRuntimes(localai_answer="Answer: 18", ollama_answer="Answer: 18")

    run(fake, tmp_path)

    calls = [(runtime, path) for runtime, path, _ in fake.requests if "version" not in path]
    assert calls == [
        ("localai", "/v1/chat/completions"),
        ("localai", "/backend/shutdown"),
        ("ollama", "/v1/chat/completions"),
        ("ollama", "/api/generate"),
    ]
    ollama_unload = next(body for _, path, body in fake.requests if path == "/api/generate")
    assert ollama_unload == {"model": MODEL, "keep_alive": 0}


def test_raw_responses_stay_local_and_only_their_checksum_is_in_the_result(
    tmp_path: Path,
) -> None:
    secret_reasoning = "<think>a long private chain of thought</think>Answer: 18"
    fake = FakeRuntimes(localai_answer=secret_reasoning, ollama_answer="Answer: 18")

    result_text = run(fake, tmp_path).read_text()

    assert "private chain of thought" not in result_text
    raw_files = list((tmp_path / "runs").rglob("*.json"))
    assert len(raw_files) == 2
    localai = next(r for r in json.loads(result_text)["results"] if r["runtime"] == "localai")
    raw = next(f for f in raw_files if "localai" in f.name)
    assert localai["raw_response_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert "private chain of thought" in raw.read_text()


def test_an_unreachable_runtime_stops_the_run_without_a_result_file(tmp_path: Path) -> None:
    fake = FakeRuntimes(localai_answer="Answer: 18", ollama_answer="Answer: 18")
    fake.down.add("ollama")

    with pytest.raises(RuntimeUnavailable, match="ollama"):
        run(fake, tmp_path)

    assert not (tmp_path / "results").exists()


def test_a_runtime_error_response_stops_the_run_with_its_message(tmp_path: Path) -> None:
    fake = FakeRuntimes(localai_answer="Answer: 18", ollama_answer="Answer: 18")
    fake.failing.add("localai")

    with pytest.raises(RuntimeRequestFailed, match="localai.*500.*model failed to load"):
        run(fake, tmp_path)

    assert not (tmp_path / "results").exists()
