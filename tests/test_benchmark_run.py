import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from harness.benchmark_run import RunInterrupted, run_benchmark
from harness.gsm8k import Problem
from harness.model_file import ModelFile
from harness.runtimes import LocalAI, Ollama, Runtime, ThinkingMode, has_reasoning
from harness.sampling import GREEDY, VENDOR, SamplingProfile

from fakes import LOCALAI_URL, MACHINE, MODEL, MODEL_FILE, OLLAMA_URL, chat_reply

PROBLEMS = [
    Problem(index=i, question=f"Question {i}: what is {i} + {i}?", reference=Decimal(2 * i))
    for i in range(5)
]
FIRST_START = datetime(2026, 11, 18, 22, 0, tzinfo=UTC)
RESTART = datetime(2026, 11, 19, 7, 30, tzinfo=UTC)

# One attempt plus three retries, waiting longer before each.
BACKOFF = [10.0, 30.0, 90.0]


def question_index(body: dict[str, Any]) -> int:
    content: str = body["messages"][0]["content"]
    return int(content.split("Question ", 1)[1].split(":", 1)[0])


Fault = Callable[[httpx.Request], httpx.Response]


class FakeRuntimes:
    """Plays LocalAI and Ollama over HTTP.

    Every Runtime answers question ``i`` with ``2 * i`` unless ``wrong`` says
    otherwise. ``faults`` maps (runtime, question index) to a list of responses
    or exceptions that are played, one per request, before the real answer.
    ``kill_after`` stops the process (as Ctrl+C or a crash would) once that many
    questions have been answered. ``seconds`` is a fake clock: loading the
    model takes ``load_seconds`` and every answer one second.
    """

    def __init__(self) -> None:
        self.asked: list[tuple[str, int]] = []
        self.wrong: set[tuple[str, int]] = set()
        self.faults: dict[tuple[str, int], list[httpx.Response | Exception]] = {}
        self.down: set[str] = set()
        self.kill_after: int | None = None
        self.truncated: set[tuple[str, int]] = set()
        self.reasoned: set[tuple[str, int]] = set()
        self.seconds = 0.0
        self.load_seconds = 9.0
        self.progress: list[str] = []
        self.chats: list[tuple[str, dict[str, Any]]] = []
        """Every question request (not warm-ups) with its JSON body."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        runtime = request.url.host
        if runtime in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path in ("/version", "/api/version"):
            return httpx.Response(200, json={"version": "1.0"})
        if path in ("/backend/shutdown", "/api/generate"):
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        if body.get("max_tokens") == 1:  # warm-up
            self.seconds += self.load_seconds
            return httpx.Response(200, json=chat_reply("Hello"))
        index = question_index(body)
        self.chats.append((runtime, body))
        pending = self.faults.get((runtime, index))
        if pending:
            fault = pending.pop(0)
            if isinstance(fault, Exception):
                raise fault
            return fault
        if self.kill_after is not None and len(self.asked) >= self.kill_after:
            raise KeyboardInterrupt
        self.asked.append((runtime, index))
        self.seconds += 1.0
        if (runtime, index) in self.truncated:
            return httpx.Response(200, json=chat_reply("<think>Step 1: first I", finish_reason="length"))
        answer = 2 * index + (1 if (runtime, index) in self.wrong else 0)
        reply = chat_reply(f"Answer: {answer}")
        if (runtime, index) in self.reasoned:
            reply["choices"][0]["message"]["reasoning"] = f"{index} plus {index} is {2 * index}."
        return httpx.Response(200, json=reply)


def run(
    fake: FakeRuntimes,
    tmp_path: Path,
    now: datetime = FIRST_START,
    problems: list[Problem] = PROBLEMS,
    sleeps: list[float] | None = None,
    thinking_mode: ThinkingMode = ThinkingMode.ON,
    sampling: SamplingProfile = VENDOR,
    only: str | None = None,
    selection: str = "range-0-4",
    model_file: ModelFile = MODEL_FILE,
) -> Path:
    http = httpx.Client(transport=httpx.MockTransport(fake.handler))
    runtimes: list[Runtime] = [LocalAI(http, LOCALAI_URL), Ollama(http, OLLAMA_URL)]
    return run_benchmark(
        runtimes=[r for r in runtimes if only in (None, r.name)],
        model=MODEL,
        model_file=model_file,
        thinking_mode=thinking_mode,
        sampling=sampling,
        problems=problems,
        selection=selection,
        machine=MACHINE,
        results_dir=tmp_path / "results",
        raw_dir=tmp_path / "runs",
        now=now,
        backoff_seconds=BACKOFF,
        sleep=(sleeps if sleeps is not None else []).append,
        clock=lambda: fake.seconds,
        progress=fake.progress.append,
    )


def by_runtime(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["runtime"]: r for r in result["results"]}


def test_a_complete_run_commits_aggregates_and_a_checksum_of_the_raw_responses(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.wrong = {("ollama", 3)}

    path = run(fake, tmp_path)
    result = json.loads(path.read_text())

    assert path.parent == tmp_path / "results" / "rtx-3080-test"
    assert result["kind"] == "benchmark-run"
    assert result["benchmark"]["questions"] == 5
    localai, ollama = by_runtime(result)["localai"], by_runtime(result)["ollama"]
    assert (localai["correct"], localai["failed"], localai["accuracy"]) == (5, 0, 1.0)
    assert (ollama["correct"], ollama["failed"], ollama["accuracy"]) == (4, 0, 0.8)
    assert localai["performance"]["completion_tokens"] == 500

    raw_files = sorted((tmp_path / "runs").rglob("localai/*.json"), key=lambda f: int(f.stem))
    assert len(raw_files) == 5
    expected = hashlib.sha256(
        "".join(f"{f.stem} {hashlib.sha256(f.read_bytes()).hexdigest()}\n" for f in raw_files).encode()
    ).hexdigest()
    assert localai["raw_responses_sha256"] == expected
    assert "Answer: 8" not in path.read_text()  # no raw responses in results/


# Resuming


def test_a_killed_run_continues_where_it_stopped_without_redoing_finished_questions(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.kill_after = 7  # all of LocalAI, then two Ollama questions

    with pytest.raises(KeyboardInterrupt):
        run(fake, tmp_path)
    assert not (tmp_path / "results").exists()

    fake.kill_after = None
    path = run(fake, tmp_path, now=RESTART)

    assert fake.asked[7:] == [("ollama", 2), ("ollama", 3), ("ollama", 4)]
    result = json.loads(path.read_text())
    assert [r["correct"] for r in result["results"]] == [5, 5]
    # The run keeps the time it first started.
    assert result["started_at"] == FIRST_START.isoformat()
    assert path.name.startswith("20261118T220000Z")


def test_a_runtime_with_nothing_left_to_ask_is_not_loaded_again(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.kill_after = 6
    with pytest.raises(KeyboardInterrupt):
        run(fake, tmp_path)
    fake.kill_after = None
    http_paths: list[tuple[str, str]] = []
    original = fake.handler

    def recording(request: httpx.Request) -> httpx.Response:
        http_paths.append((request.url.host, request.url.path))
        return original(request)

    fake.handler = recording  # type: ignore[method-assign]
    run(fake, tmp_path, now=RESTART)

    assert not any(host == "localai" for host, _ in http_paths)


def test_a_line_cut_short_by_the_kill_is_ignored(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.kill_after = 3
    with pytest.raises(KeyboardInterrupt):
        run(fake, tmp_path)
    journal = next((tmp_path / "runs").rglob("journal.jsonl"))
    with journal.open("a", encoding="utf-8") as file:
        file.write('{"type": "question", "runtime": "loc')

    fake.kill_after = None
    result = json.loads(run(fake, tmp_path, now=RESTART).read_text())

    assert fake.asked[3:5] == [("localai", 3), ("localai", 4)]
    assert [r["correct"] for r in result["results"]] == [5, 5]


def test_a_different_set_of_questions_is_a_separate_run(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    run(fake, tmp_path, problems=PROBLEMS[:2])

    run(fake, tmp_path, now=RESTART, problems=PROBLEMS)

    assert len(fake.asked) == 2 * 2 + 2 * 5


# Retrying


def unavailable(status: int = 503) -> httpx.Response:
    return httpx.Response(status, json={"error": "loading model"})


def test_transient_errors_are_retried_with_growing_waits(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    timeout = httpx.ReadTimeout("timed out")
    fake.faults[("localai", 1)] = [unavailable(503), timeout, unavailable(429)]
    sleeps: list[float] = []

    result = json.loads(run(fake, tmp_path, sleeps=sleeps).read_text())

    assert sleeps == [10.0, 30.0, 90.0]
    assert by_runtime(result)["localai"]["correct"] == 5
    assert by_runtime(result)["localai"]["failed"] == 0


def test_a_question_that_keeps_failing_is_recorded_and_the_run_goes_on(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.faults[("ollama", 2)] = [unavailable(500)] * 4
    sleeps: list[float] = []

    result = json.loads(run(fake, tmp_path, sleeps=sleeps).read_text())

    ollama = by_runtime(result)["ollama"]
    assert (ollama["answered"], ollama["correct"], ollama["failed"], ollama["accuracy"]) == (4, 4, 1, 0.8)
    assert [(f["index"], f["attempts"]) for f in ollama["failures"]] == [(2, 4)]
    assert "500" in ollama["failures"][0]["error"]
    assert ("ollama", 3) in fake.asked
    assert sleeps == BACKOFF


def test_a_persistent_timeout_is_recorded_as_a_failed_question(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.faults[("localai", 0)] = [httpx.ReadTimeout("timed out")] * 4

    result = json.loads(run(fake, tmp_path).read_text())

    assert [f["index"] for f in by_runtime(result)["localai"]["failures"]] == [0]
    assert "timed out" in by_runtime(result)["localai"]["failures"][0]["error"]


def test_a_client_error_is_not_retried(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.faults[("localai", 4)] = [httpx.Response(400, json={"error": "prompt exceeds the context window"})]
    sleeps: list[float] = []

    result = json.loads(run(fake, tmp_path, sleeps=sleeps).read_text())

    assert sleeps == []
    assert [(f["index"], f["attempts"]) for f in by_runtime(result)["localai"]["failures"]] == [(4, 1)]


def test_failed_questions_are_asked_again_when_the_run_is_resumed(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.faults[("ollama", 2)] = [unavailable(500)] * 4
    run(fake, tmp_path)
    asked_before = len(fake.asked)

    result = json.loads(run(fake, tmp_path, now=RESTART).read_text())

    assert fake.asked[asked_before:] == [("ollama", 2)]
    assert by_runtime(result)["ollama"]["failed"] == 0


def test_a_runtime_that_stays_unreachable_stops_the_run_and_keeps_the_progress(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.down.add("ollama")
    sleeps: list[float] = []

    with pytest.raises(RunInterrupted, match="ollama.*same command again"):
        run(fake, tmp_path, sleeps=sleeps)

    assert sleeps == BACKOFF
    assert not (tmp_path / "results").exists()
    fake.down.clear()
    run(fake, tmp_path, now=RESTART)
    assert [runtime for runtime, _ in fake.asked] == ["localai"] * 5 + ["ollama"] * 5


def test_a_failure_records_how_many_attempts_were_actually_made(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.faults[("localai", 1)] = [unavailable(503), httpx.Response(400, json={"error": "bad request"})]

    result = json.loads(run(fake, tmp_path).read_text())

    assert [(f["index"], f["attempts"]) for f in by_runtime(result)["localai"]["failures"]] == [(1, 2)]


def test_a_different_model_file_under_the_same_name_is_a_separate_run(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    run(fake, tmp_path)

    run(fake, tmp_path, now=RESTART, model_file=ModelFile(name=MODEL_FILE.name, sha256="requantized"))

    assert len(fake.asked) == 2 * 2 * 5


def test_a_timeout_is_retried_only_once_because_it_usually_repeats(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.faults[("localai", 0)] = [httpx.ReadTimeout("timed out")] * 4
    sleeps: list[float] = []

    result = json.loads(run(fake, tmp_path, sleeps=sleeps).read_text())

    assert sleeps == [10.0]
    assert [(f["index"], f["attempts"]) for f in by_runtime(result)["localai"]["failures"]] == [(0, 2)]


# Truncated responses


def test_responses_cut_off_at_the_token_limit_are_counted_as_truncated(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.truncated = {("ollama", 1), ("ollama", 4)}

    result = json.loads(run(fake, tmp_path).read_text())

    ollama = by_runtime(result)["ollama"]
    assert (ollama["answered"], ollama["correct"], ollama["truncated"], ollama["failed"]) == (5, 3, 2, 0)
    assert by_runtime(result)["localai"]["truncated"] == 0


def test_a_truncated_answer_counts_as_wrong_even_if_it_contains_the_right_number(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    cut_off = chat_reply("2 + 2 = 4, so the answer is 4. Wait, let me check: 4", finish_reason="length")
    fake.faults[("localai", 2)] = [httpx.Response(200, json=cut_off)]

    result = json.loads(run(fake, tmp_path).read_text())

    localai = by_runtime(result)["localai"]
    assert (localai["correct"], localai["truncated"], localai["accuracy"]) == (4, 1, 0.8)
    assert any("#2" in line and "wrong" in line and "truncated" in line for line in fake.progress)


# What a person watching a long run sees


def test_the_cold_load_time_is_kept_when_a_resumed_run_finds_the_model_still_loaded(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.kill_after = 2
    with pytest.raises(KeyboardInterrupt):
        run(fake, tmp_path)
    fake.kill_after = None
    fake.load_seconds = 0.3  # the killed run left the model in memory

    result = json.loads(run(fake, tmp_path, now=RESTART).read_text())

    assert by_runtime(result)["localai"]["performance"]["load_seconds"] == 9.0


def test_retries_are_counted_in_the_results(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.faults[("ollama", 1)] = [unavailable(500), unavailable(503)]
    fake.faults[("ollama", 3)] = [unavailable(500)]
    fake.faults[("localai", 0)] = [httpx.Response(400, json={"error": "bad request"})]

    result = json.loads(run(fake, tmp_path).read_text())

    assert by_runtime(result)["ollama"]["retries"] == 3
    assert by_runtime(result)["localai"]["retries"] == 0


def test_progress_reports_every_question_retry_and_resume(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.kill_after = 2
    with pytest.raises(KeyboardInterrupt):
        run(fake, tmp_path)
    fake.kill_after = None
    fake.faults[("localai", 3)] = [unavailable(503)]
    fake.wrong = {("localai", 4)}
    fake.progress.clear()

    run(fake, tmp_path, now=RESTART)

    localai = [line for line in fake.progress if line.startswith("localai")]
    assert "2 of 5 already answered" in localai[0]
    assert any("#2" in line and "3/5" in line and "correct" in line for line in localai)
    assert any("#3" in line and "Retrying in 10 s" in line and "503" in line for line in localai)
    assert any("#4" in line and "5/5" in line and "wrong" in line for line in localai)


# Thinking mode


def test_thinking_off_asks_every_runtime_to_answer_without_reasoning(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    run(fake, tmp_path, thinking_mode=ThinkingMode.OFF)

    assert {runtime for runtime, _ in fake.chats} == {"localai", "ollama"}
    assert all(body["reasoning_effort"] == "none" for _, body in fake.chats)


def test_thinking_on_is_asked_for_explicitly_not_left_to_the_model_default(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    run(fake, tmp_path, thinking_mode=ThinkingMode.ON)

    assert {runtime for runtime, _ in fake.chats} == {"localai", "ollama"}
    assert all(body["reasoning_effort"] == "high" for _, body in fake.chats)


def test_the_two_thinking_modes_are_separate_runs_with_separate_result_files(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    on = run(fake, tmp_path, thinking_mode=ThinkingMode.ON)
    off = run(fake, tmp_path, thinking_mode=ThinkingMode.OFF)

    assert len(fake.asked) == 2 * 2 * 5  # the second run did not resume the first
    assert on != off
    assert on.name == "20261118T220000Z-gsm8k-qwen3.5-9b-thinking-on-vendor.json"
    assert off.name == "20261118T220000Z-gsm8k-qwen3.5-9b-thinking-off-vendor.json"
    assert json.loads(on.read_text())["thinking_mode"] == "on"
    assert json.loads(off.read_text())["thinking_mode"] == "off"
    assert json.loads(on.read_text())["run_id"] != json.loads(off.read_text())["run_id"]


def test_answers_that_came_with_reasoning_are_counted(tmp_path: Path) -> None:
    fake = FakeRuntimes()
    fake.reasoned = {("ollama", 0), ("ollama", 3)}

    result = json.loads(run(fake, tmp_path, thinking_mode=ThinkingMode.OFF).read_text())

    # With Thinking mode off this shows whether the Runtime really switched it off.
    assert by_runtime(result)["ollama"]["reasoned"] == 2
    assert by_runtime(result)["localai"]["reasoned"] == 0


# Run configuration in the result file


def test_the_result_records_the_context_window_and_which_problems_were_asked(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    result = json.loads(run(fake, tmp_path, selection="subset-5").read_text())

    assert result["context_window"] == 16384
    assert result["benchmark"] == {"name": "gsm8k", "questions": 5, "selection": "subset-5"}


def test_accuracy_can_be_measured_on_a_single_runtime(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    result = json.loads(run(fake, tmp_path, only="localai").read_text())

    assert {runtime for runtime, _ in fake.asked} == {"localai"}
    assert [(r["runtime"], r["accuracy"]) for r in result["results"]] == [("localai", 1.0)]


def test_an_empty_think_block_left_by_the_chat_template_is_not_reasoning() -> None:
    def reply(content: str, **message: str) -> dict[str, Any]:
        response = chat_reply(content)
        response["choices"][0]["message"].update(message)
        return response

    assert not has_reasoning(reply("<think>\n\n</think>\n\nAnswer: 4"))
    assert not has_reasoning(reply("Answer: 4", reasoning=""))
    assert has_reasoning(reply("<think>2 plus 2 is 4.</think>\nAnswer: 4"))
    assert has_reasoning(reply("Answer: 4", reasoning_content="2 plus 2 is 4."))


# Sampling profile


def test_every_runtime_gets_the_same_pinned_sampling_with_every_question(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    run(fake, tmp_path, sampling=VENDOR)

    def sampling(body: dict[str, Any]) -> dict[str, Any]:
        return {k: body[k] for k in VENDOR.request_fields()}

    assert {runtime for runtime, _ in fake.chats} == {"localai", "ollama"}
    assert all(sampling(body) == VENDOR.request_fields() for _, body in fake.chats)
    assert VENDOR.request_fields() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "seed": VENDOR.seed,
    }


def test_the_greedy_profile_decodes_at_temperature_zero_without_penalties(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    run(fake, tmp_path, sampling=GREEDY)

    assert all(
        (body["temperature"], body["presence_penalty"], body["repeat_penalty"]) == (0.0, 0.0, 1.0)
        for _, body in fake.chats
    )


def test_two_sampling_profiles_are_separate_runs_and_the_profile_is_recorded(tmp_path: Path) -> None:
    fake = FakeRuntimes()

    vendor = run(fake, tmp_path, sampling=VENDOR)
    greedy = run(fake, tmp_path, sampling=GREEDY)

    assert len(fake.asked) == 2 * 2 * 5
    assert greedy.name == "20261118T220000Z-gsm8k-qwen3.5-9b-thinking-on-greedy.json"
    recorded = json.loads(vendor.read_text())["sampling"]
    assert recorded == {"profile": "vendor", **VENDOR.request_fields()}
