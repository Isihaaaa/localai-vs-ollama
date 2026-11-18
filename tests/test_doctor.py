import httpx
import pytest

from fakes import FakeHost, apple_silicon_mac
from harness.doctor import Check, format_checklist, run_checks
from harness.machine import describe_machine
from harness.runtimes import LocalAI, Ollama, Runtime

MODEL = "qwen3.5-9b"


class FakeRuntimes:
    """Both Runtimes behind one mock transport; each can be down or missing the model."""

    def __init__(self) -> None:
        self.down: set[str] = set()
        self.without_model: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        runtime = request.url.host
        if runtime in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/v1/models":
            models = [] if runtime in self.without_model else [{"id": f"{MODEL}:latest"}]
            return httpx.Response(200, json={"data": models})
        return httpx.Response(200, json={"version": "1.0"})

    def runtimes(self) -> list[Runtime]:
        http = httpx.Client(transport=httpx.MockTransport(self.handler))
        return [LocalAI(http, "http://localai:8080"), Ollama(http, "http://ollama:11434")]


def check(host: FakeHost, fake: FakeRuntimes | None = None) -> dict[str, Check]:
    fake = fake or FakeRuntimes()
    checks = run_checks(host, describe_machine(host), fake.runtimes(), MODEL, disk_path=".")
    return {c.name: c for c in checks}


def failed(checks: dict[str, Check]) -> list[str]:
    return [name for name, c in checks.items() if not c.passed]


def test_a_ready_windows_machine_passes_every_check() -> None:
    checks = check(FakeHost())

    assert list(checks) == [
        "Platform",
        "Docker",
        "GPU access",
        "Disk space",
        "localai reachable",
        f"localai lists {MODEL}",
        "ollama reachable",
        f"ollama lists {MODEL}",
    ]
    assert failed(checks) == []
    assert "Docker" in checks["Platform"].detail
    assert "tier M" in checks["Platform"].detail


def test_docker_not_running_fails() -> None:
    host = FakeHost()
    del host.outputs["docker info"]

    assert "Docker" in failed(check(host))


def test_containers_that_cannot_see_the_gpu_fail_the_gpu_check() -> None:
    host = FakeHost()
    del host.outputs["docker run"]

    assert failed(check(host)) == ["GPU access"]


def test_gpu_access_is_checked_from_inside_a_container() -> None:
    host = FakeHost()

    check(host)

    assert ["docker", "run", "--rm", "--gpus", "all", "ubuntu", "nvidia-smi", "-L"] in host.commands


def test_too_little_disk_space_for_the_first_download_fails() -> None:
    fake = FakeRuntimes()
    fake.without_model.add("ollama")

    checks = check(FakeHost(free_bytes=10 * 1024**3), fake)

    assert "Disk space" in failed(checks)
    assert "15 GB" in checks["Disk space"].detail


def test_once_the_model_is_downloaded_less_disk_space_is_enough() -> None:
    assert failed(check(FakeHost(free_bytes=10 * 1024**3))) == []


def test_an_unreachable_runtime_fails_both_its_checks() -> None:
    fake = FakeRuntimes()
    fake.down.add("localai")

    checks = check(FakeHost(), fake)

    assert failed(checks) == ["localai reachable", f"localai lists {MODEL}"]


def test_a_runtime_without_the_model_fails_the_model_check() -> None:
    fake = FakeRuntimes()
    fake.without_model.add("ollama")

    assert failed(check(FakeHost(free_bytes=100 * 1024**3), fake)) == [f"ollama lists {MODEL}"]


def test_macos_dry_run_takes_the_native_path_without_docker() -> None:
    host = apple_silicon_mac(memory_gb=32)

    checks = check(host)

    assert "Docker" not in checks
    assert failed(checks) == []
    assert "native" in checks["Platform"].detail
    assert "tier L" in checks["Platform"].detail
    assert "Metal" in checks["GPU access"].detail
    assert not any(command[0] == "docker" for command in host.commands)


def test_an_unsupported_platform_fails() -> None:
    checks = check(FakeHost(outputs={}))

    assert failed(checks)[0] == "Platform"


def test_checklist_marks_each_check_pass_or_fail() -> None:
    text = format_checklist(
        [Check("Docker", True, "running"), Check("GPU access", False, "no GPU in containers")]
    )

    assert text.splitlines() == [
        "[PASS] Docker: running",
        "[FAIL] GPU access: no GPU in containers",
    ]


@pytest.mark.parametrize("passing", [True, False])
def test_cli_exit_code_reflects_the_checks(monkeypatch: pytest.MonkeyPatch, passing: bool) -> None:
    from harness import cli

    monkeypatch.setattr(cli, "run_checks", lambda *args, **kwargs: [Check("Docker", passing, "x")])

    assert cli.main(["doctor"]) == (0 if passing else 1)


def test_on_macos_a_missing_model_is_not_blamed_on_docker_compose() -> None:
    fake = FakeRuntimes()
    fake.without_model.add("ollama")

    checks = check(apple_silicon_mac(), fake)

    assert "docker" not in checks[f"ollama lists {MODEL}"].detail
