"""Prerequisite checks before a run: ``uv run harness doctor``."""

from collections.abc import Sequence
from dataclasses import dataclass

from harness.machine import DOCKER, NATIVE, Host, MachineDescription
from harness.runtimes import Runtime, RuntimeUnavailable

GB = 1024**3
# First setup: ~6 GB Core model file plus the Runtime images (see README).
FIRST_SETUP_DISK_GB = 15
# Once the model is in place, a run only writes result files and raw responses.
RUN_DISK_GB = 2

# The README's GPU test: can a container see the GPU through Docker Desktop / WSL2?
DOCKER_GPU_TEST = ["docker", "run", "--rm", "--gpus", "all", "ubuntu", "nvidia-smi", "-L"]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def run_checks(
    host: Host,
    machine: MachineDescription,
    runtimes: Sequence[Runtime],
    model: str,
    disk_path: str,
) -> list[Check]:
    """Check everything a run needs, in the order a person would fix it.

    The Docker checks apply only where the Runtimes run in Docker; on macOS
    they run natively on Metal and Docker is not needed for them.
    """
    checks = [_platform(machine)]
    if machine.runtime_path == DOCKER:
        checks.append(_docker(host))
    checks.append(_gpu_access(host, machine))

    runtime_checks: list[Check] = []
    model_everywhere = True
    for runtime in runtimes:
        reachable, has_model = _runtime(runtime, model, machine.runtime_path)
        runtime_checks += [reachable, has_model]
        model_everywhere = model_everywhere and has_model.passed

    checks.append(_disk_space(host, disk_path, RUN_DISK_GB if model_everywhere else FIRST_SETUP_DISK_GB))
    return checks + runtime_checks


def format_checklist(checks: Sequence[Check]) -> str:
    return "\n".join(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in checks)


def _platform(machine: MachineDescription) -> Check:
    if machine.runtime_path is None:
        return Check(
            "Platform",
            False,
            f"{machine.os} with {', '.join(g.name for g in machine.gpus) or 'no NVIDIA GPU'} is not supported. "
            "Supported: Windows + NVIDIA GPU (Docker), macOS on Apple Silicon (native).",
        )
    where = "Runtimes in Docker" if machine.runtime_path == DOCKER else "Runtimes native on Metal"
    return Check(
        "Platform",
        True,
        f"{machine.platform}, {where}; {machine.model_memory_gb} GB model memory, "
        f"Hardware tier {machine.hardware_tier}",
    )


def _docker(host: Host) -> Check:
    version = host.run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if version is None:
        return Check("Docker", False, "Docker is not running or not installed. Start Docker Desktop.")
    return Check("Docker", True, f"running, server {version.strip()}")


def _gpu_access(host: Host, machine: MachineDescription) -> Check:
    if machine.runtime_path == NATIVE:
        return Check("GPU access", True, f"Apple Silicon GPU via Metal ({machine.cpu})")
    if machine.runtime_path is None:
        return Check("GPU access", False, "no NVIDIA GPU (nvidia-smi) or Apple Silicon found")
    output = host.run(DOCKER_GPU_TEST)
    if not output or "GPU" not in output:
        return Check(
            "GPU access",
            False,
            f"{machine.gpus[0].name} is not visible inside containers. Run `{' '.join(DOCKER_GPU_TEST)}` "
            "and check Docker Desktop's WSL2 backend and the Windows NVIDIA driver.",
        )
    return Check("GPU access", True, f"containers see {output.strip().splitlines()[0]}")


def _disk_space(host: Host, path: str, needed_gb: int) -> Check:
    free_gb = host.free_disk_bytes(path) / GB
    detail = f"{free_gb:.0f} GB free, {needed_gb} GB needed"
    if free_gb < needed_gb:
        return Check("Disk space", False, f"{detail}. Free up space on the drive holding {path}.")
    return Check("Disk space", True, detail)


def _runtime(runtime: Runtime, model: str, runtime_path: str | None) -> tuple[Check, Check]:
    model_check = f"{runtime.name} lists {model}"
    try:
        version = runtime.version()
        has_model = runtime.has_model(model)
    except RuntimeUnavailable as error:
        return (
            Check(f"{runtime.name} reachable", False, str(error)),
            Check(model_check, False, f"{runtime.name} is not reachable"),
        )
    reachable = Check(f"{runtime.name} reachable", True, f"version {version or 'unknown'}")
    if not has_model:
        fix = (
            f"Check `docker compose logs model-fetch ollama-model {runtime.name}`."
            if runtime_path == DOCKER
            else f"Download the Model file and register it in {runtime.name} as '{model}'."
        )
        return reachable, Check(model_check, False, f"the Model file is not downloaded or not registered. {fix}")
    return reachable, Check(model_check, True, "Model file downloaded and registered")
