"""Automatic machine description for result files."""

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import psutil

from harness import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]


class Host(Protocol):
    """Where machine facts come from. ``LocalHost`` asks the real machine."""

    def os_name(self) -> str: ...
    def cpu_name(self) -> str: ...
    def total_memory_bytes(self) -> int: ...
    def system_name(self) -> str:
        """``sys.platform``: "win32", "darwin" or "linux"."""
        ...
    def architecture(self) -> str:
        """``platform.machine()``, e.g. "AMD64", "x86_64" or "arm64"."""
        ...
    def free_disk_bytes(self, path: str) -> int: ...
    def run(self, command: Sequence[str]) -> str | None:
        """Run a command and return its stdout, or None if it is missing or fails."""
        ...


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_mb: int
    driver: str


# Platforms the harness supports, and where the Runtimes run on each.
WINDOWS_NVIDIA = "windows-nvidia"
MACOS_APPLE_SILICON = "macos-apple-silicon"
UNSUPPORTED = "unsupported"

DOCKER = "docker"
NATIVE = "native"  # Docker containers cannot reach the GPU on macOS, so the Runtimes use Metal directly.

_RUNTIME_PATHS = {WINDOWS_NVIDIA: DOCKER, MACOS_APPLE_SILICON: NATIVE}

# macOS lets the GPU use about this share of unified memory by default; the rest stays with the system.
APPLE_GPU_MEMORY_SHARE = 0.75

# Lower bound of model memory (GB) for each Hardware tier, largest first.
_TIER_FLOORS_GB = (("XL", 48.0), ("L", 24.0), ("M", 12.0))


@dataclass(frozen=True)
class MachineDescription:
    machine_id: str
    os: str
    cpu: str
    memory_gb: int
    gpus: list[Gpu]
    script_version: str
    script_commit: str | None
    platform: str = UNSUPPORTED
    runtime_path: str | None = None
    model_memory_gb: float | None = None
    """VRAM of the largest GPU, or the GPU's share of unified memory on Apple Silicon."""
    hardware_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_machine(host: Host, machine_id: str | None = None) -> MachineDescription:
    cpu = host.cpu_name()
    memory_bytes = host.total_memory_bytes()
    memory_gb = round(memory_bytes / 1024**3)
    apple_silicon = host.system_name() == "darwin" and host.architecture().lower() == "arm64"
    gpus = [] if apple_silicon else _nvidia_gpus(host)
    if apple_silicon:
        platform_name = MACOS_APPLE_SILICON
    elif gpus and host.system_name() == "win32":
        platform_name = WINDOWS_NVIDIA
    else:
        platform_name = UNSUPPORTED
    model_memory_gb = _model_memory_gb(platform_name, gpus, memory_bytes)
    commit = host.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    return MachineDescription(
        machine_id=machine_id or _derive_machine_id(cpu, memory_gb, gpus),
        os=host.os_name(),
        cpu=cpu,
        memory_gb=memory_gb,
        gpus=gpus,
        script_version=__version__,
        script_commit=commit.strip() if commit else None,
        platform=platform_name,
        runtime_path=_RUNTIME_PATHS.get(platform_name),
        model_memory_gb=model_memory_gb,
        hardware_tier=None if model_memory_gb is None else hardware_tier(model_memory_gb),
    )


def hardware_tier(model_memory_gb: float) -> str:
    """S below 12 GB of model memory, M up to 24, L up to 48, XL from 48."""
    for tier, floor in _TIER_FLOORS_GB:
        if model_memory_gb >= floor:
            return tier
    return "S"


def _model_memory_gb(platform_name: str, gpus: list[Gpu], total_memory_bytes: int) -> float | None:
    if platform_name == MACOS_APPLE_SILICON:
        return round(total_memory_bytes * APPLE_GPU_MEMORY_SHARE / 1024**3, 1)
    if platform_name == WINDOWS_NVIDIA:
        return round(max(g.vram_mb for g in gpus) / 1024, 1)
    return None


def _nvidia_gpus(host: Host) -> list[Gpu]:
    output = host.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    )
    gpus = []
    for line in (output or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit():
            gpus.append(Gpu(name=parts[0], vram_mb=int(parts[1]), driver=parts[2]))
    return gpus


def _derive_machine_id(cpu: str, memory_gb: int, gpus: list[Gpu]) -> str:
    """Stable for the same hardware, independent of hostname and OS version."""
    hardware = {"cpu": cpu, "memory_gb": memory_gb, "gpus": [(g.name, g.vram_mb) for g in gpus]}
    digest = hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest()[:8]
    label = gpus[0].name if gpus else cpu
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return f"{slug}-{digest}"


class LocalHost:
    """The machine the harness runs on (Windows, Linux/WSL2 or macOS)."""

    def os_name(self) -> str:
        if sys.platform == "darwin":
            return f"macOS {platform.mac_ver()[0]}"
        if sys.platform == "linux":
            try:
                pretty = platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
            except OSError:
                pretty = "Linux"
            return f"{pretty} (kernel {platform.release()})"
        return f"{platform.system()} {platform.release()} ({platform.version()})"

    def cpu_name(self) -> str:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            brand = self.run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if brand:
                return brand.strip()
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return platform.processor() or "unknown"

    def total_memory_bytes(self) -> int:
        return int(psutil.virtual_memory().total)

    def system_name(self) -> str:
        return sys.platform

    def architecture(self) -> str:
        return platform.machine()

    def free_disk_bytes(self, path: str) -> int:
        return shutil.disk_usage(path).free

    def run(self, command: Sequence[str]) -> str | None:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None
