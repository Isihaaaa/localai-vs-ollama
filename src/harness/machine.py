"""Automatic machine description for result files."""

import hashlib
import json
import platform
import re
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
    def run(self, command: Sequence[str]) -> str | None:
        """Run a command and return its stdout, or None if it is missing or fails."""
        ...


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_mb: int
    driver: str


@dataclass(frozen=True)
class MachineDescription:
    machine_id: str
    os: str
    cpu: str
    memory_gb: int
    gpus: list[Gpu]
    script_version: str
    script_commit: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_machine(host: Host, machine_id: str | None = None) -> MachineDescription:
    cpu = host.cpu_name()
    memory_gb = round(host.total_memory_bytes() / 1024**3)
    gpus = _nvidia_gpus(host)
    commit = host.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    return MachineDescription(
        machine_id=machine_id or _derive_machine_id(cpu, memory_gb, gpus),
        os=host.os_name(),
        cpu=cpu,
        memory_gb=memory_gb,
        gpus=gpus,
        script_version=__version__,
        script_commit=commit.strip() if commit else None,
    )


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

    def run(self, command: Sequence[str]) -> str | None:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None
