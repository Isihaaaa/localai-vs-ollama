"""Test doubles shared by several test files."""

from collections.abc import Sequence
from dataclasses import dataclass, field

NVIDIA_SMI_OUTPUT = "NVIDIA GeForce RTX 3080 Laptop GPU, 16384, 581.57\n"
DOCKER_GPU_OUTPUT = "GPU 0: NVIDIA GeForce RTX 3080 Laptop GPU (UUID: GPU-1234)\n"


@dataclass
class FakeHost:
    """A host whose facts and command outputs are fixed by the test.

    ``outputs`` maps a command prefix ("docker info", or just "git") to its
    stdout; the longest matching prefix wins. A command with no match fails.
    """

    os: str = "Windows 11 (10.0.26100)"
    cpu: str = "Intel(R) Core(TM) i9-11980HK"
    memory_bytes: int = 64 * 1024**3
    system: str = "win32"
    arch: str = "AMD64"
    free_bytes: int = 500 * 1024**3
    outputs: dict[str, str] = field(
        default_factory=lambda: {
            "nvidia-smi": NVIDIA_SMI_OUTPUT,
            "git": "9a12fc3\n",
            "docker info": "29.9.3\n",
            "docker run": DOCKER_GPU_OUTPUT,
        }
    )
    commands: list[list[str]] = field(default_factory=list)

    def os_name(self) -> str:
        return self.os

    def cpu_name(self) -> str:
        return self.cpu

    def total_memory_bytes(self) -> int:
        return self.memory_bytes

    def system_name(self) -> str:
        return self.system

    def architecture(self) -> str:
        return self.arch

    def free_disk_bytes(self, path: str) -> int:
        return self.free_bytes

    def run(self, command: Sequence[str]) -> str | None:
        self.commands.append(list(command))
        line = " ".join(command)
        matches = [prefix for prefix in self.outputs if line == prefix or line.startswith(prefix + " ")]
        return self.outputs[max(matches, key=len)] if matches else None


def apple_silicon_mac(memory_gb: int = 32) -> FakeHost:
    return FakeHost(
        os="macOS 15.4",
        cpu="Apple M3 Pro",
        memory_bytes=memory_gb * 1024**3,
        system="darwin",
        arch="arm64",
        outputs={"git": "9a12fc3\n"},
    )
