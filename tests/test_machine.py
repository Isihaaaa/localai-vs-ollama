from collections.abc import Sequence
from dataclasses import dataclass, field

from harness import __version__
from harness.machine import describe_machine

NVIDIA_SMI_OUTPUT = "NVIDIA GeForce RTX 3080 Laptop GPU, 16384, 581.57\n"


@dataclass
class FakeHost:
    """A host whose facts and command outputs are fixed by the test."""

    os: str = "Windows 11 (10.0.26100)"
    cpu: str = "Intel(R) Core(TM) i9-11980HK"
    memory_bytes: int = 64 * 1024**3
    outputs: dict[str, str] = field(
        default_factory=lambda: {"nvidia-smi": NVIDIA_SMI_OUTPUT, "git": "9a12fc3\n"}
    )

    def os_name(self) -> str:
        return self.os

    def cpu_name(self) -> str:
        return self.cpu

    def total_memory_bytes(self) -> int:
        return self.memory_bytes

    def run(self, command: Sequence[str]) -> str | None:
        return self.outputs.get(command[0])


def test_describes_cpu_gpu_memory_os_and_script_version() -> None:
    machine = describe_machine(FakeHost())

    assert machine.cpu == "Intel(R) Core(TM) i9-11980HK"
    assert machine.os == "Windows 11 (10.0.26100)"
    assert machine.memory_gb == 64
    assert [(g.name, g.vram_mb, g.driver) for g in machine.gpus] == [
        ("NVIDIA GeForce RTX 3080 Laptop GPU", 16384, "581.57")
    ]
    assert machine.script_version == __version__
    assert machine.script_commit == "9a12fc3"


def test_a_machine_without_nvidia_smi_has_no_gpus() -> None:
    machine = describe_machine(FakeHost(outputs={}))

    assert machine.gpus == []
    assert machine.script_commit is None


def test_machine_id_is_stable_for_the_same_hardware() -> None:
    assert describe_machine(FakeHost()).machine_id == describe_machine(FakeHost()).machine_id


def test_machine_id_ignores_os_updates() -> None:
    before = describe_machine(FakeHost(os="Windows 11 (10.0.26100)"))
    after = describe_machine(FakeHost(os="Windows 11 (10.0.26200)"))
    assert before.machine_id == after.machine_id


def test_machine_id_differs_for_different_hardware() -> None:
    laptop = describe_machine(FakeHost())
    bigger = describe_machine(FakeHost(memory_bytes=128 * 1024**3))
    assert laptop.machine_id != bigger.machine_id


def test_machine_id_is_readable_and_filesystem_safe() -> None:
    machine_id = describe_machine(FakeHost()).machine_id
    assert machine_id.startswith("nvidia-geforce-rtx-3080-laptop-gpu-")
    assert all(c.isalnum() or c == "-" for c in machine_id)


def test_cpu_only_machine_id_is_named_after_the_cpu() -> None:
    machine_id = describe_machine(FakeHost(outputs={})).machine_id
    assert machine_id.startswith("intel-r-core-tm-i9-11980hk-")


def test_machine_id_can_be_overridden() -> None:
    assert describe_machine(FakeHost(), machine_id="steve-laptop").machine_id == "steve-laptop"
