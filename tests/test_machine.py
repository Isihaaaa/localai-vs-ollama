import pytest

from fakes import FakeHost, apple_silicon_mac
from harness import __version__
from harness.machine import describe_machine, hardware_tier


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


# Platform and Hardware tier


def test_windows_with_an_nvidia_gpu_runs_the_runtimes_in_docker() -> None:
    machine = describe_machine(FakeHost())

    assert machine.platform == "windows-nvidia"
    assert machine.runtime_path == "docker"


def test_apple_silicon_runs_the_runtimes_natively_on_metal() -> None:
    machine = describe_machine(apple_silicon_mac())

    assert machine.platform == "macos-apple-silicon"
    assert machine.runtime_path == "native"
    assert machine.gpus == []


def test_the_mac_does_not_look_for_nvidia_smi() -> None:
    host = apple_silicon_mac()

    describe_machine(host)

    assert not any(command[0] == "nvidia-smi" for command in host.commands)


@pytest.mark.parametrize(
    "host",
    [
        FakeHost(outputs={}),  # Windows without an NVIDIA GPU
        FakeHost(system="darwin", arch="x86_64", outputs={}),  # Intel Mac
    ],
    ids=["windows-without-nvidia", "intel-mac"],
)
def test_other_machines_are_unsupported_and_get_no_tier(host: FakeHost) -> None:
    machine = describe_machine(host)

    assert machine.platform == "unsupported"
    assert machine.runtime_path is None
    assert machine.model_memory_gb is None
    assert machine.hardware_tier is None


def test_model_memory_on_nvidia_is_the_vram_of_the_largest_gpu() -> None:
    two_gpus = "NVIDIA RTX A2000, 6144, 581.57\nNVIDIA GeForce RTX 4090, 24564, 581.57\n"
    machine = describe_machine(FakeHost(outputs={"nvidia-smi": two_gpus}))

    assert machine.model_memory_gb == 24.0


def test_the_tier_m_dev_machine_is_tier_m() -> None:
    machine = describe_machine(FakeHost())

    assert machine.model_memory_gb == 16.0
    assert machine.hardware_tier == "M"


def test_on_apple_silicon_three_quarters_of_unified_memory_is_available_to_models() -> None:
    machine = describe_machine(apple_silicon_mac(memory_gb=32))

    assert machine.model_memory_gb == 24.0
    assert machine.hardware_tier == "L"


@pytest.mark.parametrize(
    ("model_memory_gb", "tier"),
    [(8.0, "S"), (11.9, "S"), (12.0, "M"), (16.0, "M"), (24.0, "L"), (47.9, "L"), (48.0, "XL"), (96.0, "XL")],
)
def test_tier_boundaries(model_memory_gb: float, tier: str) -> None:
    assert hardware_tier(model_memory_gb) == tier


def test_the_tier_is_part_of_the_stored_description() -> None:
    stored = describe_machine(FakeHost()).to_dict()

    assert stored["platform"] == "windows-nvidia"
    assert stored["runtime_path"] == "docker"
    assert stored["model_memory_gb"] == 16.0
    assert stored["hardware_tier"] == "M"


def test_linux_is_unsupported_even_with_an_nvidia_gpu() -> None:
    machine = describe_machine(FakeHost(system="linux", arch="x86_64"))

    assert machine.platform == "unsupported"
    assert machine.hardware_tier is None
