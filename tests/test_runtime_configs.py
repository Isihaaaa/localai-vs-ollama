"""The Runtimes' model configs must match what the harness records in result files.

Ollama's OpenAI-compatible endpoint ignores ``top_k``, ``min_p`` and
``repeat_penalty`` in a request, so those come from its Modelfile. They are
pinned in both configs, and must equal what every Sampling profile records.
"""

import re

from harness.machine import REPO_ROOT
from harness.runtimes import CONTEXT_WINDOW
from harness.sampling import PROFILES, RUNTIME_PINNED

LOCALAI_CONFIG = REPO_ROOT / "docker" / "localai" / "qwen3.5-9b.yaml"
OLLAMA_MODELFILE = REPO_ROOT / "docker" / "ollama" / "Modelfile"


def localai_settings() -> dict[str, float]:
    """The numeric ``key: value`` lines of the LocalAI model YAML, at any indentation."""
    text = LOCALAI_CONFIG.read_text(encoding="utf-8")
    return {k: float(v) for k, v in re.findall(r"^\s*(\w+):\s*(-?[\d.]+)\s*$", text, re.MULTILINE)}


def ollama_parameters() -> dict[str, float]:
    text = OLLAMA_MODELFILE.read_text(encoding="utf-8")
    return {k: float(v) for k, v in re.findall(r"^PARAMETER\s+(\w+)\s+(-?[\d.]+)\s*$", text, re.MULTILINE)}


def test_both_runtimes_load_the_model_with_the_recorded_context_window() -> None:
    assert localai_settings()["context_size"] == CONTEXT_WINDOW
    assert ollama_parameters()["num_ctx"] == CONTEXT_WINDOW


def test_both_runtimes_pin_the_sampling_settings_a_request_cannot_set_on_ollama() -> None:
    localai, ollama = localai_settings(), ollama_parameters()

    for name, value in RUNTIME_PINNED.items():
        assert localai[name] == value, f"LocalAI {name}"
        assert ollama[name] == value, f"Ollama {name}"
    for profile in PROFILES.values():
        assert {k: profile.request_fields()[k] for k in RUNTIME_PINNED} == RUNTIME_PINNED
