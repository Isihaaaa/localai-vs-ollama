"""Sampling profiles: the decoding settings pinned identically on every Runtime.

Each Runtime has its own defaults for everything a request leaves out, so a
comparison that sets only ``temperature`` is not controlled. A Sampling
profile sets every setting explicitly, with every question, and is recorded
in the Run configuration (decided in #4).

``top_k``, ``min_p`` and ``repeat_penalty`` are not part of the OpenAI API.
LocalAI takes them per request; Ollama's OpenAI-compatible endpoint ignores
them, so both Runtimes' model configs pin them to ``RUNTIME_PINNED`` as well.
That is why every profile must use those same values.
"""

from dataclasses import asdict, dataclass
from typing import Any

# Fixed so that sampled runs are as repeatable as a Runtime allows.
SEED = 20261119

RUNTIME_PINNED: dict[str, float] = {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0}
"""Must match docker/localai/*.yaml and docker/ollama/Modelfile."""


@dataclass(frozen=True)
class SamplingProfile:
    name: str
    temperature: float
    top_p: float
    presence_penalty: float
    top_k: int = 20
    """No effect at temperature 0."""
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    seed: int = SEED

    def request_fields(self) -> dict[str, Any]:
        """The fields sent with every chat request, and recorded in the result file."""
        fields = asdict(self)
        del fields["name"]
        return fields


VENDOR = SamplingProfile("vendor", temperature=1.0, top_p=0.95, presence_penalty=1.5)
"""Qwen's recommendation for Qwen3.5 on reasoning tasks, the same in both Thinking modes.

The default: the vendor warns that greedy decoding makes Thinking mode repeat itself endlessly.
"""

GREEDY = SamplingProfile("greedy", temperature=0.0, top_p=1.0, presence_penalty=0.0)
"""Deterministic decoding, kept for the pilot that compares it with ``vendor``."""

PROFILES = {profile.name: profile for profile in (VENDOR, GREEDY)}
