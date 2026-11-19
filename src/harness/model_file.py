"""The Model file a run loads: one Model at one Quantization, as a GGUF file."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str | None
    """Checksum recorded when the file was downloaded; ``None`` if it was not recorded."""

    def to_dict(self) -> dict[str, str | None]:
        return {"name": self.name, "sha256": self.sha256}
