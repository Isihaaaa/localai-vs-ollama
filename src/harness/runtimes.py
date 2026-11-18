"""Adapters for the Runtimes, reached through their OpenAI-compatible API.

Chat goes through ``/v1/chat/completions`` on both. Version and model
unloading use each Runtime's own endpoints.
"""

from abc import ABC, abstractmethod
from typing import Any

import httpx

# Qwen3.5 thinks by default, so leave room for the reasoning within the 16k Context window.
MAX_TOKENS = 12_000


class RuntimeUnavailable(RuntimeError):
    """The Runtime did not answer. Check `docker compose ps` and the container logs."""


class RuntimeTimeout(RuntimeUnavailable):
    """The Runtime accepted the request but did not answer in time."""


class RuntimeRequestFailed(RuntimeError):
    """The Runtime answered with an error status."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class Runtime(ABC):
    name: str
    _version_path: str

    def __init__(self, http: httpx.Client, base_url: str) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    def version(self) -> str | None:
        response = self._request("GET", self._version_path, check=False)
        if response.status_code != 200:
            return None
        version = response.json().get("version")
        return str(version) if version else None

    def has_model(self, model: str) -> bool:
        """Whether the Runtime is up and lists ``model`` (raises if unreachable)."""
        response = self._request("GET", "/v1/models", check=False)
        if response.status_code != 200:
            return False
        return any(entry.get("id", "").split(":")[0] == model for entry in response.json()["data"])

    def load(self, model: str) -> None:
        """Get the model into memory with a one-token request.

        Called before the timed request, so its duration measures generation
        only, not loading the Model file into VRAM.
        """
        self._request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0,
                "max_tokens": 1,
            },
        )

    def chat(self, model: str, prompt: str) -> dict[str, Any]:
        """Send one user message and return the raw JSON response."""
        response = self._request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
            },
        )
        body: dict[str, Any] = response.json()
        return body

    @abstractmethod
    def unload(self, model: str) -> None:
        """Free the model's memory so the next Runtime has the VRAM to itself."""

    def _request(self, method: str, path: str, *, check: bool = True, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, f"{self._base_url}{path}", **kwargs)
        except httpx.TimeoutException as error:
            raise RuntimeTimeout(
                f"{self.name} {method} {path} timed out ({type(error).__name__})."
            ) from error
        except httpx.TransportError as error:
            raise RuntimeUnavailable(
                f"{self.name} at {self._base_url} is not reachable ({error}). "
                "Is it running? Check `docker compose ps`."
            ) from error
        if check and response.is_error:
            raise RuntimeRequestFailed(
                f"{self.name} {method} {path} failed with HTTP {response.status_code}: "
                f"{response.text[:500]}. Check `docker compose logs {self.name}`.",
                status_code=response.status_code,
            )
        return response


class LocalAI(Runtime):
    name = "localai"
    _version_path = "/version"

    def unload(self, model: str) -> None:
        self._request("POST", "/backend/shutdown", json={"model": model})


class Ollama(Runtime):
    name = "ollama"
    _version_path = "/api/version"

    def unload(self, model: str) -> None:
        self._request("POST", "/api/generate", json={"model": model, "keep_alive": 0})


def message_content(response: dict[str, Any]) -> str:
    """The visible answer text of a chat completion response."""
    content = response["choices"][0]["message"].get("content")
    return str(content or "")
