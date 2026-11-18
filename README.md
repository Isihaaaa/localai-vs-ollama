# LocalAI vs Ollama

Running generative AI on-premise with **LocalAI** and **Ollama**: we compare the two Runtimes and measure how small open Models perform against the Reference model (Claude). The plan is in [`docs/plan.md`](docs/plan.md), terms in [`CONTEXT.md`](CONTEXT.md) and decisions in [`docs/adr/`](docs/adr/).

## Quick start: Windows + NVIDIA (Docker Desktop / WSL2)

This walking skeleton starts both Runtimes, sends one GSM8K problem to each, scores the answers and writes a result file to `results/<machine-id>/`.

### Prerequisites

- An NVIDIA GPU with a current **Windows** driver. Do not install a separate driver inside WSL2: the Windows driver serves WSL2 too.
- **Docker Desktop** with the WSL2 backend (Settings → General → "Use the WSL 2 based engine"). Check that containers can see the GPU:
  ```powershell
  docker run --rm --gpus all ubuntu nvidia-smi
  ```
- **uv** for the Python harness: `winget install astral-sh.uv`
- About 15 GB free disk space: ~6 GB for the Model file, plus the Runtime images.

### Run

From the repository root in PowerShell:

```powershell
# 1. Start both Runtimes. The first start downloads the Core model file (~6 GB).
docker compose up -d
docker compose logs -f model-fetch   # follow the download, Ctrl+C when it says "ready"

# 2. Check the prerequisites: Docker, GPU access, disk space, both Runtimes and the model.
uv run harness doctor

# 3. Send one GSM8K problem to each Runtime and record the result.
uv run harness skeleton
```

`harness doctor` prints a pass/fail checklist and exits non-zero if anything is missing. It also detects the platform (Windows + NVIDIA → Runtimes in Docker; macOS on Apple Silicon → Runtimes native on Metal) and the Hardware tier from the memory available to models: the VRAM of the largest GPU, or 75% of unified memory on Apple Silicon. S is below 12 GB, M up to 24 GB, L up to 48 GB, XL from 48 GB. Platform and tier are recorded in every result file's machine description.

`docker compose up -d` does all of this:

- downloads `Qwen3.5-9B-Q4_K_M.gguf` once into the shared `models` volume and records its sha256 in `.state/`;
- starts LocalAI (CUDA image), which loads the file through `docker/localai/qwen3.5-9b.yaml`;
- starts Ollama and registers the same file as `qwen3.5-9b` through `docker/ollama/Modelfile`.

Both Runtimes use the same Model file at the same 16k Context window.

The harness waits until both Runtimes list the model, then queries them **one after the other**. After each answer it unloads the model, so the two copies never share the 16 GB VRAM. It writes:

- `results/<machine-id>/<timestamp>-gsm8k-skeleton.json`: correct/incorrect per Runtime, plus the machine description (CPU, GPU, memory, OS, Runtime versions, script version) and the Model file checksum. This file is meant to be committed.
- `runs/<machine-id>/…`: the raw model responses. These stay on your machine (git-ignored); the result file holds only their checksums.

Options: `--index N` picks another GSM8K problem, `--machine-id NAME` overrides the derived machine id.

### Longer runs: resumable and fault-tolerant

```powershell
uv run harness gsm8k --count 50
```

asks both Runtimes the first 50 GSM8K test problems (`--start N` picks the first index; without `--count` it runs the whole 1,319-problem split). Such runs can take all night, so:

- **Resume:** progress is journaled after every answer under `runs/<machine-id>/<run-id>/`. If the run is killed (Ctrl+C, crash, reboot), run **the same command** again: finished questions are not asked again. The run id comes from the model, the Model file (name and sha256), the Runtimes and the exact prompts, so different options, or a re-downloaded Model file, start a separate run.
- **Retries:** connection drops, timeouts and HTTP 408/429/5xx are retried after 10, 30 and 90 seconds. A question that still fails is recorded as failed (it counts as incorrect) and the run goes on; resuming asks failed questions again. If a Runtime stays unreachable, the run stops and keeps its progress.
- **What is committed:** `results/<machine-id>/<start time>-gsm8k-qwen3.5-9b.json` holds per-Runtime aggregates (correct, failed with their errors, accuracy, token counts, throughput) and a checksum over the raw responses. The raw responses and the journal stay in the git-ignored `runs/`. To verify the checksum, take the answered questions in index order and hash the lines `<index> <sha256 of runs/…/<runtime>/<index>.json>\n` with sha256.

### Troubleshooting

- **The GPU is not used:** run `docker compose logs localai ollama` and look for CUDA in the output. Re-check the `nvidia-smi` test above.
- **LocalAI does not answer:** on first start it installs its llama.cpp backend, which can take a few minutes. Check `docker compose logs -f localai`.
- **Ollama does not list the model:** run `docker compose logs ollama-model`; it registers the model once Ollama is healthy.
- **The two Runtimes answer very differently:** the weights are identical, but each Runtime builds the prompt from the chat template in the GGUF on its own. Compare the two raw responses in `runs/<machine-id>/`. If Ollama's output looks unformatted (no `<think>` handling, chat tokens in the text), its template detection failed and `docker/ollama/Modelfile` needs an explicit `TEMPLATE`.
- **Image version not found:** the Runtime versions are pinned in `docker-compose.yml`. Override them in a `.env` file, e.g. `OLLAMA_VERSION=...` or `LOCALAI_VERSION=...`.

## Development

```bash
uv run pytest   # tests
uv run mypy     # type checking
```
