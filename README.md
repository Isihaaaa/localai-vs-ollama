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

asks both Runtimes the first 50 GSM8K test problems (`--start N` picks the first index). Which problems, which Runtimes, which Thinking mode and which Sampling profile:

- **Problems:** without options, the whole 1,319-problem test split. `--subset N` takes a fixed, reproducible subset of N problems: the split shuffled once with a fixed seed, so every machine gets the same problems, and a smaller subset is always part of a larger one. `--start N --count M` takes a contiguous range. The result file records the choice as `benchmark.selection` (`full`, `subset-500`, `range-0-49`).
- **Thinking mode:** switched explicitly with every request, never left to the model's default. `--thinking on` (default) sends `reasoning_effort: "high"`, `--thinking off` sends `"none"`; for an on/off-only model like Qwen3.5 both LocalAI and Ollama turn these into `enable_thinking=true/false` in the chat template. The prompt and every other setting stay the same. *Reasoned* in the results counts answers in which reasoning was actually observed: with Thinking mode off it should be 0, otherwise the Runtime ignored the switch.
- **Sampling profile:** every decoding setting is pinned, identically on both Runtimes, so neither falls back to its own defaults. `--sampling vendor` (default) is Qwen's recommendation for Qwen3.5 on reasoning tasks, the same in both Thinking modes: `temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, repeat_penalty 1.0`. `--sampling greedy` is `temperature 0, top_p 1.0, presence_penalty 0` with the rest the same. Both use a fixed seed. Qwen warns that greedy decoding makes Thinking mode repeat itself endlessly; it is kept for the pilot below. The harness sends the profile with every request; `top_k`, `min_p` and `repeat_penalty` are also pinned in `docker/localai/qwen3.5-9b.yaml` and `docker/ollama/Modelfile`, because Ollama's OpenAI-compatible endpoint ignores them in a request (a test keeps the three places in step). After changing those files, run `docker compose up -d --force-recreate localai ollama-model`. With `vendor` sampling the verification run on the other Runtime confirms the results statistically, not answer by answer.
- **Runtimes:** `--runtime localai` asks only that Runtime (repeat the option for several). Accuracy needs only one Runtime; the other confirms it on a subset.

Thinking mode and Sampling profile go into the run id and the result file name, so different settings never resume or overwrite each other.

**Pilot before the full run:** 50 problems, Thinking mode on, one Runtime, `vendor` against `greedy`. If `vendor` has clearly fewer truncated answers, it stays the default.

```powershell
uv run harness gsm8k --subset 50 --runtime localai --sampling vendor
```

```powershell
uv run harness gsm8k --subset 50 --runtime localai --sampling greedy
```

Then accuracy of the Core model on one Runtime, on the full split, in both Thinking modes:

```powershell
uv run harness gsm8k --runtime localai --thinking on
```

```powershell
uv run harness gsm8k --runtime localai --thinking off
```

Such runs can take all night, so:

- **Progress:** every answer, retry and resumed Runtime is printed with a timestamp, e.g. `21:14:03 ollama 12/50 #11: correct (20.2 s, 1158 tokens)`.
- **Resume:** progress is journaled after every answer under `runs/<machine-id>/<run-id>/`. If the run is killed (Ctrl+C, crash, reboot), run **the same command** again: finished questions are not asked again. The run id comes from the model, the Model file (name and sha256), the Runtimes, the Thinking mode and the exact prompts, so different options, or a re-downloaded Model file, start a separate run.
- **Retries:** connection drops and HTTP 408/429/5xx are retried after 10, 30 and 90 seconds. A timeout is retried only once: with a fixed seed a model that does not finish within the request timeout usually does not finish on the next attempt either. A question that still fails is recorded as failed (it counts as incorrect) and the run goes on; resuming asks failed questions again. If a Runtime stays unreachable, the run stops and keeps its progress.
- **What is committed:** `results/<machine-id>/<start time>-gsm8k-qwen3.5-9b-thinking-<on|off>-<sampling profile>.json` holds the Run configuration (Model file, Thinking mode, Sampling profile with every setting, Context window, which problems) and per-Runtime aggregates and a checksum over the raw responses. The aggregates are: correct, failed (with their errors), truncated, retries, accuracy, token counts, throughput, and the cold model load time (the first load of the run, since a resumed run can find the model still in memory). *Truncated* counts answers cut off at the token limit (`finish_reason: "length"`), typically a Thinking mode that used up its budget or got stuck in a loop. They are reported next to accuracy and **count as wrong**, even when the cut-off text happens to contain the right number: the user got no answer. *Retries* counts attempts beyond the first, so a flaky night shows up in the results. The raw responses and the journal stay in the git-ignored `runs/`. To verify the checksum, take the answered questions in index order and hash the lines `<index> <sha256 of runs/…/<runtime>/<index>.json>\n` with sha256. The checksum only proves that the local raw responses are the ones the aggregates were computed from. It does not prove reproducibility: rerunning gives different bytes, if only because responses carry an `id` and a `created` time.

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
