# Project plan

Decisions from the planning session on 2026-11-16. Terms follow [`CONTEXT.md`](../CONTEXT.md); architectural decisions are in [`docs/adr/`](adr/).

## Goal

Build an on-premise generative AI environment with **LocalAI** and **Ollama**, compare the two Runtimes, and show with numbers where small local Models stand against the Reference model (Claude). We look at general Benchmarks and, above all, at how usable a local Model is behind our Coding agent (Claude Code). The outcome should let us tell a client what an on-premise solution can and cannot do today.

## Environment

- **Development machine (tier M):** RTX 3080 Laptop GPU (16 GB VRAM), 64 GB RAM, Windows + Docker Desktop/WSL2.
- **Hardware profiles:** fully in VRAM, VRAM + RAM offload, CPU-only (CPU-only on subsets only).
- **Runtimes:** LocalAI and Ollama in Docker, from one shared `docker-compose.yml`. Ollama is also measured once natively, to quantify the cost of the WSL2 layer.
- **Offline proof:** containers run on an internal Docker network (`internal: true`) with pre-downloaded Model files, plus a recorded demo with Wi-Fi switched off.
- **Chat demo:** Open WebUI, connected to both Runtimes. Claude is deliberately not connected, so the "no external AI API" claim stays clean.

## Reproducibility: anyone can run the measurements

- **Supported platforms:** Windows + NVIDIA GPU (Docker Desktop/WSL2) and macOS on Apple Silicon. On macOS the Runtimes run natively (Metal), because Docker containers cannot reach the GPU there. Open WebUI and the Sample system backend can stay in Docker. Scripts detect the platform and pick the right path.
- **Hardware tiers:** the tooling measures the memory available to models (VRAM, or unified memory on Apple Silicon) and assigns a tier: S / M / L / XL (below 12 GB / 12–24 GB / 24–48 GB / from 48 GB; on Apple Silicon 75% of unified memory counts, the macOS default GPU share). Each tier has a fixed set of Models. The Models below are tier M. The S / L / XL sets are still to be defined.
- **Core model:** one Model runs on every tier (proposed: Qwen3.5 9B), so any two machines can be compared directly.
- **Run presets:**
  - `quick` (~1 h): Core model, fixed subsets, speed measurements, both Runtimes, no Agent trial.
  - `standard` (one night): all Models of the tier on subsets, both Thinking modes. The Agent trial is optional behind a flag.
  - `full` (several nights): full datasets, all modes, Agent trial included.
- **`doctor` command:** checks prerequisites (Docker, GPU access, disk space, Runtimes, downloaded Models) before a run.
- **Result collection:** results go into `results/<machine-id>/` with an automatically captured machine description (CPU, GPU, memory, OS, Runtime versions, tier, script version) and are submitted as a pull request. Raw model responses stay local; only aggregates and a checksum are committed. The dashboard is built from everything under `results/`.
- **Reference results:** Claude is measured once, centrally, with the company API key, and stored in `results/reference/` with the exact model version and date. Everyone else runs local-only, with no API key. Anyone with their own key can re-run the reference (e.g. when a new Claude version ships); the tooling warns about the expected cost first.

## Models (tier M)

| Hardware profile | Model | Quantization |
|---|---|---|
| VRAM | Qwen3.5 9B (Core model) | Q4_K_M |
| VRAM | Gemma 4 12B | Q4_K_M |
| VRAM | Qwen2.5-Coder 14B (coding specialist) | Q4_K_M |
| VRAM | gpt-oss-20b | MXFP4 (native) |
| VRAM + RAM offload | Qwen3.6-35B-A3B (MoE) | Q4_K_M |

- **Reference models:** Claude Sonnet 5.5 (primary) and Claude Haiku 5.5 on everything. Claude Opus 5.5 only on GSM8K and HumanEval, as a ceiling.
- **Thinking mode:** every Model runs both with and without thinking.
- **Context window:** 16k for all Benchmarks. 32k for every local Model in the Agent trial, plus an extra 64k run for the best local Model.

## Benchmarks

Full datasets in the `full` preset:

- **MMLU:** ~14,000 questions, 57 subjects.
- **GSM8K:** 1,319 problems (official test split).
- **HumanEval:**
  - Kotlin: `JetBrains/Kotlin_HumanEval`, 161 problems
  - C# and TypeScript: MultiPL-E, ~160 per language
  - Python: HumanEval+, 164 problems

How it runs:

- An in-house Python harness talks to every Runtime through the OpenAI-compatible API and to Claude through the Anthropic SDK.
- Scoring is generative (ADR-0001).
- Runs are resumable, retry on transient errors and keep raw responses.
- Accuracy is measured once per Model on one Runtime. A ~500-question verification run on the other Runtime confirms the results match. CPU-only runs use the same ~500-question subset.

## Runtime comparison

- **Measured:** tokens/s, time to first token, model load time, VRAM/RAM use, and throughput under 1, 4 and 8 concurrent requests.
- **Features** (works / partially / not):
  - all Runtimes: OpenAI API, Anthropic API, tool calling, JSON-schema output, embeddings, image input
  - LocalAI only: image generation, speech-to-text, text-to-speech
- **Operations** (scored): installation, model management, configurability, documentation, UI, community, licence.

## Agent trial

Claude Code drives every model (ADR-0002).

- **Sample system:** a todo app ("Todo items"). A .NET Web API backend serves two clients with identical functionality, a Kotlin Android app and a React Native (TypeScript) app, all running together locally.
- **11 Agent tasks:**
  - for each layer (Android / React Native / .NET): one new feature, one bug fix, one refactor (9 tasks)
  - 2 Cross-layer agent tasks: a simple one (a new field through all layers) and a complex one (a new entity with relations, CRUD in all layers)
- **Each task has:** a fixed task prompt, hidden acceptance tests and a verified reference solution.
- **Rules:**
  - fully autonomous (`claude -p`), plus a single standard "nudge" variant reported as a separate metric
  - limit: 30 minutes or 50 tool calls
  - 3 repetitions per Run configuration, each on a fresh working copy, with no permission prompts
- **Configurations:** Claude Code + Claude, Claude Code + Ollama, Claude Code + LocalAI.
- **Bias handling:** Claude Code builds the Sample system. A human reviews and rewrites the task prompts and hidden tests. The documentation openly discusses this bias and possible benchmark contamination.

## Cost

- **Three views:**
  - marginal cost: electricity, from GPU power readings
  - full cost: electricity plus hardware amortization
  - break-even: the monthly token volume above which local hardware is cheaper than the Claude API
- Computed for the laptop and for a hypothetical server.
- **Units:** HUF per 1M tokens and HUF per Agent task.
- **Parameters** (electricity price, hardware price, amortization period, utilization) live in a config file. For now they are market averages, marked as estimates.

## Deliverables

- **Repository (English):** technical documentation (including a "What didn't work" chapter), results, generated charts, `CONTEXT.md`, ADRs.
- **Internal summary (Hungarian):** a document plus an interactive dashboard.
- **Client material (Hungarian):** a short 5–10 slide deck.

## Open items

- Define the model sets for tiers S, L and XL.
- Exact electricity and hardware prices for the cost model.
