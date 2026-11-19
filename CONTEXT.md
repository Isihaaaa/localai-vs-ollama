# LocalAI vs Ollama

Running generative AI on-premise with LocalAI and Ollama: comparing the two tools, and measuring how small open models perform against a cloud reference (Claude), both on public benchmarks and as the model behind a coding agent.

## Language

### Running models

**Runtime** (HU: futtatókörnyezet):
Server software that loads Model files and serves them over an HTTP API. In this project: LocalAI and Ollama.
_Avoid_: backend, platform, engine

**Inference engine** (HU: inferencia-motor):
The library underneath a Runtime that actually performs the model computation (e.g. llama.cpp).
_Avoid_: backend, engine (alone), runtime

**Platform** (HU: platform):
The operating system and GPU combination a machine runs, which decides where the Runtimes run: Windows + NVIDIA GPU (Runtimes in Docker) or macOS on Apple Silicon (Runtimes native on Metal).
_Avoid_: OS (alone), host

**Hardware profile** (HU: hardverprofil):
How the model is placed on the machine during a run: fully in VRAM, VRAM + RAM offload (part of the model in system memory), or CPU-only.
_Avoid_: environment, setup

**Hardware tier** (HU: hardverszint):
A class of machines grouped by the memory available to models (dedicated VRAM, or unified memory on Apple Silicon), e.g. S / M / L / XL. Each tier has a fixed set of Models.
_Avoid_: machine class, level

**Core model** (HU: mag modell):
The one Model that runs on every Hardware tier, so that any two machines can be compared directly.
_Avoid_: baseline model, common model

**Context window** (HU: kontextusablak):
The maximum number of tokens the model sees in one request: prompt, history, files read and its own answer combined.
_Avoid_: context, num_ctx, memory

### Models

**Model** (HU: modell):
An open-weight language model, independent of any Runtime (e.g. gpt-oss-20b, Gemma 4 12B).
_Avoid_: LLM, AI

**Quantization** (HU: kvantálás):
The storage precision of a Model's weights (e.g. Q4_K_M ≈ 4 bits per weight); smaller and faster at a small quality cost.
_Avoid_: compression, quant

**Model file** (HU: modellfájl):
One Model at one Quantization, as a single GGUF file that Runtimes load.
_Avoid_: checkpoint, weights file

**Reference model** (HU: referenciamodell):
A cloud-hosted Claude model that local Models are compared against.
_Avoid_: Claude Code (that is the Coding agent), frontier model

### Coding

**Coding agent** (HU: kódoló ágens):
A program that drives a model in a tool-using loop (reading/writing files, running commands) to complete a development task. In this project: Claude Code.
_Avoid_: agent (alone), harness, "Claude" when the agent is meant

### Measurement

**Benchmark** (HU: benchmark):
An automatically scored measurement on a public dataset, with the same questions for every model (MMLU, GSM8K, HumanEval in Kotlin / C# / TypeScript / Python).
_Avoid_: test, eval (alone)

**Thinking mode** (HU: gondolkodási mód):
Whether the model may reason before answering (reasoning / extended thinking, or a prompt asking for step-by-step working) or answers directly.
_Avoid_: reasoning, CoT, thinking (alone)

**Run configuration** (HU: mérési konfiguráció):
All fixed parameters of one run: Model file, Runtime (or Reference model), Hardware profile, Thinking mode, Context window, and the Benchmark or Agent task.
_Avoid_: setup, run, settings

**Runtime comparison** (HU: futtatókörnyezet-összehasonlítás):
One Benchmark with one Model file asked of every Runtime on one machine: one Run configuration per Runtime, identical except for the Runtime. Measuring accuracy on a single Runtime is the same run with only that Runtime's Run configuration.
_Avoid_: benchmark run, comparison run

**Run preset** (HU: mérési csomag):
A named, predefined bundle of Run configurations a person starts with one command: `quick` (Core model, fixed subsets), `standard` (all Models of the Hardware tier, subsets) or `full` (full datasets, Agent trial included).
_Avoid_: profile (that word belongs to Hardware profile), mode, suite

**Reference results** (HU: referencia-eredmények):
Results of the Reference model, measured once centrally and stored with the exact model version and date. Local runs are compared against them, with no API key needed.
_Avoid_: baseline, Claude run

**Agent trial** (HU: ágens-próba):
Completing realistic development tasks on the Sample system with the Coding agent, once with the Reference model and once with local Models.
_Avoid_: benchmark, demo

**Agent task** (HU: ágens-feladat):
A fixed-wording development task on the Sample system, with hidden acceptance tests, given to the Coding agent.
_Avoid_: task (alone), ticket, test case

**Cross-layer agent task** (HU: rétegen átívelő ágens-feladat):
An Agent task that changes all three layers of the Sample system (backend and both clients).
_Avoid_: full-stack task, end-to-end task

**Sample system** (HU: mintarendszer):
The locally co-running system built for Agent trials: a .NET Web API backend and two clients with identical functionality (Kotlin Android and React Native).
_Avoid_: mini project, demo app, sample

### Sample system domain

**Todo item** (HU: teendő):
The Sample system's main entity: a unit of work in a project, with status, due date and assignee.
_Avoid_: task (that word belongs to measurement), todo

## Relationships

- A **Model file** is exactly one **Model** at one **Quantization**.
- The same **Model file** loads in both **Runtimes**. When comparing Runtimes the Model file is fixed and only the Runtime varies.
- A **Run configuration** has exactly one **Runtime**; a **Runtime comparison** bundles one Run configuration per Runtime.
- Both **Runtimes** typically use the same **Inference engine** (llama.cpp), so with the same Model file quality is identical and the difference lies in speed and operations.
- The **Coding agent** is model-independent: the same Claude Code can be driven by the **Reference model** or by a local **Model file** served by a **Runtime**.
- An **Agent trial** is a set of **Agent tasks**, each run under one **Run configuration**.

## System overview

How the whole system fits together. Solid boxes exist in the repo today; dashed boxes are planned (see [`docs/plan.md`](docs/plan.md)). The diagrams render on GitHub.

### The big picture

```mermaid
flowchart LR
    person(["Person"])

    subgraph onprem["On-premise machine: works offline once the Model files are downloaded"]
        harness["Harness<br/>uv run harness …"]
        webui["Open WebUI<br/>chat demo"]:::planned
        claudecode["Claude Code<br/>Coding agent"]:::planned
        subgraph runtimes["Runtimes, in Docker (native on macOS)"]
            localai["LocalAI :8080"]
            ollama["Ollama :11434"]
        end
        engine["llama.cpp<br/>Inference engine"]
        modelfile[("Model file<br/>GGUF, same for both")]
        gpu["GPU / RAM / CPU<br/>Hardware profile"]
        sample["Sample system<br/>.NET + Kotlin Android + React Native"]:::planned
        results[("results/MACHINE_ID/")]
    end

    subgraph cloud["Cloud"]
        claude["Claude API<br/>Reference models"]:::planned
        repo["GitHub repo<br/>results from every machine"]
        dashboard["Dashboard, charts,<br/>client deck"]:::planned
    end

    person --> harness
    person -.->|"chat"| webui
    harness -->|"OpenAI-compatible API"| localai
    harness -->|"OpenAI-compatible API"| ollama
    webui -.-> localai
    webui -.-> ollama
    claudecode -.->|"ANTHROPIC_BASE_URL"| localai
    claudecode -.->|"ANTHROPIC_BASE_URL"| ollama
    claudecode -.->|"Reference model"| claude
    claudecode -.->|"Agent tasks"| sample
    harness -.->|"Reference results, once, centrally"| claude
    localai --> engine
    ollama --> engine
    engine --> modelfile
    engine --> gpu
    harness --> results
    results -->|"pull request"| repo
    repo -.-> dashboard

    classDef planned stroke-dasharray: 5 5
```

### Preparing the Runtimes (`docker compose up -d`)

Both Runtimes load the very same GGUF file, and its sha256 goes into every result file to prove it.

```mermaid
flowchart TB
    hf["Hugging Face<br/>Qwen3.5-9B-Q4_K_M.gguf"]
    fetch["model-fetch (one-shot)<br/>docker/fetch-model.sh<br/>resumable download, sha256"]
    models[("models volume<br/>the Model file")]
    state[(".state/MODEL_FILE.sha256<br/>read by the Harness")]
    lacfg["docker/localai/qwen3.5-9b.yaml<br/>context_size 16384, gpu_layers 99,<br/>temperature 0, GGUF chat template"]
    olcfg["docker/ollama/Modelfile<br/>num_ctx 16384, temperature 0"]
    ollamamodel["ollama-model (one-shot)<br/>ollama create qwen3.5-9b"]
    localai["localai container :8080<br/>CUDA llama.cpp backend"]
    ollama["ollama container :11434"]
    gpu["NVIDIA GPU via WSL2<br/>(Metal, native, on macOS)"]

    hf -->|"first start only"| fetch
    fetch --> models
    fetch --> state
    models --> localai
    lacfg --> localai
    models --> ollamamodel
    olcfg --> ollamamodel
    ollamamodel -->|"registers qwen3.5-9b"| ollama
    localai --> gpu
    ollama --> gpu
```

### Inside the Harness

```mermaid
flowchart TB
    person(["Person"])

    subgraph cli["cli.py: commands"]
        doctor["doctor<br/>prerequisite checklist"]
        skeleton["skeleton<br/>one GSM8K problem"]
        gsm8k["gsm8k<br/>resumable Runtime comparison"]
        presets["quick / standard / full<br/>Run presets"]:::planned
    end

    subgraph modules["Modules"]
        machine["machine.py<br/>machine id, Platform,<br/>Hardware tier S/M/L/XL"]
        bench["gsm8k.py<br/>problems + prompt"]
        otherbench["MMLU, HumanEval<br/>Kotlin / C# / TS / Python"]:::planned
        run["benchmark_run.py<br/>journal, resume, retries"]
        adapters["runtimes.py<br/>LocalAI / Ollama adapters<br/>load, chat, unload"]
        refadapter["Anthropic SDK adapter"]:::planned
        scoring["answers.py + results.py<br/>generative scoring, ADR-0001"]
    end

    data[("data/gsm8k/test.jsonl")]
    runs[("runs/MACHINE_ID/RUN_ID/<br/>manifest, journal, raw responses<br/>local only")]
    results[("results/MACHINE_ID/*.json<br/>aggregates, machine description,<br/>checksum of raw responses")]
    reference[("results/reference/")]:::planned
    runtimes["LocalAI and Ollama"]
    claude["Claude API"]:::planned

    person --> doctor
    person --> skeleton
    person --> gsm8k
    person -.-> presets
    presets -.-> gsm8k
    presets -.-> otherbench
    doctor --> machine
    doctor --> adapters
    gsm8k --> bench
    data --> bench
    gsm8k --> run
    skeleton --> adapters
    run --> adapters
    adapters -->|"HTTP, one Runtime at a time"| runtimes
    run --> scoring
    otherbench -.-> refadapter
    refadapter -.-> claude
    claude -.-> reference
    run --> runs
    scoring --> results
    machine --> results

    classDef planned stroke-dasharray: 5 5
```

### Agent trial (ADR-0002, planned)

```mermaid
flowchart LR
    task["Agent task<br/>fixed prompt, 11 tasks"]
    copy["Fresh working copy<br/>of the Sample system"]
    cc["Claude Code<br/>claude -p, limit 30 min / 50 tool calls"]
    subgraph backends["Run configurations"]
        claude["Claude API<br/>Reference model"]
        ol["Ollama"]
        la["LocalAI"]
    end
    tests["Hidden acceptance tests"]
    out[("pass / fail, time, cost<br/>3 repeats per configuration")]

    task --> cc
    copy --> cc
    cc --> claude
    cc --> ol
    cc --> la
    cc -->|"edits code"| copy
    copy --> tests --> out
```

### Layers of one request

The same Model file goes through both Runtimes, so a difference in answers points at Runtime configuration (sampling defaults, chat template, Thinking mode handling), not at the Model.

```mermaid
flowchart LR
    caller["Caller<br/>Harness, Open WebUI or Claude Code"]
    subgraph rt["Runtime"]
        la["LocalAI"]
        ol["Ollama"]
    end
    eng["Inference engine<br/>llama.cpp"]
    mf[("Model file<br/>one Model at one Quantization, GGUF")]
    hw["Hardware<br/>CUDA GPU, Metal or CPU"]

    caller -->|"/v1/chat/completions"| la
    caller -->|"/v1/chat/completions"| ol
    la --> eng
    ol --> eng
    eng -->|"loads weights"| mf
    eng --> hw
```

### One Runtime comparison (`uv run harness gsm8k`)

```mermaid
sequenceDiagram
    autonumber
    actor P as Person
    participant H as Harness (cli.py, benchmark_run.py)
    participant J as runs/ journal (local)
    participant L as LocalAI
    participant O as Ollama
    participant R as results/ (committed)

    P->>H: uv run harness gsm8k [--subset N | --start N --count M] [--thinking on|off] [--runtime R]
    H->>H: load GSM8K problems, describe machine, read Model file sha256
    H->>L: GET /v1/models until the model is listed
    H->>O: GET /v1/models until the model is listed
    H->>J: open manifest + journal for this Runtime comparison (run id)
    Note over H,J: run id = hash of Benchmark, Model, Model file,<br/>Runtimes, Thinking mode, max_tokens and the exact prompts.<br/>Same command again = resume.

    loop each Runtime, strictly one after the other
        H->>J: skip questions already answered
        H->>L: GET version
        H->>L: warm-up request (1 token) = load time
        loop each pending question
            H->>L: POST /v1/chat/completions (temperature 0, max_tokens 12000,<br/>reasoning_effort "none" when Thinking mode is off)
            L-->>H: response (content, usage, finish_reason)
            H->>H: drop the think block, extract "Answer: number", compare to reference
            H->>J: append outcome + raw response
        end
        H->>L: unload the model, VRAM is free for the next Runtime
        Note over H,O: then the same steps against Ollama
    end

    alt a Runtime stays unreachable after retries (10 s / 30 s / 90 s)
        H-->>P: RunInterrupted, progress kept in the journal
    else every question has an outcome
        H->>R: write results/MACHINE_ID/STAMP-gsm8k-MODEL-thinking-MODE.json<br/>accuracy, truncations, tok/s, machine, checksum
        H-->>P: path of the result file
    end
```
