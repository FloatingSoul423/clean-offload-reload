# Clean Offload/Reload

Deadline-aware KV-cache offload and reload for long, tool-using LLM sessions on
[SGLang](https://github.com/sgl-project/sglang). The default branch contains
the selected **v9.1 demand-gated service-curve controller**.

This repository intentionally contains only the runtime patch, controller,
launch scripts, tests, and concise documentation. Models, datasets, caches,
logs, traces, benchmark outputs, and packaged runtimes are excluded.

## What v9.1 does

The controller predicts when each tool call may return and treats the missing
KV for that session as work on one shared H2D link. It admits offload only when
every EDF deadline prefix remains recoverable. v9.1 additionally requires real
HBM pressure before creating new eviction work and uses low/high reserve
watermarks to avoid repeated emergency recovery.

Selected defaults:

- shared H2D ceiling: 8 GiB/s, corrected by observed transfer throughput;
- service-curve safety factor: 0.9;
- HBM reserve low/high: 5% / 30%;
- predictive reload lead: 1 second;
- reload chunk: 0.5 GiB with a 200 ms global cooldown;
- at most two conditional deadline epochs per tool wait.

See [ALGORITHM.md](ALGORITHM.md) for the equations, version history, evidence,
and limitations.

## Repository layout

```text
controller/swe_replay.py       v9.1 controller and benchmark workload
runtime/*.patch                SGLang runtime integration and safety fixes
scripts/prepare_runtime.py     reproducible SGLang checkout and patching
scripts/launch_server.sh       guarded local server launcher
scripts/run_benchmark.sh       single benchmark entry point
tests/test_service_curve.py    GPU-free controller tests
```

## Install

Requirements: Linux, Git, Python 3.10+, CUDA, and the build dependencies for
SGLang v0.5.21.

```bash
git clone https://github.com/FloatingSoul423/clean-offload-reload.git
cd clean-offload-reload
python scripts/prepare_runtime.py ./sglang-runtime
python -m venv .venv
source .venv/bin/activate
pip install -e "./sglang-runtime/python[all]"
pip install -r requirements.txt
```

The preparation script checks out public SGLang commit
`e00930c5489053f26d86b179cee0d087f846acbb` (`v0.5.21`), verifies the bundled
patch hash, and applies it without creating a hidden commit.

## Run

Start the server in one terminal. Choose an idle GPU and a unique storage name.
Generated results, traces, and HiCache storage stay below `WORKSPACE_ROOT`.

```bash
export WORKSPACE_ROOT=$PWD/work
export ALLOWED_ROOT=$WORKSPACE_ROOT
export SGLANG_SRC_DIR=$PWD/sglang-runtime
export PYTHON_BIN=$PWD/.venv/bin/python
export MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
mkdir -p "$WORKSPACE_ROOT"
bash scripts/launch_server.sh 0 16 run-001
```

Run the controller from the repository root in another terminal:

```bash
export PYTHON_BIN=$PWD/.venv/bin/python
export MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
bash scripts/run_benchmark.sh
```

The default benchmark uses four independent SWE-agent-shaped sessions for 120
seconds. It downloads public trajectory shapes and SWE-bench issue text from
Hugging Face. For offline use, set `ISSUES_DATASET` to a local
`datasets.load_from_disk` directory. `SEED`, `SESSIONS`, `DURATION`,
`INITIAL_CONTEXT_TOKENS`, and `OUTPUT_JSON` override the default workload.

## Test

The controller tests require neither a GPU nor a running model server.

```bash
python tests/test_service_curve.py
python -m py_compile controller/swe_replay.py scripts/prepare_runtime.py
```

## Versions

Each milestone is a real, runnable Git snapshot:

| Tag | Purpose |
|---|---|
| `v8.1-runtime` | Patched SGLang runtime, deadline floor, and residency fixes |
| `v9.0` | Cumulative shared-link EDF service curve |
| `v9.1` | Demand gating, HBM hysteresis, measured bandwidth, chunked reload; selected default |

Use `git checkout <tag>` to inspect or run a milestone. Later Goodput-first and
peak-aware variants were tested but not selected because Goodput varied across
seeds.

## License and provenance

Licensed under Apache-2.0. The runtime patch is derived from SGLang v0.5.21;
SGLang remains copyright its respective contributors.
