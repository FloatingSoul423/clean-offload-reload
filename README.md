# Clean Offload/Reload

Minimal, reproducible code for deadline-aware KV-cache offload and reload in
[SGLang](https://github.com/sgl-project/sglang). This release adds the v9.0
cumulative EDF service-curve controller to the patched runtime.

## Prepare the runtime

Requirements: Linux, Git, Python 3.10+, CUDA, and the normal SGLang build
dependencies. The preparation script checks out the public SGLang `v0.5.21`
commit and applies the runtime patch without committing it.

```bash
python scripts/prepare_runtime.py ./sglang-runtime
cd sglang-runtime
pip install -e "python[all]"
```

The patch adds the session-residency control API, transfer tracing, deadline
floors, idle reserve control, and the residency safety fixes required by the
controllers. It is kept as one reviewable patch rather than a vendored SGLang
tree.

## Start SGLang

Choose an idle GPU and a unique storage directory. All generated data remains
under `WORKSPACE_ROOT`.

```bash
export WORKSPACE_ROOT=$PWD/work
export ALLOWED_ROOT=$WORKSPACE_ROOT
export SGLANG_SRC_DIR=$PWD/sglang-runtime
export PYTHON_BIN=$PWD/.venv/bin/python
export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
mkdir -p "$WORKSPACE_ROOT"

bash scripts/launch_server.sh 0 16 run-001
```

The server binds to `127.0.0.1:30000`. Set `SGLANG_POLICY=baseline` for a
runtime baseline or leave it unset for deadline-floor support.

## Run the controller

Install the small benchmark dependencies, start the server in one terminal,
then run the controller from the repository root in another terminal:

```bash
pip install -r requirements.txt
export MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
bash scripts/run_benchmark.sh
```

The default workload uses four independent SWE-agent-shaped sessions for 120
seconds. It downloads public trajectory shapes and SWE-bench issue text through
Hugging Face. Set `ISSUES_DATASET` to a local `datasets.load_from_disk` directory
to run offline. Outputs go to the ignored `results/` directory.

The controller enforces cumulative shared-link feasibility:

```text
C(t) = safety * observed_H2D_bandwidth * t
D(t) = sum(missing KV bytes with deadline <= t)
require D(t) <= C(t) for every deadline prefix
```

v9.0 removes the older per-request bandwidth cap. It can use the full idle H2D
link for one session while still accounting for all sessions together.

Run the controller tests without a model or GPU:

```bash
python tests/test_service_curve.py
```

## License and provenance

The runtime patch is derived from SGLang `v0.5.21`
(`e00930c5489053f26d86b179cee0d087f846acbb`) and is distributed under the
Apache License 2.0. SGLang remains copyright its respective contributors.
