# Clean Offload/Reload

Minimal, reproducible code for deadline-aware KV-cache offload and reload in
[SGLang](https://github.com/sgl-project/sglang). This tag contains the patched
runtime used by the later service-curve controllers. Check out the latest
release tag for the current algorithm and benchmark driver.

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

## License and provenance

The runtime patch is derived from SGLang `v0.5.21`
(`e00930c5489053f26d86b179cee0d087f846acbb`) and is distributed under the
Apache License 2.0. SGLang remains copyright its respective contributors.
