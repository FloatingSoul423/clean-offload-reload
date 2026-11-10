# KV 缓存卸载与恢复

本仓库提供一套面向长上下文、工具调用型大模型会话的 KV 缓存卸载与恢复算法，
基于 [SGLang](https://github.com/sgl-project/sglang) 实现。默认分支包含当前选定的
**v9.1 需求门控服务曲线控制器**。

仓库只保留运行时补丁、控制器、启动脚本、测试和必要说明。模型、数据集、缓存、
日志、传输跟踪、测试结果和打包后的运行时均不会纳入版本控制。

## v9.1 的作用

控制器预测每次工具调用可能返回的时间，并把该会话缺失的 KV 数据视为一条共享
H2D 链路上的恢复任务。只有当所有 EDF 截止时间前缀仍可完成恢复时，控制器才
允许卸载。v9.1 还要求出现真实 HBM 压力后才创建新的驱逐任务，并使用高低双
水位，减少反复触发紧急恢复。

选定的默认参数如下：

- 共享 H2D 带宽上限：8 GiB/s，并根据实际传输吞吐率修正；
- 服务曲线安全系数：0.9；
- HBM 保留空间低/高水位：5% / 30%；
- 预测恢复提前量：1 秒；
- 恢复分块：0.5 GiB，全局冷却时间为 200 ms；
- 每次工具等待最多开启两个条件截止时间 epoch。

算法公式、版本演化、实验依据和已知限制见 [算法说明](ALGORITHM.md)。

## 仓库结构

```text
controller/swe_replay.py       v9.1 控制器和测试负载
runtime/*.patch                SGLang 运行时集成与安全修复
scripts/prepare_runtime.py     可复现的 SGLang 检出与补丁脚本
scripts/launch_server.sh       带路径保护的本地服务启动脚本
scripts/run_benchmark.sh       单次基准测试入口
tests/test_service_curve.py    不需要 GPU 的控制器测试
```

## 安装

环境要求：Linux、Git、Python 3.10+、CUDA，以及 SGLang v0.5.21 的构建依赖。

```bash
git clone https://github.com/FloatingSoul423/clean-offload-reload.git
cd clean-offload-reload
python scripts/prepare_runtime.py ./sglang-runtime
python -m venv .venv
source .venv/bin/activate
pip install -e "./sglang-runtime/python[all]"
pip install -r requirements.txt
```

准备脚本会检出公开的 SGLang 提交
`e00930c5489053f26d86b179cee0d087f846acbb`（`v0.5.21`），校验随仓库提供的
补丁哈希，并应用补丁。脚本不会创建隐藏提交。

## 运行

先在一个终端启动服务。请选用空闲 GPU，并为存储目录指定一个不重复的名称。
结果、传输跟踪和 HiCache 存储都被限制在 `WORKSPACE_ROOT` 下。

```bash
export WORKSPACE_ROOT=$PWD/work
export ALLOWED_ROOT=$WORKSPACE_ROOT
export SGLANG_SRC_DIR=$PWD/sglang-runtime
export PYTHON_BIN=$PWD/.venv/bin/python
export MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
mkdir -p "$WORKSPACE_ROOT"
bash scripts/launch_server.sh 0 16 run-001
```

然后在仓库根目录的另一个终端运行控制器：

```bash
export PYTHON_BIN=$PWD/.venv/bin/python
export MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
bash scripts/run_benchmark.sh
```

默认基准测试运行 4 个相互独立、形状近似 SWE-agent 的会话，持续 120 秒。脚本会
从 Hugging Face 下载公开的轨迹形状和 SWE-bench 问题文本。离线运行时，可以把
`ISSUES_DATASET` 指向一个本地 `datasets.load_from_disk` 数据集目录。可以通过
`SEED`、`SESSIONS`、`DURATION`、`INITIAL_CONTEXT_TOKENS` 和 `OUTPUT_JSON`
覆盖默认负载参数。

## 测试

控制器测试不需要 GPU，也不需要正在运行的模型服务。

```bash
python tests/test_service_curve.py
python -m py_compile controller/swe_replay.py scripts/prepare_runtime.py
```

## 版本

每个里程碑标签都对应一个真实、可运行的 Git 快照：

| 标签 | 内容 |
|---|---|
| `v8.1-runtime` | 修复后的 SGLang 运行时、截止时间驻留下限和 residency 安全修复 |
| `v9.0` | 累计共享链路 EDF 服务曲线 |
| `v9.1` | 需求门控、HBM 双水位、实测带宽和分块恢复；当前选定版本 |

使用 `git checkout <标签>` 可以查看或运行对应里程碑。后续还测试过 Goodput-first
和更激进的高峰感知变体，但它们在不同 seed 间的 Goodput 波动较大，因此未被
选为默认版本。

## 许可证与来源

本项目使用 Apache-2.0 许可证。运行时补丁基于 SGLang v0.5.21；SGLang 的著作权
归其各自贡献者所有。
