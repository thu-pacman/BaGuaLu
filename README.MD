# BaGuaLu-2

BaGuaLu-2 is a training-system extension for managing activation-memory pressure in
large-scale, fine-grained Mixture-of-Experts (MoE) training. It integrates with
Megatron-LM and adds pipeline-aware activation offload/reload, an interleaved
1F1B schedule, an analytical activation-memory model, patched Transformer
Engine operators, and MoE load telemetry.

Fine-grained token routing can intermittently send many tokens to experts on the
same GPU. During pipeline warmup, several forward microbatches may keep their
activations live at once, so a short-lived routing imbalance can become a large
HBM spike. Static activation policies must either provision for the worst case
on every iteration or risk an out-of-memory restart. BaGuaLu-2 provides
scheduler-integrated, fixed-budget activation offload/reload at microbatch or
Transformer MLP-layer boundaries, with asynchronous copy streams and optional
reload prefetch.

## Artifact scope

This public artifact packages the analytical retained-activation model for
C1 and the scheduler-integrated, fixed-budget offload/reload data path used by
C2. It does not include the paper's full adaptive per-operator orchestration
among recomputation, offloading, and quantization, or the C3 fused dual-batch
DP scheduler.

The release also does not include the deployment-specific datasets,
checkpoints, complete cluster launcher, or evaluation harness used for the
paper's large-scale results.

## What is included

| Component | Description |
| --- | --- |
| `pretrain_gpt.py` | Megatron GPT/MoE training entry point that loads the BaGuaLu-2 patches before Megatron initializes. |
| `bgl2/adapter/` | Runtime patches for Megatron arguments, schedule selection, MoE telemetry, fused SwiGLU, and Transformer Engine modules. |
| `bgl2/core/models/gpt/gpt_activation.py` | Analytical retained-activation model for attention and MoE operators. |
| `bgl2/core/pipeline_parallel/` | Pipeline schedules and scheduler-scoped activation offload/reload. |
| `ext/TransformerEngine/patch_te/` | BaGuaLu-2-compatible `Linear`, `GroupedLinear`, `LayerNormLinear`, and MoE permutation implementations. |
| `bgl2/data/moe_token_pressure.py` | Parser and report generator for per-rank MoE token-pressure logs. |
| `bgl2/tests/` | Unit tests for patching, scheduling/offload behavior, and telemetry analysis. |
| `scripts/workers/moe_imbalance_minimal.sh` | Site-oriented worker template for collecting MoE imbalance traces. |
| `ext/Megatron-LM` | Pinned Megatron-LM submodule based on Megatron Core v0.13.0. |

## Requirements

- Linux with NVIDIA GPUs
- A CUDA-enabled PyTorch installation with NCCL
- Python 3.10 or newer, as required by the pinned Megatron-LM revision
- Transformer Engine **2.12.0**
- Enough GPUs for the selected tensor, pipeline, expert, and data-parallel
  topology

The Transformer Engine version is strict: the bundled patches reject any
version other than 2.12.0. Ensure that the PyTorch, CUDA, and Transformer Engine
builds in your environment are mutually compatible.

## Installation

Clone the repository with its pinned Megatron-LM submodule:

```bash
git clone --recurse-submodules https://github.com/thu-pacman/BaGuaLu.git
cd BaGuaLu
git submodule update --init --recursive
```

Starting from a CUDA-enabled PyTorch environment, install the matching runtime
dependencies and the two editable packages:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install "transformer-engine[pytorch]==2.12.0"
python -m pip install -e "./ext/Megatron-LM[mlm]"
python -m pip install -e ./bgl2
```

Verify the versions before launching distributed training:

```bash
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)"
python -c "import transformer_engine as te; print('Transformer Engine', te.__version__)"
git -C ext/Megatron-LM describe --tags --always
```

Megatron-LM has additional environment and performance recommendations. See
[`ext/Megatron-LM/README.md`](ext/Megatron-LM/README.md) for its current setup,
dataset preparation, and distributed-launch documentation.

## Quick start

The following four-GPU MoE smoke run uses Megatron's mock dataset. It exercises
virtual pipeline parallelism, BaGuaLu-2's interleaved 1F1B scheduler, patched
Transformer Engine operators, MoE telemetry, and a fixed 64 MiB
activation-offload budget per microbatch. The example requires Ampere or newer
GPUs because Megatron's grouped MoE GEMM path requires compute capability 8.0 or
higher. It also prints modeled and observed activation-memory usage on rank 0;
the model assumes routed-token pressure can reach 50% above the balanced case.

```bash
mkdir -p outputs
set -o pipefail

CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc-per-node=4 pretrain_gpt.py \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 32000 \
  --transformer-impl transformer_engine \
  --num-layers 8 \
  --hidden-size 512 \
  --ffn-hidden-size 2048 \
  --num-attention-heads 8 \
  --num-experts 8 \
  --moe-router-topk 2 \
  --moe-router-load-balancing-type aux_loss \
  --moe-aux-loss-coeff 1e-2 \
  --moe-grouped-gemm \
  --moe-token-dispatcher-type alltoall \
  --moe-permute-fusion \
  --disable-bias-linear \
  --swiglu \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --micro-batch-size 1 \
  --global-batch-size 16 \
  --train-iters 10 \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --lr-decay-iters 10 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --no-gradient-accumulation-fusion \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 4 \
  --expert-model-parallel-size 1 \
  --num-layers-per-virtual-pipeline-stage 1 \
  --pipeline-schedule-backend interleaved_1f1b \
  --bgl2-enable-memory-modeling \
  --bgl2-memory-max-vio 0.5 \
  --bgl2-patch-te \
  --bgl2-enable-activation-offload \
  --bgl2-activation-offload-ratio 1.0 \
  --bgl2-per-batch-offload-size 64 \
  --bgl2-activation-offload-stages 2 \
  --bgl2-export-moe-imbalance-ratio \
  --log-interval 1 \
  2>&1 | tee outputs/train.log
```

Because this command uses `--expert-model-parallel-size 1`, each expert-parallel
group has a single receiver rank and its reported `max/average` imbalance ratio
is expected to be exactly `1.0`. This validates telemetry emission and parsing,
not cross-rank imbalance. To exercise the cross-rank metric on four GPUs, use a
compatible layout with `--pipeline-model-parallel-size 2` and
`--expert-model-parallel-size 2`.

Adjust the model dimensions and parallelism degrees together. In particular,
`interleaved_1f1b` requires:

- `--pipeline-model-parallel-size` greater than 1
- virtual pipeline parallelism, normally configured with
  `--num-layers-per-virtual-pipeline-stage`
- model-layer and microbatch counts that satisfy Megatron's pipeline
  divisibility constraints

Use `--pipeline-schedule-backend vanilla` to retain Megatron's normal schedule.
When activation offload is enabled with the vanilla backend, BaGuaLu-2 still
selects its schedule-compatible runtime hooks.

## Activation-memory modeling

The analytical model estimates retained activation memory for one MoE
Transformer layer. It derives tensor shapes and dtypes from the active Megatron
arguments and transformer configuration, including:

- microbatch size, sequence length, tensor parallelism, and expert tensor
  parallelism
- MHA, GQA, and multi-latent attention projection layouts
- flash/fused or unfused attention
- expert count, router top-k, MoE hidden size, and SwiGLU expansion
- BF16, FP16, FP8 storage, router/index dtypes, and selective recomputation
- which retained tensors are eligible for the configured offload operators

Enable runtime reporting with:

```bash
--bgl2-enable-memory-modeling
```

With BaGuaLu-2's interleaved schedule, rank 0 then reports modeled memory split
between offload-eligible and non-eligible activations, the active offload ratio,
and CUDA memory observed during each pipeline phase.

`--bgl2-memory-max-vio RATIO` models routed MoE activations at `1 + RATIO`
times the balanced token count. This is a configured pressure assumption; the
runtime does not automatically replace it with values from the telemetry log.
`--bgl2-memory-flash-attention-kv-threshold N` controls the head-dimension
threshold used to choose flash rather than unfused attention when the Megatron
attention backend is `auto`.

## BaGuaLu-2 options

The most important runtime options are:

| Option | Purpose |
| --- | --- |
| `--pipeline-schedule-backend {vanilla,interleaved_1f1b}` | Select the Megatron or BaGuaLu-2 schedule. |
| `--bgl2-enable-activation-offload` | Enable scheduler-scoped activation offload and reload. |
| `--bgl2-enable-memory-modeling` | Enable the analytical activation model and rank-0 memory reports. |
| `--bgl2-memory-max-vio RATIO` | Model MoE routed-token pressure at `1 + RATIO` times the balanced count. |
| `--bgl2-patch-te` | Use the bundled Transformer Engine operator patches. Activation offload also enables these patches. |
| `--bgl2-offload-granularity {microbatch,layer}` | Offload after a full scheduler microbatch or immediately after each Transformer MLP layer. The default is `microbatch`. |
| `--bgl2-per-batch-offload-size MIB` | Set the fixed offload budget for each microbatch; `0` disables fixed-budget offload. |
| `--bgl2-activation-offload-ratio RATIO` | Enable ratio-based activation capture with a value in `[0, 1]`. The current runtime uses the value as a tensor-ordering hint; copied bytes are determined by `--bgl2-per-batch-offload-size`. |
| `--bgl2-activation-offload-threshold MIB` | Enable resident-memory accounting and dynamic ratio calculation against the threshold. This does not by itself copy bytes; configure a positive fixed offload budget. |
| `--bgl2-offload-min-bytes BYTES` | Ignore eligible tensors smaller than this size. The default is 1 MiB. |
| `--bgl2-offload-modules ...` | Choose patched operator classes to offload. The defaults are `GroupedLinear`, `LayerNormLinear`, `swiglu`, and `permutation`. |
| `--bgl2-activation-offload-stages N` | Partition each transfer into `N` internal issue groups. The current scheduler submits all groups together; this does not implement the paper's C3 fused dual-batch DP scheduler. |
| `--bgl2-prefetch-activation-reload` | Reload the next interleaved microbatch during current compute. This can improve overlap but keeps two restored activation groups live. |
| `--bgl2-offload-non-contiguous` | Opt in to offloading non-contiguous or view-backed tensors. Keep this disabled unless the fused kernels in use accept restored layouts. |
| `--bgl2-enable-pp-warmup-forward-backward` | Run a synthetic forward/backward before pipeline iteration 0. This is only valid with interleaved pipeline parallelism. |
| `--report-memory-every-iteration` | Keep Megatron's memory report enabled for every training iteration. |

All BaGuaLu-2 options are registered through the standard Megatron argument
parser, so `python pretrain_gpt.py --help` shows them alongside the upstream
options once the runtime dependencies are installed.

For detailed offload lifecycle logs, set `BGL2_OFFLOAD_DEBUG=1`. Restrict debug
output to selected global ranks with a comma- or space-separated list:

```bash
BGL2_OFFLOAD_DEBUG=1 BGL2_OFFLOAD_DEBUG_RANKS="0,4" torchrun ...
```

## MoE imbalance telemetry

Add the following option to a Megatron MoE run to emit per-layer imbalance
metrics and structured allocation/free events:

```bash
--bgl2-export-moe-imbalance-ratio
```

The event stream includes iteration, microbatch, layer, rank, received-token
count, and a lifecycle event ID. Capture the distributed training output, then
generate CSV and text reports:

```bash
python -m bgl2.data.moe_token_pressure outputs/train.log \
  --output-dir outputs/moe-token-pressure \
  --window-size 8 \
  --top 12
```

The analyzer reconstructs exact live-token pressure when matching `alloc` and
`free` events are present. It also supports the earlier
`step/layer/rank/received_tokens` log format and falls back to rolling-window
pressure estimates. Reports include per-rank peaks, lifecycle mismatches,
per-layer imbalance summaries, event windows, and optional per-rank time series.

The repository also includes
`scripts/workers/moe_imbalance_minimal.sh` as a multi-process cluster worker
template. It expects site-specific `scripts/lib/resolve_master_addr.sh`,
`scripts/lib/rank_log_prefix.sh`, and optionally `scripts/submit_task.sh`; those
launcher helpers are not included in this release. Use the self-contained
`torchrun` command above unless equivalent cluster helpers are available.

## Using BaGuaLu-2 from another Megatron entry point

Import the adaptor before importing Megatron's parser, training loop, or
pipeline schedule. Import order matters because BaGuaLu-2 patches
already-imported Megatron aliases as well as their defining modules.

```python
import bgl2.megatron_adaptor  # applies patches on import

from megatron.training import pretrain
```

The repository's `pretrain_gpt.py` also prepends the repository root,
`ext/Megatron-LM`, and `ext/TransformerEngine` to `PYTHONPATH`, so it can be run
directly from a checkout.

## Tests

Run the BaGuaLu-2 unit tests from the repository root:

```bash
python -m unittest discover -s bgl2/tests -v
```

Most tests use lightweight stand-ins for Megatron, PyTorch, and CUDA. The suite
also covers activation estimates for GQA, MLA, FP8 storage, recomputation, and
dynamic expert-token pressure. Tests that exercise real tensor offload require
PyTorch and are skipped when it is not available. A successful unit-test run
does not replace a multi-GPU smoke test for the target CUDA, NCCL, and
Transformer Engine environment.

## Paper evaluation context (not reproduced by this artifact)

The accompanying paper reports evaluation on fine-grained MoE models up to
1,040 billion parameters and 12,288 GPUs. The reported end-to-end throughput
improvements are 1.14x over Megatron-Kuai and 1.13x over AdaPipe. An ablation
against a static full-recomputation configuration reports a 1.38x improvement
from fine-grained mitigation, dynamic response, and overlap combined.

These figures describe the paper's cluster-scale evaluation; they are not
expected results for the mock-data smoke command above.

## Repository layout

```text
.
|-- bgl2/
|   |-- adapter/                  # Megatron and Transformer Engine patches
|   |-- core/models/gpt/          # analytical activation-memory model
|   |-- core/pipeline_parallel/   # schedules and activation offload runtime
|   |-- data/                     # MoE pressure analysis
|   `-- tests/
|-- ext/
|   |-- Megatron-LM/              # pinned submodule
|   `-- TransformerEngine/patch_te/
|-- scripts/workers/              # site-oriented training worker templates
|-- pretrain_gpt.py
`-- README.MD
```

## License

Unless otherwise noted, original BaGuaLu-2 code is licensed under the
[MIT License](LICENSE).

The modified Transformer Engine files under `ext/TransformerEngine/patch_te/`
are derived from NVIDIA Transformer Engine 2.12.0 and remain subject to the
[Apache License 2.0](ext/TransformerEngine/LICENSE). See the accompanying
[provenance and modification notice](ext/TransformerEngine/NOTICE.md).

The `ext/Megatron-LM` submodule retains its own copyright notices and license
terms; see [`ext/Megatron-LM/LICENSE`](ext/Megatron-LM/LICENSE).
