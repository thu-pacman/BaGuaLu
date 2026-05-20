#!/usr/bin/env bash
set -x
set -euo pipefail

# Minimal MoE run for exporting dispatch imbalance metrics.
#
# Example:
#   WORKER_SCRIPT=scripts/workers/moe_imbalance_minimal.sh bash scripts/submit_task.sh
#
# Optional:
#   GPUS_PER_NODE=8 EXPERT_MODEL_PARALLEL_SIZE=8 WORKER_SCRIPT=scripts/workers/moe_imbalance_minimal.sh bash scripts/submit_task.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${REPO_DIR}/scripts/lib/resolve_master_addr.sh"
RANK_LOG_PREFIX="${RANK_LOG_PREFIX:-${REPO_DIR}/scripts/lib/rank_log_prefix.sh}"

NNODES="${NNODES:-${SLURM_NNODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-${SLURM_PROCID:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-$(resolve_master_addr)}"
MASTER_PORT="${MASTER_PORT:-29500}"

JOB_NAME="${JOB_NAME:-moe-imbalance-minimal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/${JOB_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
CONDA_ENV="${CONDA_ENV:-megatron}"

TRAIN_ITERS="${TRAIN_ITERS:-1000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"

NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-2048}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-${SEQ_LENGTH}}"
VOCAB_SIZE="${VOCAB_SIZE:-50304}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-${GPUS_PER_NODE}}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
NUM_EXPERTS="${NUM_EXPERTS:-${EXPERT_MODEL_PARALLEL_SIZE}}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"

mkdir -p "${TENSORBOARD_DIR}" "${LOG_DIR}"

# Keep environment setup opt-in so the script can run inside an already-active env.
set +u
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/megatron/lib/:\$LD_LIBRARY_PATH"
conda activate "${CONDA_ENV}"
set -u

cd "${REPO_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR MASTER_PORT

# submit_task.sh exports the caller environment; keep this worker free of traces.
unset CUDA_LAUNCH_BLOCKING
unset TORCH_SHOW_CPP_STACKTRACES
unset TORCH_NCCL_BLOCKING_WAIT
unset TORCH_NCCL_TRACE_BUFFER_SIZE
unset TORCH_NCCL_TRACE_CPP_STACK
unset TORCH_NCCL_DUMP_ON_TIMEOUT
unset TORCH_NCCL_ENABLE_TIMING
unset TORCH_NCCL_DEBUG_INFO_DIR

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

MODEL_ARGS=(
  --use-mcore-models
  --num-layers "${NUM_LAYERS}"
  --hidden-size "${HIDDEN_SIZE}"
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
  --num-attention-heads "${NUM_ATTENTION_HEADS}"
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --tokenizer-type NullTokenizer
  --vocab-size "${VOCAB_SIZE}"
  --attention-backend auto
  --swiglu
)

MOE_ARGS=(
  --num-experts "${NUM_EXPERTS}"
  --moe-router-topk "${MOE_ROUTER_TOPK}"
  --moe-router-load-balancing-type aux_loss
  --moe-aux-loss-coeff 1e-2
  --moe-grouped-gemm
  --moe-token-dispatcher-type alltoall
)

TRAINING_ARGS=(
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --lr 1.0e-4
  --min-lr 1.0e-5
  --lr-decay-style cosine
  --lr-decay-iters "${TRAIN_ITERS}"
  --weight-decay 0.1
  --clip-grad 1.0
  --init-method-std 0.02
  --bf16
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
)

DATA_ARGS=(
  --mock-data
  --split 949,50,1
  --num-workers 0
)

LOGGING_ARGS=(
  --log-interval 1
  --eval-interval 1000000
  --eval-iters 1
  --tensorboard-dir "${TENSORBOARD_DIR}"
)

BGL2_ARGS=(
  --export-moe-imbalance-ratio
)

echo "torchrun node_rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} gpus_per_node=${GPUS_PER_NODE}"

exec torchrun "${DISTRIBUTED_ARGS[@]}" --no_python "${RANK_LOG_PREFIX}" pretrain_gpt.py \
  "${MODEL_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "${BGL2_ARGS[@]}" \
  "$@"
