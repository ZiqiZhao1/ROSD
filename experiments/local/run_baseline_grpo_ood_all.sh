#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
ulimit -c 0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

# ---------------------------------------------------------------------------
# Edit lists below only.
# TRAIN_DATASETS are traversed one by one; each run uses exactly one dataset.
# EVAL_TARGETS is taken literally: if you want biology evaluated, include it.
# ---------------------------------------------------------------------------
TRAIN_DATASETS=(
  "biology"
  "chemistry"
  "material"
  "physics"
  "tooluse"
)

EVAL_TARGETS=(
  "biology"
  "chemistry"
  "material"
  "physics"
  "tooluse"
  "aime24"
)

BACKBONES=(
  "Qwen3-4B"
  "Qwen3-8B"
)

PREPARE_DATASETS=true
CONFIG_NAME="baseline_grpo"
CKPT_DIR="$REPO_ROOT/checkpoints"
MODEL_ROOT="${MODEL_ROOT:-/path/to/your/models}"
N_GPUS_PER_NODE=8
TOTAL_EPOCHS=10
TEST_FREQ=10
PPO_MICRO_BATCH_SIZE_PER_GPU=8
# WANDB_PROJECT="SDPO-generalization"

target_dir_for() {
  case "$1" in
    biology) printf '%s' "datasets/sciknoweval-biology" ;;
    chemistry) printf '%s' "datasets/sciknoweval-chemistry" ;;
    material) printf '%s' "datasets/sciknoweval-material" ;;
    physics) printf '%s' "datasets/sciknoweval-physics" ;;
    tooluse) printf '%s' "datasets/tooluse" ;;
    aime24) printf '%s' "datasets/math-aime24" ;;
    *)
      echo "Unknown target: $1" >&2
      exit 1
      ;;
  esac
}

build_hydra_list() {
  local split="$1"
  shift
  local hydra="["
  local suffix

  if [[ "$split" == "train" ]]; then
    suffix="train.parquet"
  else
    suffix="test.parquet"
  fi

  for target in "$@"; do
    local rel_dir
    local file_path
    rel_dir="$(target_dir_for "$target")"
    file_path="$REPO_ROOT/$rel_dir/$suffix"

    if [[ ! -f "$file_path" ]]; then
      echo "Missing dataset artifact: $file_path" >&2
      echo "Run: python data/prepare_processreflection_ood.py --targets $target" >&2
      exit 1
    fi

    hydra="${hydra}'$file_path',"
  done

  hydra="${hydra%,}]"
  printf '%s' "$hydra"
}

join_by_dash() {
  local IFS='-'
  printf '%s' "$*"
}

if [[ ${#TRAIN_DATASETS[@]} -eq 0 ]]; then
  echo "TRAIN_DATASETS must not be empty" >&2
  exit 1
fi

if [[ ${#EVAL_TARGETS[@]} -eq 0 ]]; then
  echo "EVAL_TARGETS must not be empty" >&2
  exit 1
fi

if [[ "$PREPARE_DATASETS" == "true" ]]; then
  ALL_TARGETS=("${TRAIN_DATASETS[@]}" "${EVAL_TARGETS[@]}")
  HF_DATASETS_CACHE="$REPO_ROOT/.hf_cache" \
    python "$REPO_ROOT/data/prepare_processreflection_ood.py" --targets "$(IFS=,; echo "${ALL_TARGETS[*]}")"
fi

VAL_FILES="$(build_hydra_list eval "${EVAL_TARGETS[@]}")"
EVAL_TAG="$(join_by_dash "${EVAL_TARGETS[@]}")"

cd "$REPO_ROOT"
mkdir -p "$CKPT_DIR"

for TRAIN_TARGET in "${TRAIN_DATASETS[@]}"; do
  TRAIN_FILES="$(build_hydra_list train "$TRAIN_TARGET")"
  PRIMARY_TASK="$(target_dir_for "$TRAIN_TARGET")"
  TRAIN_TAG="$TRAIN_TARGET"
  RUN_TOTAL_EPOCHS="$TOTAL_EPOCHS"

  if [[ "$TRAIN_TARGET" == "tooluse" ]]; then
    RUN_TOTAL_EPOCHS=5
  fi

  for BACKBONE in "${BACKBONES[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${BACKBONE}"
    MODEL_TAG="$(echo "$BACKBONE" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
    EXPERIMENT="GRPO-OOD-train-${TRAIN_TAG}-${MODEL_TAG}"
    LOG_NAME="${DATE}-${EXPERIMENT}"
    OUTPUT_LOG_NAME="${LOG_NAME}-${TIME_TAG}"
    OUTPUT_DIR="$CKPT_DIR/${EXPERIMENT}-${TIME_TAG}"
    WANDB_PROJECT="SDPO-OOD-${TRAIN_TAG}"

    mkdir -p "$OUTPUT_DIR"

    echo "==============================================================="
    echo "Launching: ${EXPERIMENT}"
    echo "  TRAIN_TARGET: ${TRAIN_TARGET}"
    echo "  EVAL_TARGETS: ${EVAL_TARGETS[*]}"
    echo "  TRAIN_FILES: ${TRAIN_FILES}"
    echo "  VAL_FILES: ${VAL_FILES}"
    echo "  MODEL_PATH: ${MODEL_PATH}"
    echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
    echo "  TOTAL_EPOCHS: ${RUN_TOTAL_EPOCHS}"
    echo "  PPO_MICRO_BATCH_SIZE_PER_GPU: ${PPO_MICRO_BATCH_SIZE_PER_GPU}"

    bash training/verl_training.sh \
      "$EXPERIMENT" \
      "$CONFIG_NAME" \
      "$PRIMARY_TASK" \
      vars.dir="$REPO_ROOT" \
      vars.ckpt_dir="$CKPT_DIR" \
      custom_reward_function.path="$REPO_ROOT/verl/utils/reward_score/feedback/__init__.py" \
      trainer.logger="['console','swanlab']" \
      trainer.project_name="$WANDB_PROJECT" \
      trainer.experiment_name="$LOG_NAME" \
      trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
      trainer.nnodes=1 \
      trainer.save_freq=20000 \
      trainer.total_epochs="$RUN_TOTAL_EPOCHS" \
      trainer.test_freq="$TEST_FREQ" \
      trainer.val_before_train=True \
      trainer.max_actor_ckpt_to_keep=-1 \
      trainer.default_local_dir="$OUTPUT_DIR" \
      trainer.rollout_data_dir="$OUTPUT_DIR/rollout_traces" \
      trainer.group_name=SDPO-generalization \
      data.train_files="$TRAIN_FILES" \
      data.val_files="$VAL_FILES" \
      data.train_batch_size=32 \
      actor_rollout_ref.rollout.n=8 \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
      actor_rollout_ref.model.path="$MODEL_PATH" \
      algorithm.rollout_correction.rollout_is=token \
      actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
      actor_rollout_ref.rollout.val_kwargs.n=16 \
      2>&1 | tee ${OUTPUT_LOG_NAME}.log

    sleep 60
  done
done
