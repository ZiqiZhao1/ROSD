# ROSD

Reflective On-Policy Self-Distillation for language model reasoning across domains.

This repository contains the code used for the ROSD out-of-domain reasoning experiments. It is built on top of [`verl`](https://github.com/verl-project/verl) and [SDPO](https://github.com/lasgroup/SDPO), and extends on-policy self-distillation with:

- self-reflection over on-policy rollouts;
- error-quote localization for failed rollouts;
- token-level self-distillation only from the localized error span;
- in-domain and out-of-domain evaluation across science, tool-use, and math targets.

The main experiment entry point used for the reported ROSD runs is:

```bash
experiments/local/run_sdpo_processreflection_ood_all.sh
```

## Repository Layout

```text
data/                                      Dataset loading and preprocessing
datasets/                                  Small JSON splits and generated parquet targets
experiments/local/                         Local launch script for the main ROSD runs
training/verl_training.sh                  Wrapper around verl.trainer.main_ppo
verl/trainer/config/sdpo.yaml              ROSD/SDPO Hydra config override
verl/trainer/ppo/ray_trainer.py            Rollout, reward, reflection, and distillation batch construction
verl/workers/config/actor.py               Self-distillation and reflection config fields
verl/workers/actor/dp_actor.py             Actor update and self-distillation loss dispatch
verl/trainer/ppo/core_algos.py             RL and self-distillation objectives
verl/utils/reward_score/feedback/          Task reward functions
```

## What Is Included

This release includes the source code, main experiment launcher, reward functions, and dataset preparation scripts needed to reproduce the ROSD runs.

Model weights, checkpoints, local caches, and training logs are not included. The launch scripts expect you to provide local Hugging Face model paths.

## Environment

ROSD training requires a Linux GPU environment with CUDA, Ray, vLLM/SGLang, and PyTorch. The experiments were designed for multi-GPU nodes.

A typical setup is:

```bash
conda create -n rosd python=3.10 -y
conda activate rosd

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements_sglang.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

## Data Preparation

The ROSD OOD scripts use flat dataset directories with `train.parquet` and `test.parquet` files. Prepare all targets with:

```bash
python data/prepare_processreflection_ood.py \
  --targets biology,chemistry,material,physics,tooluse,aime24
```

This creates or refreshes:

```text
datasets/sciknoweval-biology
datasets/sciknoweval-chemistry
datasets/sciknoweval-material
datasets/sciknoweval-physics
datasets/tooluse
datasets/math-aime24
```

The main launch script sets `PREPARE_DATASETS=true`, so it will run this preparation step automatically before training. If your machine cannot access Hugging Face, prepare the JSON/parquet files manually or set `PREPARE_DATASETS=false` after placing the files in the expected directories.

## Reproducing The Main ROSD Runs

Before launching, edit only the configuration block near the top of:

```bash
experiments/local/run_sdpo_processreflection_ood_all.sh
```

Important fields:

```bash
TRAIN_DATASETS=(biology chemistry material physics tooluse)
EVAL_TARGETS=(biology chemistry material physics tooluse aime24)
BACKBONES=("Qwen3-4B Qwen3-8B")
N_GPUS_PER_NODE=8
TOTAL_EPOCHS=10
TEST_FREQ=10
PPO_MICRO_BATCH_SIZE_PER_GPU=2
```

Set `MODEL_ROOT` to the directory that contains the backbone model folders:

```bash
export MODEL_ROOT=/path/to/your/models
```

Then run:

```bash
bash experiments/local/run_sdpo_processreflection_ood_all.sh
```

By default, each training dataset is run independently and evaluated on all listed targets. Outputs are written under:

```text
checkpoints/SDPO-reflection-OOD-train-<train_target>-<model_tag>-<time_tag>/
```

Validation artifacts are stored in each run's `validation_data` directory.

For quick smoke tests, reduce `TRAIN_DATASETS`, `EVAL_TARGETS`, `BACKBONES`, `TOTAL_EPOCHS`, and `trainer.logger` in the launch script. If you do not use SwanLab, change:

```bash
trainer.logger="['console','swanlab']"
```

to:

```bash
trainer.logger="['console']"
```

## Method-Specific Config

ROSD is enabled through the SDPO loss mode plus reflection-specific self-distillation settings. The main script passes the following key overrides:

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=sdpo
actor_rollout_ref.actor.self_distillation.enable_reflection=True
actor_rollout_ref.actor.self_distillation.use_error_quote_mask=True
actor_rollout_ref.actor.self_distillation.use_reflection_in_teacher_prompt=True
```

The core reflection templates and self-distillation options are defined in:

```text
verl/workers/config/actor.py
```

The training loop builds reflection prompts, aligns `<error_quote>` spans back to rollout tokens, and constructs masked distillation batches in:

```text
verl/trainer/ppo/ray_trainer.py
```

## Notes

- The codebase is a research fork of `verl`; many upstream examples and tests are retained for compatibility.
- The launch scripts contain machine-specific defaults for model locations and logging. Update these paths before running.
- The provided OOD launcher is the canonical script for the main ROSD result path.
- Checkpoints and logs are intentionally excluded from this release.

## License

This project inherits the Apache-2.0 license from the underlying `verl` codebase. See `LICENSE`.

## Acknowledgements

This implementation builds on `verl` and the SDPO codebase. Please cite the corresponding upstream projects when using this code.

## Citation

If you find this project useful, please cite our paper:

```bibtex
@misc{zhao2026rosd,
      title={ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains}, 
      author={Ziqi Zhao and Xinyu Ma and Liu Yang and Yujie Feng and Daiting Shi and Jingzhou He and Xin Xin and Zhaochun Ren and Xiao-Ming Wu},
      year={2026},
      eprint={2605.28014},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.28014}, 
}
```
