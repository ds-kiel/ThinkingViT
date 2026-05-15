#!/bin/bash -l
#SBATCH --job-name=ThinkingViT-Swin-train
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256000
#SBATCH --time=7-00:00:00
#SBATCH --output=swin_train.out
#SBATCH --error=swin_train.err
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:4

set -euo pipefail

if ! command -v module &> /dev/null; then
    set +u
    source /etc/profile || true
    set -u
fi

for MODULE_INIT in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if ! command -v module &> /dev/null && [ -f "${MODULE_INIT}" ]; then
        set +u
        source "${MODULE_INIT}" || true
        set -u
    fi
done

if command -v module &> /dev/null; then
    module load gpu-env
    module load cuda
    module load gcc12-env
    module load python
else
    echo "Warning: environment module command not found; continuing with the current environment."
fi

source /home/aho/envs/env/bin/activate

cd /home/aho/ThinkingViTCVPR/ThinkingViT

NUM_GPUS=4
CONFIG="args_swin.yaml"
DATA_DIR="/data22/datasets/ilsvrc2012/"
MODEL="swin_small_patch4_window7_224"
HEAD_ROUND_1=(3 3 6 12)
HEAD_ROUND_2=(3 6 12 24)
LOG_FILE="ThinkingViTSwin_train.log"

torchrun --nproc_per_node="${NUM_GPUS}" train_swin.py \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --pretrained \
    --data-dir "${DATA_DIR}" \
    --head-round-1 "${HEAD_ROUND_1[@]}" \
    --head-round-2 "${HEAD_ROUND_2[@]}" \
    &> "${LOG_FILE}"

if command -v jobinfo &> /dev/null; then
    jobinfo
fi
