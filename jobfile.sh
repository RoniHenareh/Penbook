#!/bin/bash

#SBATCH -A naiss2026-4-49
#SBATCH  -p alvis

#SBATCH --gpus-per-node=A100:1
#SBATCH --time=72:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ronih@kth.se

module purge
module load Python/3.11.5-GCCcore-13.2.0

set -euxo pipefail

# Activate environment (uncomment if needed)
source /cephyr/users/ronih/Alvis/Desktop/.env/bin/activate

# Hugging Face caches
export HF_HOME="/cephyr/users/ronih/Alvis/Desktop/mimer_llm-security-research/Roni"
export TRANSFORMERS_CACHE="$HF_HOME/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$HF_HOME/.cache/huggingface/datasets"

cd /cephyr/users/ronih/Alvis/Desktop/sft/train/training
#mkdir -p logs

# Run training
srun python -u train.py "/cephyr/users/ronih/Alvis/Desktop/mimer_llm-security-research/Roni/output/sft_model/Primus-CPT"

