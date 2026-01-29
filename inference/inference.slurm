#!/bin/bash -l

#SBATCH -A naiss2026-4-49
#SBATCH  -p alvis

#SBATCH --gpus-per-node=A40:1
#SBATCH -t 5:00:00

# for gated repos
export HUGGING_FACE_HUB_TOKEN=""

# cache
export HF_HOME="/cephyr/users/ronih/Alvis/Desktop/mimer_llm-security-research/Roni"

set -euo pipefail
module purge

export API_PORT=3000

export HF_MODEL="checkpoint-36"
export BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export LORA_PATH="/cephyr/users/ronih/Alvis/Desktop/mimer_llm-security-research/Roni/output/sft_model/HackMentor_more_instruct3/checkpoint-36"

# apptainer build vllm.sif docker://vllm/vllm-openai:v0.7.3
export SIF_IMAGE="/cephyr/users/ronih/Alvis/Desktop/mimer_llm-security-research/Roni/vllm.sif"

# Build vLLM command safely
vllm_cmd=(
  vllm serve
  "${BASE_MODEL}"
  --port "${API_PORT}"
  --tensor-parallel-size="${SLURM_GPUS_ON_NODE}"
  --max-model-len=10000 # comment out for gemma
  --max-num-seqs=16
  --enforce-eager

  # comment out to run base moodel
  --enable-lora
  --lora-modules "${HF_MODEL}=${LORA_PATH}"
  --max-lora-rank 64 # varieras med lora r
  --served-model-name "roni-finetuned-model"
)

echo "Starting server with:"; printf '  %q ' "${vllm_cmd[@]}"; echo
apptainer exec --nv "${SIF_IMAGE}" "${vllm_cmd[@]}" > vllm.out 2> vllm.err & VLLM_PID=$!
trap 'kill -TERM "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null' EXIT TERM INT

echo "Waiting for server startup..."
deadline=$((SECONDS+120))
until grep -q "Application startup complete" vllm.out 2>/dev/null \
   || { command -v curl >/dev/null && curl -fs "http://127.0.0.1:${API_PORT}/v1/models" | grep -q '"data"'; }; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died."; tail -n +1 vllm.err vllm.out; exit 1; }
  (( SECONDS > deadline )) && { echo "Startup timeout."; tail -n 100 vllm.err vllm.out; exit 1; }
  sleep 2
done

echo "Server ready at http://$(hostname -f):${API_PORT}  (job ${SLURM_JOB_ID:-unknown})"
wait "$VLLM_PID"

