#!/bin/bash
#SBATCH --job-name=nect-stage3-dynamic
#SBATCH --partition=GPUQ
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=idun-07-06
#SBATCH --output=experiments/attenuation_validation/logs/dynamic-%j.out
#SBATCH --error=experiments/attenuation_validation/logs/dynamic-%j.err

set -euo pipefail
cd /cluster/work/jakobkm/NeCTv3
module purge
module load GCCcore/14.2.0
module load CUDA/12.8.0
export LD_LIBRARY_PATH="/cluster/apps/eb/software/GCCcore/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
source /cluster/work/jakobkm/NeCTv3/.venv/bin/activate

echo "Node: $(hostname)"; echo "Job: ${SLURM_JOB_ID}"; echo "Started: $(date --iso-8601=seconds)"
ldd nect/sampling/ct_sampling*.so | grep libstdc++
python - <<'PY'
import nect.sampling.ct_sampling
import porespy
import tigre
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda)
PY
python -u experiments/attenuation_validation/run_dynamic_experiment.py
echo "Finished: $(date --iso-8601=seconds)"
