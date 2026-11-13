#!/bin/bash
#SBATCH --job-name=nect-coordinates
#SBATCH --partition=GPUQ
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=experiments/attenuation_validation/logs/coordinates-%j.out
#SBATCH --error=experiments/attenuation_validation/logs/coordinates-%j.err

set -euo pipefail

cd /cluster/work/jakobkm/NeCTv3
module purge
module load GCCcore/14.2.0
module load CUDA/12.8.0
export LD_LIBRARY_PATH="/cluster/apps/eb/software/GCCcore/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
source /cluster/work/jakobkm/NeCTv3/.venv/bin/activate

python - <<'PY'
import nect.sampling.ct_sampling
import tigre
import torch
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
PY

# Ray-convention test only: four small forward projections, no optimization.
python -u experiments/attenuation_validation/coordinate_diagnostic.py

# Reuse the completed Stage-2 checkpoint; this never calls reconstruction.
python -u experiments/attenuation_validation/run_porous_experiment.py \
  --stage reevaluate-coordinates
