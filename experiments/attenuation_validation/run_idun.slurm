
#!/bin/bash
#SBATCH --job-name=nect-stage1-cube
#SBATCH --partition=GPUQ
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --exclude=idun-07-06
#SBATCH --chdir=/cluster/work/jakobkm/NeCTv3
#SBATCH --output=/cluster/work/jakobkm/NeCTv3/experiment_logs/phantom_%j.out
#SBATCH --error=/cluster/work/jakobkm/NeCTv3/experiment_logs/phantom_%j.err

set -euo pipefail

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

# Load the compiler runtime needed by NeCT's CUDA extensions
module load GCCcore/14.2.0
module load CUDA/12.8.0

# Make sure the correct C++ runtime is found first
export LD_LIBRARY_PATH="/cluster/apps/eb/software/GCCcore/14.2.0/lib64:${LD_LIBRARY_PATH:-}"

# Activate NeCTv3
source /cluster/work/jakobkm/NeCTv3/.venv/bin/activate

# Verify the environment
echo "C++ runtime selected:"
ldd nect/sampling/ct_sampling*.so | grep libstdc++

echo "Testing NeCT imports..."
python -c "import nect.sampling.ct_sampling; print('NeCT import OK')"

echo "Testing CUDA..."
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print(torch.ones(3, device="cuda"))
PY

# Run the full experiment
python -u experiments/attenuation_validation/run_experiment.py

echo "Finished: $(date)"
