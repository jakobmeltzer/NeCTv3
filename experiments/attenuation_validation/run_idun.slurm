
#!/bin/bash
#SBATCH --job-name=nect-cube-phantom
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

echo "========================================"
echo "NeCT Stage 1: Attenuation validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================"

# Load the CUDA toolkit used by NeCT
module load CUDA/12.8.0

# Activate the existing NeCTv3 environment
source /cluster/work/jakobkm/NeCTv3/.venv/bin/activate

# Display GPU information
nvidia-smi

# Verify that CUDA actually works
python - <<'PY'
import torch

print("PyTorch CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA initialization failed"

print("GPU:", torch.cuda.get_device_name(0))
print("Test tensor:", torch.ones(3, device="cuda"))
PY

echo "Starting attenuation-validation experiment..."

# Execute the entire experiment
python -u experiments/attenuation_validation/run_experiment.py

echo "========================================"
echo "Experiment completed successfully!"
echo "End: $(date)"
echo "========================================"
