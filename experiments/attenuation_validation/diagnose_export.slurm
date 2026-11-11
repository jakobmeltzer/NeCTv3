#!/bin/bash
#SBATCH --job-name=nect-export-check
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=idun-07-06
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cluster/work/jakobkm/NeCTv3
#SBATCH --output=/cluster/work/jakobkm/NeCTv3/experiment_logs/export_%j.out

set -euo pipefail

module load GCCcore/14.2.0
module load CUDA/12.8.0

export LD_LIBRARY_PATH="/cluster/apps/eb/software/GCCcore/14.2.0/lib64:${LD_LIBRARY_PATH:-}"

source /cluster/work/jakobkm/NeCTv3/.venv/bin/activate

# These checks run on the allocated GPU node, not in your terminal
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
python -c "import nect.sampling.ct_sampling; print('NeCT import successful')"

# Run the diagnostic without retraining
python -u experiments/attenuation_validation/run_experiment.py --stage diagnose-export