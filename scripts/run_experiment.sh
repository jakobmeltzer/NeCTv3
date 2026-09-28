#!/usr/bin/env bash

# ============================================================
# NeCT Experiment Runner
#
# Usage:
#   ./scripts/run_experiment.sh <experiment_name> <python_script>
#
# Example:
#   ./scripts/run_experiment.sh bentheimer_mixedcubes_8x demo/10_mixedcubes.py
#
# Creates:
#   experiment_logs/<name>_<timestamp>/
#       run.log
#       gpu.csv
#       environment.txt
#       metadata.txt
#       demo.py
# ============================================================

set -uo pipefail

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

if [ "$#" -ne 2 ]; then
    echo "Usage:"
    echo "  $0 <experiment_name> <python_script>"
    exit 1
fi

EXPERIMENT_NAME="$1"
PYTHON_SCRIPT="$2"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# ------------------------------------------------------------
# Experiment directory
# ------------------------------------------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="experiment_logs/${EXPERIMENT_NAME}_${TIMESTAMP}"

mkdir -p "$EXPERIMENT_DIR"

RUN_LOG="$EXPERIMENT_DIR/run.log"
GPU_LOG="$EXPERIMENT_DIR/gpu.csv"
ENV_LOG="$EXPERIMENT_DIR/environment.txt"
METADATA="$EXPERIMENT_DIR/metadata.txt"

# Save an exact copy of the experiment script.
cp "$PYTHON_SCRIPT" "$EXPERIMENT_DIR/demo.py"

echo "============================================================"
echo "NeCT experiment"
echo "============================================================"
echo "Name:      $EXPERIMENT_NAME"
echo "Script:    $PYTHON_SCRIPT"
echo "Directory: $EXPERIMENT_DIR"
echo "============================================================"

# ------------------------------------------------------------
# Record environment
# ------------------------------------------------------------

{
    echo "=== EXPERIMENT ==="
    echo "Name: $EXPERIMENT_NAME"
    echo "Script: $PYTHON_SCRIPT"
    echo

    echo "=== DATE ==="
    date
    echo

    echo "=== HOST ==="
    hostname
    echo

    echo "=== WORKING DIRECTORY ==="
    pwd
    echo

    echo "=== GIT COMMIT ==="
    git rev-parse HEAD 2>/dev/null || echo "Not in a Git repository"
    echo

    echo "=== GIT STATUS ==="
    git status --short 2>/dev/null || true
    echo

    echo "=== PYTHON ==="
    which python
    python --version
    echo

    echo "=== PYTORCH / CUDA ==="
    python - <<'PY'
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
except Exception as e:
    print("Could not retrieve PyTorch information:", e)
PY
    echo

    echo "=== MODULES ==="
    module list 2>&1 || true
    echo

    echo "=== NVIDIA-SMI ==="
    nvidia-smi || true

} > "$ENV_LOG"

# ------------------------------------------------------------
# Initial metadata
# ------------------------------------------------------------

START_TIME=$(date +%s)
START_DATE=$(date +"%Y-%m-%d %H:%M:%S")

{
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Script: $PYTHON_SCRIPT"
    echo "Started: $START_DATE"
    echo "Host: $(hostname)"
    echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "Experiment directory: $EXPERIMENT_DIR"
} > "$METADATA"

# ------------------------------------------------------------
# GPU monitoring
# ------------------------------------------------------------

echo
echo "Starting GPU monitoring..."

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    -l 10 \
    > "$GPU_LOG" &

GPU_MONITOR_PID=$!

echo "GPU monitor PID: $GPU_MONITOR_PID"

# Make sure the GPU logger is stopped if this wrapper is interrupted.
cleanup() {
    if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

# ------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------

echo
echo "Starting reconstruction..."
echo "Output is being written to:"
echo "  $RUN_LOG"
echo

# Disable immediate exit so we can record metadata even if Python fails.
set +e

python "$PYTHON_SCRIPT" 2>&1 | tee "$RUN_LOG"

# PIPESTATUS[0] is Python's exit code rather than tee's.
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

set -e

# ------------------------------------------------------------
# Stop GPU monitoring
# ------------------------------------------------------------

cleanup
trap - EXIT INT TERM

# ------------------------------------------------------------
# Final metadata
# ------------------------------------------------------------

END_TIME=$(date +%s)
END_DATE=$(date +"%Y-%m-%d %H:%M:%S")

RUNTIME_SECONDS=$((END_TIME - START_TIME))

RUNTIME_HOURS=$((RUNTIME_SECONDS / 3600))
RUNTIME_MINUTES=$(((RUNTIME_SECONDS % 3600) / 60))
RUNTIME_SECS=$((RUNTIME_SECONDS % 60))

{
    echo "Finished: $END_DATE"
    echo "Runtime seconds: $RUNTIME_SECONDS"
    printf "Runtime: %02d:%02d:%02d\n" \
        "$RUNTIME_HOURS" \
        "$RUNTIME_MINUTES" \
        "$RUNTIME_SECS"
    echo "Exit status: $PYTHON_EXIT_CODE"
} >> "$METADATA"

echo
echo "============================================================"
echo "Experiment finished"
echo "============================================================"
echo "Exit status: $PYTHON_EXIT_CODE"
printf "Runtime: %02d:%02d:%02d\n" \
    "$RUNTIME_HOURS" \
    "$RUNTIME_MINUTES" \
    "$RUNTIME_SECS"
echo
echo "Results saved in:"
echo "  $EXPERIMENT_DIR"
echo
echo "Files:"
echo "  run.log          NeCT console output"
echo "  gpu.csv          GPU statistics every 10 seconds"
echo "  environment.txt  Software/hardware environment"
echo "  metadata.txt     Experiment metadata and runtime"
echo "  demo.py          Exact experiment script"
echo "============================================================"

exit "$PYTHON_EXIT_CODE"
