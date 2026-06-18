#!/bin/bash
set -e  # Crash immediately if any command fails

echo "=========================================================="
echo "🚀 INITIALIZING SDiD PIPELINE AUTOMATION"
echo "=========================================================="

# 1. Initialize Conda for the shell environment
# Assumes conda is installed in a standard location or accessible via `conda info`
source $(conda info --base)/etc/profile.d/conda.sh

# 2. Create the Python 3.10 environment (if it doesn't exist)
if ! conda info --envs | grep -q "ev-policy-sdid"; then
    echo "[SETUP] Creating strictly isolated Python 3.10 Conda environment..."
    conda env create -f environment.yml
else
    echo "[SETUP] Environment 'ev-policy-sdid' already exists. Updating..."
    conda env update -f environment.yml --prune
fi

# 3. Activate the isolated environment
echo "[SETUP] Activating environment: ev-policy-sdid"
conda activate ev-policy-sdid

# 4. Sequentially execute the pipeline
echo "[EXECUTION] 1. Data Ingestion & Engineering..."
# Depending on the architecture, if duckdb_joins handles ingestion or if we just run engineer
make engineer

echo "[EXECUTION] 2. Modeling (SDiD)..."
python src/models/synthetic_control.py

echo "[EXECUTION] 3. Weight Validation..."
python src/eda/validate_weights.py

echo "=========================================================="
echo "✅ PIPELINE EXECUTED FLAWLESSLY"
echo "=========================================================="
