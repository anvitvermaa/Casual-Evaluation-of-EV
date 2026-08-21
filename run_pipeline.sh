#!/bin/bash
set -e  # Crash immediately if any command fails

echo "=========================================================="
echo "INITIALIZING SDiD PIPELINE AUTOMATION"
echo "=========================================================="

# 1. Initialize Conda — source it from its known absolute path
#    (handles cases where the shell is non-interactive and .bashrc is not sourced)
CONDA_BASE="/home/vityarthi/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. Create the Python environment if it does not exist
if ! conda info --envs | grep -q "ev-policy-sdid"; then
    echo "[SETUP] Creating ev-policy-sdid conda environment..."
    conda env create -f environment.yml
else
    echo "[SETUP] Environment 'ev-policy-sdid' already exists."
fi

# 3. Activate the isolated environment
echo "[SETUP] Activating environment: ev-policy-sdid"
conda activate ev-policy-sdid

# 4. Stage 1: Data Ingestion (Pandas CSV parser)
echo ""
echo "[STAGE 1] Parsing Vahan fuel CSVs..."
python src/data_ingestion/parse_vahan_data.py

# 5. Stage 2: DuckDB SQL Joins
echo ""
echo "[STAGE 2] Running DuckDB joins..."
python src/data_engineering/duckdb_joins.py

# 6. Stage 3: Polars Feature Engineering + Balance Assertion
echo ""
echo "[STAGE 3] Running Polars feature engineering + balance check..."
python src/features/polars_transform.py

# 7. Stage 4: SDiD Causal Modeling (dual specification)
echo ""
echo "[STAGE 4] Running Synthetic DiD estimator..."
python src/models/synthetic_control.py

# 8. Stage 5: Weight Validation EDA
echo ""
echo "[STAGE 5] Validating donor weights..."
python src/eda/validate_weights.py

echo ""
echo "=========================================================="
echo "PIPELINE COMPLETE"
echo "=========================================================="
