"""
Global settings and constants for the Maharashtra EV Policy Causal Evaluation.
Ensures reproducibility and centralizes configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Reproducibility Seeds
RANDOM_SEED = 42

# Policy Details
TREATMENT_DATE = "2025-05-23" # Policy announcement date
TREATMENT_MONTH = "2025-05" # Policy announcement month

# SCM Configuration — Maharashtra EV Policy 2025
# Pre-treatment: Jan 2022 – Apr 2025 (38 months of baseline)
# Post-treatment: May 2025 – present (policy divergence window)
PRE_TREATMENT_START = "2022-01"
PRE_TREATMENT_END = "2025-04"
POST_TREATMENT_START = "2025-05"
POST_TREATMENT_END = "2026-06"
RMSPE_TOLERANCE = 0.15

# Real Data API Endpoints (To be provided by the USER)
API_URL_VEHICLE_REGISTRATIONS = os.getenv("API_URL_VEHICLE_REGISTRATIONS", "USER_MUST_PROVIDE_API_URL")
API_URL_AIR_QUALITY = os.getenv("API_URL_AIR_QUALITY", "USER_MUST_PROVIDE_API_URL")
API_URL_ECONOMIC_SURVEY = os.getenv("API_URL_ECONOMIC_SURVEY", "USER_MUST_PROVIDE_API_URL")

# Memory Optimization
DUCKDB_MEMORY_LIMIT = "1GB"
DUCKDB_THREADS = 4
POLARS_MAX_THREADS = 4

def setup_directories():
    """Ensure all expected directories exist."""
    dirs = [
        RAW_DATA_DIR / "vehicle_registrations",
        RAW_DATA_DIR / "air_quality",
        RAW_DATA_DIR / "economic_survey",
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR / "scm_results",
        REPORTS_DIR / "figures",
        REPORTS_DIR / "tables",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    setup_directories()
