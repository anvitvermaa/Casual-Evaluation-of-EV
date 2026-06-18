"""
Data Engineering Module: Polars Feature Engineering (Macro-State Level)
Applies lazy-evaluated feature engineering (rolling means, penetration rates)
to prepare the final matrix for the Causal Machine Learning models.
"""

import os
import sys
import polars as pl
from datetime import datetime

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def engineer_features():
    """Builds final macro-state feature matrix using Polars LazyFrames."""
    print("Starting Polars Macro-State Feature Engineering Pipeline...")
    
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "unified_state_dataset.parquet")
    output_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_state_feature_matrix.parquet")
    
    if not os.path.exists(input_path):
        print(f"Input file {input_path} does not exist. Run duckdb_joins.py first.")
        return

    print("Loading state data lazily...")
    # 1. Start Lazy Pipeline
    lf = pl.scan_parquet(input_path)
    
    # 2. Derive Penetration Rates & Time Indicators
    print("Computing derived state-level features (penetration rate, rolling averages)...")
    lf = lf.with_columns([
        # EV Penetration Rate (%)
        (pl.col("ev_registrations") / pl.col("total_registrations") * 100).fill_nan(0.0).alias("ev_penetration_rate"),
        
        # Time identifiers for DiD and Event Study
        pl.col("month").dt.year().alias("year"),
        pl.col("month").dt.month().alias("month_num")
    ])
    
    # Sort to ensure rolling calculations are correct
    lf = lf.sort(["state", "month"])
    
    # 3. Window Functions (Rolling averages and deltas by state)
    lf = lf.with_columns([
        # Month-over-month delta in penetration
        (pl.col("ev_penetration_rate") - pl.col("ev_penetration_rate").shift(1)).over("state").alias("ev_penetration_delta"),
        
        # 3-Month rolling average (smoothing volatility)
        pl.col("ev_penetration_rate").rolling_mean(window_size=3).over("state").alias("ev_pen_rolling_3m"),
    ])
    
    # 4. Treatment Assignment (Macro-State)
    print("Assigning Macro-State treatment labels...")
    treatment_date = datetime.strptime(settings.TREATMENT_DATE, "%Y-%m-%d")
    treated_states = ["MAHARASHTRA"] # The only treated unit
    
    lf = lf.with_columns([
        # Dummy variable: 1 if state is MAHARASHTRA, 0 otherwise
        pl.col("state").is_in(treated_states).cast(pl.Int8).alias("is_treated_state"),
        
        # Dummy variable: 1 if period is post-treatment, 0 otherwise
        (pl.col("month") >= treatment_date).cast(pl.Int8).alias("is_post_treatment"),
        
        # Interaction term (DiD Treat x Post)
        (
            pl.col("state").is_in(treated_states) & (pl.col("month") >= treatment_date)
        ).cast(pl.Int8).alias("did_treat_post")
    ])
    
    # 5. Execute Pipeline and Save
    print(f"Collecting lazy frame and exporting to {output_path}...")
    # .collect() triggers the multi-threaded optimized execution
    df_final = lf.collect()
    
    # Ensure there are no nulls in target variables before writing
    df_final = df_final.drop_nulls(subset=["ev_penetration_rate"])
    
    df_final.write_parquet(output_path)
    
    # Final Sanity Check
    print(f"Feature engineering complete. Final shape: {df_final.shape}")
    print("Schema preview:")
    print(df_final.schema)

if __name__ == "__main__":
    engineer_features()
