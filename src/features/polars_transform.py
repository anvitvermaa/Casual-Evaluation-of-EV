"""
Data Engineering Module: Polars Feature Engineering
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
    """Builds final feature matrix using Polars LazyFrames."""
    print("Starting Polars Feature Engineering Pipeline...")
    
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "unified_dataset.parquet")
    output_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_feature_matrix.parquet")
    
    if os.path.exists(output_path):
        print(f"Feature matrix already exists at {output_path}. Skipping.")
        return
        
    print("Loading data lazily...")
    # 1. Start Lazy Pipeline
    lf = pl.scan_parquet(input_path)
    
    # 2. Derive Penetration Rates & Time Indicators
    print("Computing derived features (penetration rate, rolling averages)...")
    lf = lf.with_columns([
        # EV Penetration Rate (%)
        (pl.col("ev_registrations") / pl.col("total_registrations") * 100).fill_nan(0.0).alias("ev_penetration_rate"),
        
        # Time identifiers for DiD and Event Study
        pl.col("month").dt.year().alias("year"),
        pl.col("month").dt.month().alias("month_num")
    ])
    
    # Sort to ensure rolling calculations are correct
    lf = lf.sort(["district", "month"])
    
    # 3. Window Functions (Rolling averages and deltas by district)
    lf = lf.with_columns([
        # Month-over-month delta in penetration
        (pl.col("ev_penetration_rate") - pl.col("ev_penetration_rate").shift(1)).over("district").alias("ev_penetration_delta"),
        
        # 3-Month rolling average (smoothing volatility)
        pl.col("ev_penetration_rate").rolling_mean(window_size=3, min_periods=1).over("district").alias("ev_pen_rolling_3m"),
        
        # Lagged PM2.5 (to control for past pollution driving current EV demand)
        pl.col("pm25_monthly_mean").shift(1).over("district").alias("pm25_lag_1m")
    ])
    
    # 4. Treatment Assignment
    print("Assigning treatment labels...")
    treatment_date = datetime.strptime(settings.TREATMENT_DATE, "%Y-%m-%d")
    treated_districts = ["MUMBAI", "PUNE", "THANE", "NASHIK", "NAGPUR", "AURANGABAD"]
    
    lf = lf.with_columns([
        # Dummy variable: 1 if district is in treatment group, 0 otherwise
        pl.col("district").is_in(treated_districts).cast(pl.Int8).alias("is_treated_district"),
        
        # Dummy variable: 1 if period is post-treatment, 0 otherwise
        (pl.col("month") >= treatment_date).cast(pl.Int8).alias("is_post_treatment"),
        
        # Interaction term (DiD Treat x Post)
        (
            pl.col("district").is_in(treated_districts) & (pl.col("month") >= treatment_date)
        ).cast(pl.Int8).alias("did_treat_post")
    ])
    
    # 5. Execute Pipeline and Save
    print(f"Collecting lazy frame and exporting to {output_path}...")
    # .collect() triggers the multi-threaded optimized execution
    df_final = lf.collect()
    
    df_final.write_parquet(output_path)
    
    # Final Sanity Check
    print(f"Feature engineering complete. Final shape: {df_final.shape}")
    print("Schema preview:")
    print(df_final.schema)

if __name__ == "__main__":
    engineer_features()
