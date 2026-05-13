"""
Data Ingestion Module: Air Quality Data
Generates synthetic Hourly PM2.5 and NOx data for local pipeline testing.
Will be replaced by real CPCB API data later per user instruction.
"""

import os
import sys
import polars as pl
from datetime import datetime, timedelta
import numpy as np

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def generate_mock_air_quality_data(output_path: str):
    """Generates synthetic hourly AQI data."""
    print("Generating mock hourly air quality records...")
    
    np.random.seed(settings.RANDOM_SEED + 1)
    
    districts = [
        "Mumbai", "Mumbai Suburban", "Thane", "Pune", "Nashik", "Nagpur",
        "Chhatrapati Sambhajinagar", "Raigad", "Kolhapur", "Solapur", "Satara",
        "Sangli", "Jalgaon", "Ahmednagar", "Latur", "Beed"
    ]
    
    start_date = datetime.strptime(settings.PRE_TREATMENT_START, "%Y-%m")
    end_date = datetime.strptime(settings.POST_TREATMENT_END, "%Y-%m")
    
    date_range = pl.datetime_range(
        start=start_date,
        end=end_date,
        interval="1h",
        eager=True
    )
    
    df_dates = pl.DataFrame({"datetime": date_range})
    df_districts = pl.DataFrame({"district": districts})
    
    df = df_dates.join(df_districts, how="cross")
    n_records = len(df)
    print(f"Base grid created with {n_records} hourly records.")
    
    base_pm25 = np.random.normal(50, 15, n_records)
    months = df["datetime"].dt.month()
    winter_multiplier = pl.when(months.is_in([11, 12, 1, 2])).then(1.5).otherwise(1.0)
    
    df = df.with_columns([
        (pl.lit(base_pm25) * winter_multiplier).alias("pm25_base")
    ])
    
    treated_districts = ["Mumbai", "Mumbai Suburban", "Thane", "Pune", "Nashik", "Nagpur", "Chhatrapati Sambhajinagar", "Raigad"]
    treatment_date = datetime.strptime(settings.TREATMENT_DATE, "%Y-%m-%d")
    
    mask = (df["datetime"] >= treatment_date) & (df["district"].is_in(treated_districts))
    
    df = df.with_columns([
        pl.when(mask).then(pl.col("pm25_base") * 0.95).otherwise(pl.col("pm25_base")).alias("pm25"),
        (pl.col("pm25_base") * 0.6 + np.random.normal(10, 5, n_records)).alias("nox")
    ]).drop("pm25_base")
    
    df = df.with_columns([
        (pl.lit("MH_") + pl.col("district").str.slice(0, 3).str.to_uppercase() + pl.lit("_01")).alias("station_id")
    ])
    
    print(f"Writing dataset to {output_path} as CSV...")
    df.write_csv(output_path)
    print("Mock air quality data generation complete.")

def fetch_data():
    """Main execution function."""
    output_dir = os.path.join(settings.RAW_DATA_DIR, "air_quality")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "maharashtra_aqi_2019_2026.csv")
    
    if os.path.exists(output_file):
        print(f"Data already exists at {output_file}. Skipping ingestion.")
        return
        
    generate_mock_air_quality_data(output_file)

if __name__ == "__main__":
    fetch_data()
