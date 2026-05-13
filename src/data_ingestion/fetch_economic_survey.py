"""
Data Ingestion Module: Economic Survey
Generates synthetic district-wise GSDP and demographic data for local pipeline testing.
Will be replaced by real API data later per user instruction.
"""

import os
import sys
import polars as pl
import numpy as np

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def generate_mock_economic_data(output_path: str):
    """Generates synthetic annual economic survey data."""
    print("Generating mock economic survey records...")
    
    np.random.seed(settings.RANDOM_SEED + 2)
    
    districts = [
        "Mumbai", "Mumbai Suburban", "Thane", "Pune", "Nashik", "Nagpur",
        "Chhatrapati Sambhajinagar", "Raigad", "Kolhapur", "Solapur", "Satara",
        "Sangli", "Jalgaon", "Ahmednagar", "Latur", "Beed"
    ]
    
    years = list(range(2018, 2027))
    
    df_districts = pl.DataFrame({"district": districts})
    df_years = pl.DataFrame({"year": years})
    
    df = df_districts.join(df_years, how="cross")
    n_records = len(df)
    
    base_gsdp = {
        "Mumbai": 5.0, "Mumbai Suburban": 4.5, "Pune": 4.0, "Thane": 3.5, 
        "Nagpur": 2.8, "Nashik": 2.5, "Chhatrapati Sambhajinagar": 2.0
    }
    
    df = df.with_columns([
        pl.col("district").replace_strict(base_gsdp, default=1.5).alias("base_gsdp")
    ])
    
    df = df.with_columns([
        (pl.col("base_gsdp") * (1.06 ** (pl.col("year") - 2018))).alias("gsdp_per_capita")
    ])
    
    df = df.with_columns([
        (pl.col("base_gsdp") * 15 + np.random.normal(10, 5, n_records)).clip(20, 100).alias("urban_population_pct"),
        (pl.col("base_gsdp") * 10).alias("charging_station_density")
    ]).drop("base_gsdp")
    
    print(f"Writing dataset to {output_path} as CSV...")
    df.write_csv(output_path)
    print("Mock economic data generation complete.")

def fetch_data():
    """Main execution function."""
    output_dir = os.path.join(settings.RAW_DATA_DIR, "economic_survey")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "maharashtra_economic_survey_2018_2026.csv")
    
    if os.path.exists(output_file):
        print(f"Data already exists at {output_file}. Skipping ingestion.")
        return
        
    generate_mock_economic_data(output_file)

if __name__ == "__main__":
    fetch_data()
