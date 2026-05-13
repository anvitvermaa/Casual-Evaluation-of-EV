"""
Data Ingestion Module: Vehicle Registrations
Generates synthetic representative EV registration data for local pipeline testing.
Will be replaced by real API data later.
"""

import os
import sys
import polars as pl
from datetime import datetime, timedelta
import numpy as np

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def generate_mock_vehicle_data(output_path: str, n_records: int = 5_000_000):
    """Generates synthetic EV registration data for pipeline development."""
    print(f"Generating {n_records} mock vehicle registration records...")
    
    np.random.seed(settings.RANDOM_SEED)
    
    districts = [
        "Mumbai", "Mumbai Suburban", "Thane", "Pune", "Nashik", "Nagpur",
        "Chhatrapati Sambhajinagar", "Raigad", "Kolhapur", "Solapur", "Satara",
        "Sangli", "Jalgaon", "Ahmednagar", "Latur", "Beed"
    ]
    
    # Probabilities for different fuel types
    fuel_types = ["Petrol", "Diesel", "Electric", "CNG"]
    fuel_probs = [0.65, 0.20, 0.05, 0.10]
    
    # Generate dates between 2019-01-01 and 2026-05-31
    start_date = datetime.strptime(settings.PRE_TREATMENT_START, "%Y-%m")
    end_date = datetime.strptime(settings.POST_TREATMENT_END, "%Y-%m")
    days_diff = (end_date - start_date).days
    
    random_days = np.random.randint(0, days_diff, n_records)
    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    df = pl.DataFrame({
        "registration_id": [f"MH-{np.random.randint(1, 50):02d}-{np.random.randint(1000, 9999)}" for _ in range(n_records)],
        "registration_date": dates,
        "district": np.random.choice(districts, n_records),
        "fuel_type": np.random.choice(fuel_types, n_records, p=fuel_probs),
        "vehicle_category": np.random.choice(["2W", "3W", "4W", "Bus", "Commercial"], n_records, p=[0.7, 0.1, 0.15, 0.01, 0.04])
    })
    
    # Post-treatment EV surge in treated districts (Simulating the policy effect)
    treated_districts = ["Mumbai", "Mumbai Suburban", "Thane", "Pune", "Nashik", "Nagpur", "Chhatrapati Sambhajinagar", "Raigad"]
    treatment_date = datetime.strptime(settings.TREATMENT_DATE, "%Y-%m-%d")
    
    mask = (df["registration_date"] >= treatment_date) & (df["district"].is_in(treated_districts)) & pl.Series(np.random.random(n_records) < 0.15)
    
    df = df.with_columns(
        pl.when(mask).then(pl.lit("Electric")).otherwise(pl.col("fuel_type")).alias("fuel_type")
    )
    
    print(f"Writing dataset to {output_path} as Parquet...")
    df.write_parquet(output_path)
    print("Mock data generation complete.")

def fetch_data():
    """Main execution function."""
    output_dir = os.path.join(settings.RAW_DATA_DIR, "vehicle_registrations")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "maharashtra_registrations_2019_2026.parquet")
    
    if os.path.exists(output_file):
        print(f"Data already exists at {output_file}. Skipping ingestion.")
        return
        
    generate_mock_vehicle_data(output_file, n_records=5_000_000)

if __name__ == "__main__":
    fetch_data()
