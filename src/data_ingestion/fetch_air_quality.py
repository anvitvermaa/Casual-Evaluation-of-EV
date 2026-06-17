import os
import requests
import polars as pl
from pathlib import Path
from datetime import datetime
import time
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

API_KEY = os.getenv("OPENAQ_API_KEY", "5788e6181e31d6fd79b9046ab7d00f7dc87205d383f8a1b1cd0a898d21546878")
HEADERS = {'X-API-Key': API_KEY}
BASE_URL = "https://api.openaq.org/v3"

# Major RTO centers mapped to their coordinates for bounding box search (25km radius)
CITIES = {
    'Mumbai': '19.0760,72.8777',
    'Pune': '18.5204,73.8567',
    'Nagpur': '21.1458,79.0882',
    'Nashik': '19.9975,73.7898',
    'Thane': '19.2183,72.9781',
    'Aurangabad': '19.8762,75.3433',
    'Solapur': '17.6599,75.9064',
    'Kolhapur': '16.7050,74.2433'
}

def get_locations(coords):
    url = f"{BASE_URL}/locations"
    params = {'coordinates': coords, 'radius': 25000, 'parameter': 'pm25', 'limit': 100}
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if r.status_code == 200:
        return r.json().get('results', [])
    return []

def get_pm25_sensor_id(location_id):
    url = f"{BASE_URL}/locations/{location_id}/sensors"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code == 200:
        for s in r.json().get('results', []):
            if s['parameter']['name'].lower() == 'pm25':
                return s['id']
    return None

def fetch_daily_data(sensor_id):
    all_data = []
    page = 1
    while True:
        url = f"{BASE_URL}/sensors/{sensor_id}/days"
        params = {'limit': 1000, 'page': page}
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        
        if r.status_code == 200:
            results = r.json().get('results', [])
            if not results: break
            all_data.extend(results)
            page += 1
            if len(results) < 1000: break # Last page
        elif r.status_code == 429:
            print("Rate limit hit, sleeping 5s...")
            time.sleep(5)
        else:
            break
            
    return all_data

def main():
    print("Fetching OpenAQ Data (Real PM2.5)...")
    records = []
    
    for city, coords in CITIES.items():
        print(f"-> Processing {city}...")
        locations = get_locations(coords)
        
        for loc in locations:
            loc_id = loc['id']
            sensor_id = get_pm25_sensor_id(loc_id)
            
            if not sensor_id: continue
                
            daily_data = fetch_daily_data(sensor_id)
            for d in daily_data:
                dt_str = d['period']['datetimeFrom']['utc']
                # Parse '2025-01-01T00:00:00Z' to '2025-01'
                year_month = dt_str[:7]
                
                records.append({
                    "rto_name": city.upper(), # Map back to RTO style
                    "station_id": str(loc_id),
                    "year_month": year_month,
                    "pm25_daily_avg": d.get('summary', {}).get('avg', d.get('value', 0))
                })
                
        time.sleep(1) # Prevent rate limiting

    if not records:
        print("Error: No data fetched from OpenAQ.")
        return
        
    # Aggregate daily into monthly RTO-level averages
    df = pl.DataFrame(records)
    
    # Filter 2022-2026
    df = df.filter((pl.col("year_month") >= "2022-01") & (pl.col("year_month") <= "2026-06"))
    
    monthly_df = df.group_by(["rto_name", "year_month"]).agg(
        pl.col("pm25_daily_avg").mean().alias("pm25_monthly_mean")
    ).sort(["rto_name", "year_month"])
    
    out_path = settings.RAW_DATA_DIR / "air_quality" / "openaq_pm25.parquet"
    monthly_df.write_parquet(out_path)
    print(f"Successfully processed {len(monthly_df)} monthly records -> {out_path}")

if __name__ == "__main__":
    main()
