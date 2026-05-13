"""
Data Engineering Module: DuckDB SQL Joins
Loads raw Parquet/CSV files via DuckDB and performs out-of-core JOINs 
to create a unified analytical dataset without loading everything into memory.
"""

import os
import sys
import duckdb
from datetime import datetime

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def execute_joins():
    """Performs multi-source SQL joins using DuckDB."""
    print("Starting Phase 3: DuckDB Multi-Source JOINs...")
    
    con = duckdb.connect(database=':memory:')
    con.execute(f"PRAGMA memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={settings.DUCKDB_THREADS}")
    
    # Paths
    reg_path = os.path.join(settings.RAW_DATA_DIR, "vehicle_registrations", "maharashtra_registrations_2019_2026.parquet")
    aqi_path = os.path.join(settings.RAW_DATA_DIR, "air_quality", "maharashtra_aqi_2019_2026.csv")
    eco_path = os.path.join(settings.RAW_DATA_DIR, "economic_survey", "maharashtra_economic_survey_2018_2026.csv")
    output_path = os.path.join(settings.PROCESSED_DATA_DIR, "unified_dataset.parquet")
    
    if os.path.exists(output_path):
        print(f"Unified dataset already exists at {output_path}. Skipping.")
        return
        
    print("Aggregating Registrations to monthly-district level...")
    # 1. Aggregate Registrations
    # We want monthly EV counts and total counts per district
    reg_agg_query = f"""
    CREATE OR REPLACE TEMP VIEW v_registrations AS
    SELECT 
        district,
        date_trunc('month', registration_date) AS month,
        COUNT(*) AS total_registrations,
        SUM(CASE WHEN fuel_type = 'Electric' THEN 1 ELSE 0 END) AS ev_registrations
    FROM read_parquet('{reg_path}')
    GROUP BY district, date_trunc('month', registration_date)
    """
    con.execute(reg_agg_query)
    
    print("Aggregating AQI to monthly-district level...")
    # 2. Aggregate Air Quality
    # We want monthly mean PM2.5 and NOx per district
    aqi_agg_query = f"""
    CREATE OR REPLACE TEMP VIEW v_aqi AS
    SELECT 
        district,
        date_trunc('month', datetime) AS month,
        AVG(pm25) AS pm25_monthly_mean,
        AVG(nox) AS nox_monthly_mean
    FROM read_csv_auto('{aqi_path}')
    GROUP BY district, date_trunc('month', datetime)
    """
    con.execute(aqi_agg_query)
    
    print("Joining Datasets...")
    # 3. Join everything together
    # We join registrations and AQI on (district, month), then left join economic on (district, year)
    join_query = f"""
    CREATE OR REPLACE TEMP VIEW v_unified AS
    SELECT 
        r.district,
        r.month,
        EXTRACT(year FROM r.month) AS year,
        r.total_registrations,
        r.ev_registrations,
        a.pm25_monthly_mean,
        a.nox_monthly_mean,
        e.gsdp_per_capita,
        e.urban_population_pct,
        e.charging_station_density
    FROM v_registrations r
    INNER JOIN v_aqi a ON r.district = a.district AND r.month = a.month
    LEFT JOIN read_csv_auto('{eco_path}') e ON r.district = e.district AND EXTRACT(year FROM r.month) = e.year
    """
    con.execute(join_query)
    
    print(f"Exporting Unified Dataset to {output_path}...")
    export_query = f"""
    COPY (SELECT * FROM v_unified ORDER BY district, month) 
    TO '{output_path}' (FORMAT PARQUET)
    """
    con.execute(export_query)
    
    print("DuckDB JOINs completed successfully.")
    con.close()

if __name__ == "__main__":
    execute_joins()
