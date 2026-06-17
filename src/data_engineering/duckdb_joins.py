"""
Data Engineering Module: DuckDB SQL Joins
Loads raw Parquet/CSV files via DuckDB and performs out-of-core JOINs 
to create a unified analytical dataset.
"""

import os
import sys
import duckdb
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def execute_joins():
    print("Starting DuckDB JOINs for Real Data...")
    
    con = duckdb.connect(database=':memory:')
    con.execute(f"PRAGMA memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={settings.DUCKDB_THREADS}")
    
    # Paths
    reg_path = os.path.join(settings.RAW_DATA_DIR, "vehicle_registrations", "vahan_panel.parquet")
    aqi_path = os.path.join(settings.RAW_DATA_DIR, "air_quality", "openaq_pm25.parquet")
    eco_path = os.path.join(settings.RAW_DATA_DIR, "economic_survey", "maharashtra_economic_survey_2018_2026.csv")
    output_path = os.path.join(settings.PROCESSED_DATA_DIR, "unified_dataset.parquet")
    
    # RTOs to standard District Mapping
    # RTO names from Vahan: 'MUMBAI', 'PUNE', 'NAGPUR', 'NASHIK', etc.
    # AQI names from OpenAQ: 'MUMBAI', 'PUNE', etc.
    
    print("Joining Datasets...")
    join_query = f"""
    CREATE OR REPLACE TEMP VIEW v_unified AS
    SELECT 
        r.rto_name AS district,
        r.year_month || '-01' AS month, -- Convert '2022-01' to valid date string
        CAST(SUBSTRING(r.year_month, 1, 4) AS INT) AS year,
        r.total_registrations,
        r.ev_registrations,
        a.pm25_monthly_mean,
        e.gsdp_per_capita,
        e.urban_population_pct,
        e.charging_station_density
    FROM read_parquet('{reg_path}') r
    LEFT JOIN read_parquet('{aqi_path}') a 
        ON UPPER(r.rto_name) = UPPER(a.rto_name) AND r.year_month = a.year_month
    LEFT JOIN read_csv_auto('{eco_path}') e 
        ON UPPER(r.rto_name) = UPPER(e.district) AND CAST(SUBSTRING(r.year_month, 1, 4) AS INT) = e.year
    """
    con.execute(join_query)
    
    print(f"Exporting Unified Dataset to {output_path}...")
    export_query = f"""
    COPY (
        SELECT 
            district, 
            CAST(month AS DATE) as month, 
            year, 
            total_registrations, 
            ev_registrations, 
            pm25_monthly_mean, 
            gsdp_per_capita, 
            urban_population_pct, 
            charging_station_density 
        FROM v_unified 
        ORDER BY district, month
    ) TO '{output_path}' (FORMAT PARQUET)
    """
    con.execute(export_query)
    
    print("DuckDB JOINs completed successfully.")
    con.close()

if __name__ == "__main__":
    execute_joins()
