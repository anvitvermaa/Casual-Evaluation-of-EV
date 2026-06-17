"""
Data Engineering Module: DuckDB SQL Joins
Loads raw State-Level Parquet/CSV files via DuckDB and performs out-of-core JOINs 
to create a unified analytical dataset for Macro-State Spatial SCM.
"""

import os
import sys
import duckdb
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def execute_joins():
    print("Starting DuckDB JOINs for Macro-State Data...")
    
    con = duckdb.connect(database=':memory:')
    con.execute(f"PRAGMA memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={settings.DUCKDB_THREADS}")
    
    # State-level Paths
    # We will read all state CSVs dynamically from the states directory
    reg_path = os.path.join(settings.RAW_DATA_DIR, "vehicle_registrations", "states", "*.csv")
    aqi_path = os.path.join(settings.RAW_DATA_DIR, "air_quality", "state_openaq_pm25.parquet")
    eco_path = os.path.join(settings.RAW_DATA_DIR, "economic_survey", "state_economic_survey.csv")
    output_path = os.path.join(settings.PROCESSED_DATA_DIR, "unified_state_dataset.parquet")
    
    print("Joining State Datasets...")
    # NOTE: The join condition uses 'state' as the primary cross-sectional key
    join_query = f"""
    CREATE OR REPLACE TEMP VIEW v_unified AS
    SELECT 
        UPPER(r.state) AS state,
        r.year_month || '-01' AS month, -- Convert '2022-01' to valid date string
        CAST(SUBSTRING(r.year_month, 1, 4) AS INT) AS year,
        r.total_registrations,
        r.ev_registrations,
        a.pm25_monthly_mean,
        e.gsdp_per_capita,
        e.urban_population_pct
    FROM read_csv_auto('{reg_path}') r
    LEFT JOIN read_parquet('{aqi_path}') a 
        ON UPPER(r.state) = UPPER(a.state) AND r.year_month = a.year_month
    LEFT JOIN read_csv_auto('{eco_path}') e 
        ON UPPER(r.state) = UPPER(e.state) AND CAST(SUBSTRING(r.year_month, 1, 4) AS INT) = e.year
    """
    
    try:
        con.execute(join_query)
    except Exception as e:
        print(f"Error during JOIN: {e}")
        print("Note: Ensure that the state CSVs exist in data/raw/vehicle_registrations/states/ before running.")
        con.close()
        return
    
    print(f"Exporting Unified Dataset to {output_path}...")
    export_query = f"""
    COPY (
        SELECT 
            state, 
            CAST(month AS DATE) as month, 
            year, 
            total_registrations, 
            ev_registrations, 
            pm25_monthly_mean, 
            gsdp_per_capita, 
            urban_population_pct
        FROM v_unified 
        ORDER BY state, month
    ) TO '{output_path}' (FORMAT PARQUET)
    """
    con.execute(export_query)
    
    print("DuckDB State JOINs completed successfully.")
    con.close()

if __name__ == "__main__":
    execute_joins()
