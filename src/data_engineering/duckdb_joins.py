"""
Data Engineering Module: DuckDB SQL Joins (Macro-State Level)
Reads the unified state panel CSV and exports a Parquet file ready for Polars.
"""

import os
import sys
import duckdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def execute_joins():
    print("Starting DuckDB Macro-State JOINs...")

    con = duckdb.connect(database=':memory:')
    con.execute(f"PRAGMA memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={settings.DUCKDB_THREADS}")

    # Paths
    panel_csv   = os.path.join(settings.RAW_DATA_DIR, "vehicle_registrations", "vahan_state_panel.csv")
    output_path = os.path.join(settings.PROCESSED_DATA_DIR, "unified_state_dataset.parquet")

    if not os.path.exists(panel_csv):
        print(f"[ERROR] Panel CSV not found at {panel_csv}. Run parse_vahan_data.py first.")
        con.close()
        return

    print(f"Reading panel CSV: {panel_csv}")
    join_query = f"""
    CREATE OR REPLACE TEMP VIEW v_unified AS
    SELECT
        UPPER(state)                          AS state,
        CAST(year_month || '-01' AS DATE)     AS month,
        CAST(SUBSTRING(year_month, 1, 4) AS INT) AS year,
        CAST(ev_registrations    AS BIGINT)   AS ev_registrations,
        CAST(total_registrations AS BIGINT)   AS total_registrations
    FROM read_csv_auto('{panel_csv}', header=true)
    WHERE total_registrations > 0
    ORDER BY state, month
    """
    con.execute(join_query)

    row_count = con.execute("SELECT COUNT(*) FROM v_unified").fetchone()[0]
    print(f"  Rows in view: {row_count}")

    # Preview
    preview = con.execute("SELECT * FROM v_unified LIMIT 5").fetchdf()
    print(preview.to_string(index=False))

    print(f"\nExporting to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    con.execute(f"COPY (SELECT * FROM v_unified) TO '{output_path}' (FORMAT PARQUET)")

    print("DuckDB Macro-State JOINs completed successfully.")
    con.close()

if __name__ == "__main__":
    execute_joins()
