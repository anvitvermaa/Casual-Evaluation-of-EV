"""
Data Ingestion Module: Vahan State-Level Fuel Panel Parser
Reads all state × year FUEL CSV files (Fuel vs Month Wise cross-tabulation)
from data/raw/vehicle_registrations/states/,
extracts monthly EV and total registrations per fuel type, and outputs a
unified Parquet file ready for Polars Feature Engineering.

File naming convention expected: <State Name>_<year>_fuel_master.csv
e.g.: Maharashtra_2024_fuel_master.csv, Tamil Nadu_2022_fuel_master.csv

CSV Schema (from RevTpark/vahan-scraper api.py, Fuel vs Month Wise):
  S No | Fuel | JAN | FEB | MAR | ... | DEC | TOTAL
  Rows = Fuel types (ELECTRIC(BOV), PETROL, DIESEL, CNG ONLY, etc.)
  Values = monthly registration counts (may contain commas: "1,23,456")
"""

import os
import sys
import glob
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# All Vahan electric fuel row labels to capture EV registrations
EV_FUEL_LABELS = {"ELECTRIC(BOV)", "ELECTRIC", "STRONG HYBRID EV"}


def parse_fuel_csv(filepath: str, state_name: str, year: int) -> list[dict]:
    """
    Parse a single Vahan Fuel vs Month Wise CSV file.
    Returns a list of per-month dicts with ev_registrations and total_registrations.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    # Normalise column names
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Identify the fuel label column (S NO | FUEL | months...)
    # Find which column contains fuel labels
    fuel_col = None
    for c in df.columns:
        if c in ("FUEL", "FUEL TYPE"):
            fuel_col = c
            break
    if fuel_col is None:
        # Fallback: second column is usually fuel when S NO is first
        non_month_cols = [c for c in df.columns if c not in MONTH_COLS and c != "TOTAL" and c != "S NO"]
        if non_month_cols:
            fuel_col = non_month_cols[0]
        else:
            print(f"  [WARNING] Cannot identify fuel column in {filepath}. Skipping.")
            return []

    # Keep only month columns that are actually present
    present_month_cols = [m for m in MONTH_COLS if m in df.columns]
    if not present_month_cols:
        print(f"  [WARNING] No month columns in {filepath}. Skipping.")
        return []

    # Normalise fuel labels
    df[fuel_col] = df[fuel_col].astype(str).str.strip().str.upper()

    # Clean all numeric values (commas like "1,23,456" → int)
    for col in present_month_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    ev_by_month    = {m: 0 for m in present_month_cols}
    total_by_month = {m: 0 for m in present_month_cols}

    for _, row in df.iterrows():
        fuel_label = str(row[fuel_col]).strip().upper()
        if not fuel_label or fuel_label in ("NAN", "FUEL", "S NO", "TOTAL"):
            continue

        for month in present_month_cols:
            val = int(row[month])
            total_by_month[month] += val
            if fuel_label in EV_FUEL_LABELS:
                ev_by_month[month] += val

    # Build one record per month
    records = []
    for month in present_month_cols:
        if total_by_month[month] == 0:
            continue  # Skip months with zero data (e.g., future months in 2026)

        month_num = MONTH_COLS.index(month) + 1
        records.append({
            "state":               state_name.upper(),
            "year_month":          f"{year}-{month_num:02d}",
            "ev_registrations":    ev_by_month[month],
            "total_registrations": total_by_month[month],
        })

    return records


def parse_all_fuel_files():
    """Main entry point: parse all *_fuel_master.csv files and write unified Parquet."""
    states_dir  = os.path.join(settings.RAW_DATA_DIR, "vehicle_registrations", "states")
    output_path = os.path.join(settings.RAW_DATA_DIR, "vehicle_registrations", "vahan_fuel_panel.parquet")

    all_files = sorted(glob.glob(os.path.join(states_dir, "*_fuel_master.csv")))
    if not all_files:
        print(f"[ERROR] No *_fuel_master.csv files found in {states_dir}")
        return

    print(f"Found {len(all_files)} fuel CSV files. Parsing...\n")

    all_records = []
    for filepath in all_files:
        filename = os.path.basename(filepath).replace("_fuel_master.csv", "")
        # Split on last underscore to get state + year
        parts = filename.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  [SKIP] Unrecognised filename: {filename}")
            continue
        state_name, year_str = parts[0], parts[1]
        try:
            year = int(year_str)
        except ValueError:
            print(f"  [SKIP] Cannot parse year from '{year_str}' in {filename}")
            continue

        print(f"  Parsing {os.path.basename(filepath)}  →  state={state_name}, year={year}")
        records = parse_fuel_csv(filepath, state_name, year)
        all_records.extend(records)
        print(f"    → {len(records)} monthly rows extracted.")

    if not all_records:
        print("[ERROR] No records extracted. Check file formats and paths.")
        return

    df = pd.DataFrame(all_records)
    df["ev_penetration_rate"] = (
        df["ev_registrations"] / df["total_registrations"] * 100
    ).fillna(0.0)
    df = df.sort_values(["state", "year_month"]).reset_index(drop=True)

    df.to_parquet(output_path, index=False)
    print(f"\n✅ Fuel Panel Parquet written → {output_path}")
    print(f"   Total rows : {len(df)}")
    print(f"   States     : {sorted(df['state'].unique())}")
    print(f"   Date range : {df['year_month'].min()} → {df['year_month'].max()}")
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    parse_all_fuel_files()
