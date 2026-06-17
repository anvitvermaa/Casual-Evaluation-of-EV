import os
import openpyxl
import polars as pl
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

def clean_number(x):
    if x is None: return 0
    s = str(x).replace(',', '').strip()
    return int(s) if s.isdigit() else 0

def extract_row_data(ws, search_col_idx, search_val, month_cols):
    """Finds a specific row based on a column value and extracts the 12 month columns."""
    for row in ws.iter_rows(min_row=5, values_only=True):
        if row[search_col_idx] and search_val.lower() in str(row[search_col_idx]).lower():
            return {month: clean_number(row[idx]) for month, idx in month_cols.items()}
    return {month: 0 for month in month_cols.keys()}

def get_month_indices(ws):
    for r_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=5, values_only=True), 1):
        if 'JAN' in [str(x).strip() for x in row if x]:
            months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            month_idx = {}
            for idx, val in enumerate(row):
                if val and str(val).strip() in months:
                    month_idx[str(val).strip()] = idx
            return month_idx
    return None

def main():
    print("Parsing Vahan Data...")
    data_dir = Path("vahan data")
    
    # 1. Calculate Baseline EV Distribution (2024)
    baseline_path = data_dir / "rto_fuel_baseline_2024.xlsx"
    wb = openpyxl.load_workbook(baseline_path, read_only=True)
    ws = wb.active
    
    # Find Electric column
    header_row = list(ws.iter_rows(min_row=4, max_row=4, values_only=True))[0]
    ev_col_idx = None
    for i, val in enumerate(header_row):
        if val and 'ELECTRIC(BOV)' in str(val):
            ev_col_idx = i
            break
            
    rto_ev_shares = {}
    total_evs = 0
    for row in ws.iter_rows(min_row=5, values_only=True):
        rto_name = str(row[1]).strip()
        if not rto_name or rto_name == 'None': continue
        ev_count = clean_number(row[ev_col_idx])
        rto_ev_shares[rto_name] = ev_count
        total_evs += ev_count
        
    # Convert to percentages
    for rto in rto_ev_shares:
        rto_ev_shares[rto] = rto_ev_shares[rto] / total_evs if total_evs > 0 else 0
        
    print(f"Loaded EV baseline for {len(rto_ev_shares)} RTOs.")

    # 2. Load State EV totals per month from Fuel files
    state_ev_monthly = {} # Format: { '2022-01': 1500, ... }
    month_map = {'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06', 
                 'JUL':'07', 'AUG':'08', 'SEP':'09', 'OCT':'10', 'NOV':'11', 'DEC':'12'}
                 
    for year in ['22', '23', '24', '25', '26']:
        fuel_file = data_dir / f"Fuel/fuel_{year}.xlsx"
        if not fuel_file.exists(): continue
        
        wb = openpyxl.load_workbook(fuel_file, read_only=True)
        ws = wb.active
        m_idx = get_month_indices(ws)
        ev_data = extract_row_data(ws, 1, 'ELECTRIC(BOV)', m_idx)
        
        for m_name, val in ev_data.items():
            state_ev_monthly[f"20{year}-{month_map[m_name]}"] = val
            
    print(f"Loaded State EV monthly data for {len(state_ev_monthly)} months.")

    # 3. Load RTO Totals per month and construct final dataset
    final_records = []
    
    for year in ['22', '23', '24', '25', '26']:
        rto_file = data_dir / f"RTO/rto_{year}.xlsx"
        if not rto_file.exists(): continue
        
        wb = openpyxl.load_workbook(rto_file, read_only=True)
        ws = wb.active
        m_idx = get_month_indices(ws)
        
        for row in ws.iter_rows(min_row=5, values_only=True):
            rto_name = str(row[1]).strip()
            if not rto_name or rto_name == 'None': continue
            
            for m_name, idx in m_idx.items():
                year_month = f"20{year}-{month_map[m_name]}"
                total_registrations = clean_number(row[idx])
                
                # Apply downscaling logic
                state_ev = state_ev_monthly.get(year_month, 0)
                rto_share = rto_ev_shares.get(rto_name, 0)
                estimated_evs = int(state_ev * rto_share)
                
                ev_penetration = (estimated_evs / total_registrations * 100) if total_registrations > 0 else 0
                
                final_records.append({
                    "rto_name": rto_name,
                    "year_month": year_month,
                    "total_registrations": total_registrations,
                    "ev_registrations": estimated_evs,
                    "ev_penetration_rate": ev_penetration
                })

    df = pl.DataFrame(final_records)
    # Filter to standard timeline
    df = df.filter((pl.col("year_month") >= "2022-01") & (pl.col("year_month") <= "2026-06"))
    
    out_path = settings.RAW_DATA_DIR / "vehicle_registrations" / "vahan_panel.parquet"
    df.write_parquet(out_path)
    print(f"Successfully processed {len(df)} records -> {out_path}")

if __name__ == "__main__":
    main()
