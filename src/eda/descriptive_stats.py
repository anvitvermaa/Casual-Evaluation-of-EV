"""
EDA Module: Descriptive Statistics
Computes pre-treatment summary statistics for treated vs. control districts.
Outputs Table 1 for the final paper.
"""

import os
import sys
import polars as pl
from datetime import datetime

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def generate_descriptive_stats():
    """Calculates summary statistics and outputs a markdown table."""
    print("Starting Descriptive Statistics Generation...")
    
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_feature_matrix.parquet")
    output_path = os.path.join(settings.REPORTS_DIR, "tables", "table1_descriptive_stats.md")
    
    if not os.path.exists(input_path):
        print("Feature matrix not found. Run Phase 3 first.")
        sys.exit(1)
        
    df = pl.read_parquet(input_path)
    
    # Filter for Pre-Treatment period only (to establish baseline balance)
    treatment_date = datetime.strptime(settings.TREATMENT_DATE, "%Y-%m-%d")
    df_pre = df.filter(pl.col("month") < treatment_date)
    
    # Aggregate to district level (average over pre-treatment period)
    df_dist = df_pre.group_by(["district", "is_treated_district"]).agg([
        pl.col("ev_penetration_rate").mean().alias("mean_ev_penetration"),
        pl.col("pm25_monthly_mean").mean().alias("mean_pm25"),
        pl.col("nox_monthly_mean").mean().alias("mean_nox"),
        pl.col("gsdp_per_capita").mean().alias("mean_gsdp_per_capita"),
        pl.col("urban_population_pct").mean().alias("mean_urban_pct"),
        pl.col("charging_station_density").mean().alias("mean_charging_density")
    ])
    
    # Calculate group means and standard deviations
    summary = df_dist.group_by("is_treated_district").agg([
        pl.col("mean_ev_penetration").mean().round(2).alias("ev_pen_mean"),
        pl.col("mean_ev_penetration").std().round(2).alias("ev_pen_std"),
        pl.col("mean_pm25").mean().round(1).alias("pm25_mean"),
        pl.col("mean_pm25").std().round(1).alias("pm25_std"),
        pl.col("mean_gsdp_per_capita").mean().round(2).alias("gsdp_mean"),
        pl.col("mean_gsdp_per_capita").std().round(2).alias("gsdp_std"),
        pl.col("mean_urban_pct").mean().round(1).alias("urban_mean"),
        pl.col("mean_urban_pct").std().round(1).alias("urban_std")
    ]).sort("is_treated_district", descending=True)
    
    # Format into a Markdown table
    with open(output_path, "w") as f:
        f.write("### Table 1: Pre-Treatment Descriptive Statistics (2019-2025)\n\n")
        f.write("| Variable | Treated Districts Mean (SD) | Control Districts Mean (SD) |\n")
        f.write("|----------|-----------------------------|-----------------------------|\n")
        
        # Extract rows
        treated = summary.filter(pl.col("is_treated_district") == 1).row(0)
        control = summary.filter(pl.col("is_treated_district") == 0).row(0)
        
        variables = [
            ("EV Penetration Rate (%)", 1, 2),
            ("PM2.5 Concentration (μg/m³)", 3, 4),
            ("GSDP per Capita (₹ Lakh)", 5, 6),
            ("Urban Population (%)", 7, 8)
        ]
        
        for name, mean_idx, std_idx in variables:
            t_str = f"{treated[mean_idx]} ({treated[std_idx]})"
            c_str = f"{control[mean_idx]} ({control[std_idx]})"
            f.write(f"| {name} | {t_str} | {c_str} |\n")
            
        f.write("\n*Note: Standard deviations in parentheses. Pre-treatment window spans Jan 2019 to May 2025.*\n")
        
    print(f"Table 1 generated successfully at {output_path}")

if __name__ == "__main__":
    generate_descriptive_stats()
