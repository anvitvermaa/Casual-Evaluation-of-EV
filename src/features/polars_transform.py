"""
Data Engineering Module: Polars Feature Engineering — Aggregate Fuel Panel
Reads the unified fuel panel Parquet produced by parse_vahan_data.py,
applies lazy-evaluation transformations, assigns treatment indicators,
and writes the final SDiD-ready feature matrix.

Output: data/processed/final_state_feature_matrix_main.parquet
"""

import os
import sys
import polars as pl
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

TREATED_STATES = ["MAHARASHTRA"]


def engineer_features():
    """Build the aggregate macro-state feature matrix using Polars LazyFrames."""
    print("=" * 60)
    print("Polars Feature Engineering Pipeline (Aggregate Fuel Panel)")
    print("=" * 60)

    input_path  = os.path.join(
        settings.RAW_DATA_DIR, "vehicle_registrations", "vahan_fuel_panel.parquet"
    )
    output_path = os.path.join(
        settings.PROCESSED_DATA_DIR, "final_state_feature_matrix_main.parquet"
    )

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        print("Run src/data_ingestion/parse_vahan_data.py first.")
        return

    print(f"Loading fuel panel: {input_path}")
    lf = pl.scan_parquet(input_path)

    # ── 1. Compute EV penetration rate & time identifiers ────────────────────
    lf = lf.with_columns([
        # Penetration rate (primary outcome variable for SDiD)
        (pl.col("ev_registrations") / pl.col("total_registrations") * 100)
          .fill_nan(0.0)
          .alias("outcome"),

        # Year and month numeric decomposition for diagnostics
        pl.col("year_month").str.slice(0, 4).cast(pl.Int32).alias("year"),
        pl.col("year_month").str.slice(5, 2).cast(pl.Int32).alias("month_num"),
    ])

    # ── 2. Sort for window functions ─────────────────────────────────────────
    lf = lf.sort(["state", "year_month"])

    # ── 3. Rolling 3-month average (smoothing) ───────────────────────────────
    lf = lf.with_columns([
        pl.col("outcome")
          .rolling_mean(window_size=3)
          .over("state")
          .alias("outcome_rolling_3m"),

        pl.col("ev_registrations")
          .rolling_mean(window_size=3)
          .over("state")
          .alias("ev_regs_rolling_3m"),
    ])

    # ── 4. Treatment assignment ───────────────────────────────────────────────
    treatment_ym = (
        datetime.strptime(settings.TREATMENT_DATE, "%Y-%m-%d")
        .strftime("%Y-%m")
    )

    lf = lf.with_columns([
        pl.col("state").is_in(TREATED_STATES)
          .cast(pl.Int8).alias("is_treated_state"),

        (pl.col("year_month") >= treatment_ym)
          .cast(pl.Int8).alias("is_post_treatment"),
    ])

    lf = lf.with_columns([
        (pl.col("is_treated_state") & pl.col("is_post_treatment"))
          .cast(pl.Int8).alias("did_treat_post"),
    ])

    # ── 5. Execute lazy pipeline ──────────────────────────────────────────────
    print("Collecting lazy frame...")
    df = lf.collect()
    df = df.drop_nulls(subset=["outcome"])

    # ── 6. Strict Balance Assertion ───────────────────────────────────────────
    # SDiD requires a PERFECTLY balanced panel. If any state is missing even a
    # single month, the matrix dimensions will not align and the estimator will
    # either crash or silently produce garbage results. We check this explicitly
    # and FAIL LOUDLY rather than allowing a corrupted run to continue.
    n_states   = df["state"].n_unique()
    n_months   = df["year_month"].n_unique()
    n_treated  = df.filter(pl.col("did_treat_post") == 1).height
    total_rows = df.height

    print(f"\n=== BALANCE ASSERTION ===")
    print(f"   Unique states    : {n_states}")
    print(f"   Unique months    : {n_months}")
    print(f"   Total rows       : {total_rows}  (should equal {n_states} × {n_months} = {n_states * n_months})")

    months_per_state = df.group_by("state").len().sort("state")
    print(f"\n   Months per state (must all match {n_months}):")
    for row in months_per_state.iter_rows(named=True):
        flag = "OK" if row["len"] == n_months else "*** UNBALANCED ***"
        print(f"     {row['state']:<25}  {row['len']:>3}  {flag}")

    unbalanced = months_per_state.filter(pl.col("len") != n_months)
    if unbalanced.height > 0:
        raise AssertionError(
            f"[FATAL] Panel is NOT balanced. "
            f"{unbalanced.height} state(s) have a different number of months.\n"
            f"The SDiD matrix would be corrupt. Fix the raw data before proceeding.\n"
            f"Unbalanced states:\n{unbalanced}"
        )

    if total_rows != n_states * n_months:
        raise AssertionError(
            f"[FATAL] Row count mismatch: got {total_rows}, expected {n_states * n_months}. "
            f"There are duplicate state-month combinations in the panel."
        )

    print(f"\n[BALANCE CHECK PASSED] {n_states} states × {n_months} months = {total_rows} rows.")
    print(f"   States           : {sorted(df['state'].unique().to_list())}")
    print(f"   Treatment obs    : {n_treated}")
    print(f"   Outcome range    : {df['outcome'].min():.4f} – {df['outcome'].max():.4f} %")

    missing = df.filter(pl.col("outcome").is_null()).height
    if missing > 0:
        print(f"   [WARNING] {missing} null outcome rows found — check ev/total ratio calculation.")

    # ── 7. Write output ───────────────────────────────────────────────────────
    os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
    df.write_parquet(output_path)
    print(f"\n[OUTPUT] Saved → {output_path}")

    print("\nSchema preview:")
    print(df.schema)
    print("\nSample rows:")
    print(df.head(8))


if __name__ == "__main__":
    engineer_features()
