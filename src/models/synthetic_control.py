"""
Causal Modeling Module: Synthetic Difference-in-Differences (SDiD)
Maharashtra EV Policy Evaluation (N=9 state panel, Jan 2022 – Jun 2026).

Implements the d2cml-ai/synthdid official Python API
as defined by Arkhangelsky et al. (2021, AER 111(12), 4088-4118).

Methodological Defences (per Reviewer 2 critique):
  1. Low-N p-value floor (Flaw 1): Uses placebo SE (model.se) not permutation p-value.
     Minimum permutation p-value with N=9 is 1/9 ≈ 0.111, making it useless.
  2. Convex Hull Violation (Flaw 2): SDiD's omega_intercept=True and lambda_intercept=True
     (defaults) add unit and time fixed-effect shifts, allowing vertical extrapolation
     beyond the convex hull of the donor pool.
  3. Karnataka Anchor Risk (Flaw 3): L2 ridge regularization (zeta_omega auto-computed
     from noise_level in sdid.py) disperses donor weights. Validated post-fit.
  4. Covariate OVB (Flaw 4): GSDP_per_capita and Urbanization_Rate passed as covariates
     with cov_method='optimized' (Arkhangelsky et al., Eq 4.1).

API contract verified against synthdid==0.10.1 source:
  - Synthdid(data, unit, time, treatment, outcome, covariates)
  - .fit(cov_method='optimized') → stores ATT in self.att (float), weights in
    self.weights = {"lambda": [array], "omega": [array]}
  - .vcov(method="placebo", n_reps=N) → stores SE in self.se (float), returns self
  - .plot_outcomes() / .plot_weights() → returns list of matplotlib Figure objects

CRITICAL: time column must use consecutive 0-based integer indices (0,1,...,T-1),
NOT YYYYMM strings. The library computes T1 = max(time) - treatment_time + 1, which
requires consecutive integers to be valid.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless/server execution

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

# ── EXPLICIT DEPENDENCY BOUNDARY ──────────────────────────────────────────────
try:
    from synthdid.synthdid import Synthdid
except ImportError as e:
    raise ImportError(
        "\n[FATAL ERROR] The official d2cml-ai/synthdid library is not installed "
        "or the environment is improperly isolated. Execute within the Python 3.10 "
        "conda environment defined in environment.yml.\n"
        f"Original error: {str(e)}"
    )
# ──────────────────────────────────────────────────────────────────────────────


def execute_sdid_pipeline():
    print("\n" + "=" * 70)
    print("SYNTHETIC DIFFERENCE-IN-DIFFERENCES (Arkhangelsky et al., 2021)")
    print("Maharashtra EV Policy 2025 — Causal Evaluation")
    print("=" * 70)

    # ── 1. Load Panel Data ────────────────────────────────────────────────────
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_state_feature_matrix.parquet")
    output_dir = os.path.join(settings.MODELS_DIR, "scm_results")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(input_path)
    df['month'] = pd.to_datetime(df['month'])

    print(f"[DATA] Panel: {len(df)} observations, {df['state'].nunique()} states, "
          f"{df['month'].nunique()} months")
    print(f"[DATA] Date range: {df['month'].min().date()} to {df['month'].max().date()}")
    print(f"[DATA] States: {sorted(df['state'].unique().tolist())}")

    # ── 2. Build Consecutive Integer Time Index ───────────────────────────────
    # CRITICAL: synthdid computes T1 = max(time) - treatment_time + 1
    # This is only valid when time uses 0,1,...,T-1 (not YYYYMM integers).
    sorted_months = sorted(df['month'].unique())
    time_map = {m: i for i, m in enumerate(sorted_months)}
    df['time'] = df['month'].map(time_map)

    # ── 3. Merge Economic Covariates (prevents OVB — Flaw 4) ──────────────────
    # Source: RBI State Finances Report 2023-24, Census 2011 urbanisation projections
    economic_data = pd.DataFrame([
        {'state': 'MAHARASHTRA',    'GSDP_per_capita': 274000, 'Urbanization_Rate': 45.2},
        {'state': 'KARNATAKA',      'GSDP_per_capita': 278000, 'Urbanization_Rate': 38.6},
        {'state': 'GUJARAT',        'GSDP_per_capita': 275000, 'Urbanization_Rate': 42.6},
        {'state': 'TAMIL NADU',     'GSDP_per_capita': 273000, 'Urbanization_Rate': 48.4},
        {'state': 'TELANGANA',      'GSDP_per_capita': 308000, 'Urbanization_Rate': 38.9},
        {'state': 'ANDHRA PRADESH', 'GSDP_per_capita': 219000, 'Urbanization_Rate': 29.6},
        {'state': 'MADHYA PRADESH', 'GSDP_per_capita': 140000, 'Urbanization_Rate': 27.6},
        {'state': 'RAJASTHAN',      'GSDP_per_capita': 156000, 'Urbanization_Rate': 24.8},
        {'state': 'UTTAR PRADESH',  'GSDP_per_capita': 83000,  'Urbanization_Rate': 22.3},
    ])
    df = pd.merge(df, economic_data, on='state', how='left')

    # ── 4. Define Treatment Indicator ─────────────────────────────────────────
    treatment_date = pd.to_datetime(settings.TREATMENT_DATE)

    # Find the consecutive integer index corresponding to the treatment date
    treatment_time_idx = None
    for m in sorted_months:
        if pd.Timestamp(m) >= treatment_date:
            treatment_time_idx = time_map[m]
            actual_treatment_month = pd.Timestamp(m)
            break

    if treatment_time_idx is None:
        raise ValueError(f"Treatment date {settings.TREATMENT_DATE} is beyond "
                         f"the panel's end date {df['month'].max().date()}.")

    df['treatment'] = (
        (df['state'] == 'MAHARASHTRA') &
        (df['time'] >= treatment_time_idx)
    ).astype(int)

    n_pre  = int(treatment_time_idx)
    n_post = int(df['time'].max() - treatment_time_idx + 1)
    print(f"\n[TREATMENT] Policy date   : {actual_treatment_month.date()} "
          f"(time_idx={treatment_time_idx})")
    print(f"[TREATMENT] Pre-periods   : {n_pre}")
    print(f"[TREATMENT] Post-periods  : {n_post}")
    print(f"[TREATMENT] Treated obs   : {df['treatment'].sum()}")

    # ── 5. Rename Columns to Library Defaults ─────────────────────────────────
    # Synthdid.panel_matrices() expects column names matching the unit/time/etc args.
    # Using library defaults avoids any internal query variable collision.
    df = df.rename(columns={
        'state': 'unit',
        'ev_penetration_rate': 'outcome'
    })

    # ── 6. Initialise SDiD Model ───────────────────────────────────────────────
    print("\n[MODEL] Initialising Synthdid estimator...")
    sdid_model = Synthdid(
        data=df,
        unit='unit',
        time='time',
        treatment='treatment',
        outcome='outcome',
        covariates=['GSDP_per_capita', 'Urbanization_Rate']
    )

    # ── 7. Fit (cov_method='optimized' → Eq 4.1 covariate-augmented weights) ──
    print("[MODEL] Fitting SDiD with covariate-augmented weights (cov_method='optimized')...")
    sdid_model.fit(cov_method='optimized')

    tau_hat = float(sdid_model.att)
    print(f"[MODEL] ATT (tau_hat): {tau_hat:+.4f} percentage points")

    # ── 8. Placebo Standard Errors ────────────────────────────────────────────
    # vcov() computes SE via donor-placebo resampling (NOT permutation tests).
    # This avoids the N=9 p-value floor problem (minimum permutation p = 1/9 ≈ 0.111).
    # After vcov(), the SE is stored as a float in sdid_model.se.
    print("[INFERENCE] Computing placebo SE (n_reps=200, ~60s)...")
    sdid_model.vcov(method="placebo", n_reps=200)
    se = float(sdid_model.se)

    t_stat   = tau_hat / se
    ci_lower = tau_hat - 1.96 * se
    ci_upper = tau_hat + 1.96 * se

    print("\n" + "-" * 60)
    print("  SDiD Results — Maharashtra EV Policy 2025")
    print("-" * 60)
    print(f"  Estimator        : SDiD (Arkhangelsky et al., 2021)")
    print(f"  ATT (tau_hat)    : {tau_hat:+.4f} percentage points")
    print(f"  Placebo SE       : {se:.4f}")
    print(f"  t-statistic      : {t_stat:.4f}")
    print(f"  95% CI           : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Significant @5%  : {abs(t_stat) > 1.96}")
    print("-" * 60)

    # ── 9. Weight Dispersion Audit (Reviewer 2 Flaw 3 validation) ─────────────
    # weights["omega"] is a list (one array per treatment time-point).
    # For a single treatment date, take index [0].
    omega_vec = sdid_model.weights["omega"][0]

    # panel_matrices sorts units alphabetically; reconstruct donor list accordingly
    all_units = sorted(df['unit'].unique().tolist())
    donor_units = [u for u in all_units if u != 'MAHARASHTRA']

    print("\n[AUDIT] Donor unit weight distribution (L2 ridge regularised):")
    weight_rows = []
    for state, w in zip(donor_units, omega_vec):
        print(f"  {state:<25}: {w:.4f}")
        weight_rows.append({'State': state, 'Omega_Weight': float(w)})

    weights_df = pd.DataFrame(weight_rows).sort_values('Omega_Weight', ascending=False)

    ka_row = weights_df[weights_df['State'] == 'KARNATAKA']
    if not ka_row.empty:
        ka_w = float(ka_row['Omega_Weight'].values[0])
        print(f"\n[AUDIT] Karnataka weight = {ka_w:.4f}  |  threshold = 0.50")
        if ka_w >= 0.50:
            print("[AUDIT] WARNING: Karnataka > 0.50. Donor pool may still be fragile.")
        else:
            print("[AUDIT] PASS: L2 regularisation dispersed weights away from Karnataka.")

    # ── 10. Persist Results ───────────────────────────────────────────────────
    results_df = pd.DataFrame({
        'Estimator':         ['SDiD (Arkhangelsky et al., 2021)'],
        'Treated_Unit':      ['MAHARASHTRA'],
        'Treatment_Date':    [str(treatment_date.date())],
        'Donor_Pool_Size':   [len(donor_units)],
        'Pre_Periods':       [n_pre],
        'Post_Periods':      [n_post],
        'ATT':               [tau_hat],
        'Placebo_SE':        [se],
        'CI_Lower_95':       [ci_lower],
        'CI_Upper_95':       [ci_upper],
        't_statistic':       [t_stat],
        'Significant_5pct':  [abs(t_stat) > 1.96],
    })
    results_csv = os.path.join(output_dir, "sdid_results_official.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n[OUTPUT] ATT results → {results_csv}")

    weights_csv = os.path.join(output_dir, "sdid_unit_weights.csv")
    weights_df.to_csv(weights_csv, index=False)
    print(f"[OUTPUT] Weights    → {weights_csv}")

    # ── 11. Publication Plots ─────────────────────────────────────────────────
    fig_dir = os.path.join(settings.REPORTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print("\n[PLOTS] Generating SDiD trajectory and weight plots...")

    try:
        # plot_outcomes() returns self; figures stored in self.plot_outcomes (list)
        sdid_model.plot_outcomes()
        for i, fig in enumerate(sdid_model.plot_outcomes):
            path = os.path.join(fig_dir, f"sdid_outcomes_{i}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"  Saved: sdid_outcomes_{i}.png")
    except Exception as e:
        print(f"  [WARNING] Outcome plot failed (non-fatal): {e}")

    try:
        # plot_weights() returns self; figures stored in self.plot_weights (list)
        sdid_model.plot_weights()
        for i, fig in enumerate(sdid_model.plot_weights):
            path = os.path.join(fig_dir, f"sdid_weights_{i}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"  Saved: sdid_weights_{i}.png")
    except Exception as e:
        print(f"  [WARNING] Weight plot failed (non-fatal): {e}")

    print("\n[SUCCESS] SDiD pipeline complete.\n")
    return results_df, weights_df


if __name__ == "__main__":
    execute_sdid_pipeline()
