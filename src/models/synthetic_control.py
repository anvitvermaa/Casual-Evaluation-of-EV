"""
Causal Modeling Module: Synthetic Difference-in-Differences (SDiD)
Maharashtra EV Policy Evaluation — N=16 macro-state panel, Jan 2022–Jun 2026.

Implements the d2cml-ai/synthdid official Python API
as defined by Arkhangelsky et al. (2021, AER 111(12), 4088-4118).

Dual-Specification Architecture:
  1. Main Model       (N=16): Maharashtra vs. all 15 donor states.
  2. Spatial Donut Hole (N=11): Maharashtra vs. 10 non-bordering donors.
     Bordering states dropped: Gujarat, Madhya Pradesh, Chhattisgarh,
     Telangana, Karnataka (SUTVA cross-border spillover check).

Methodological Defences:
  - Placebo SE via .vcov(method='placebo', n_reps=200): bypasses the
    1/N p-value floor issue for small macro-panels.
  - cov_method='optimized': projects out GSDP and Urbanisation OVB
    (Arkhangelsky et al., Eq 4.1).
  - L2 ridge regularisation (default zeta_omega from synthdid): disperses
    donor weights away from single-state anchor fragility.
  - Consecutive 0-based integer time index: required by synthdid internals
    (CRITICAL — do NOT use YYYYMM strings).

Outputs:
  - models/scm_results/sdid_unified_results.csv
  - paper/tables/main_empirical_results.tex
  - reports/figures/sdid_{model}_outcomes_{i}.png
  - reports/figures/sdid_{model}_weights_{i}.png
"""

import os
import sys
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

try:
    from synthdid.synthdid import Synthdid
except ImportError as e:
    raise ImportError(
        "\n[FATAL] d2cml-ai/synthdid not found in environment. "
        "Activate: conda activate ev-policy-sdid\n"
        f"Original: {e}"
    )

# ── Economic Covariates: all 16 states ──────────────────────────────────────
# Source: RBI Handbook of Statistics on Indian States 2023-24 (Table 1)
# Urbanisation: Census 2011 projections (Office of the Registrar General)
ECONOMIC_DATA = pd.DataFrame([
    {'state': 'MAHARASHTRA',    'GSDP_per_capita': 274000, 'Urbanization_Rate': 45.2},
    {'state': 'KARNATAKA',      'GSDP_per_capita': 278000, 'Urbanization_Rate': 38.6},
    {'state': 'GUJARAT',        'GSDP_per_capita': 275000, 'Urbanization_Rate': 42.6},
    {'state': 'TAMIL NADU',     'GSDP_per_capita': 273000, 'Urbanization_Rate': 48.4},
    {'state': 'TELANGANA',      'GSDP_per_capita': 308000, 'Urbanization_Rate': 38.9},
    {'state': 'ANDHRA PRADESH', 'GSDP_per_capita': 219000, 'Urbanization_Rate': 29.6},
    {'state': 'MADHYA PRADESH', 'GSDP_per_capita': 140000, 'Urbanization_Rate': 27.6},
    {'state': 'RAJASTHAN',      'GSDP_per_capita': 156000, 'Urbanization_Rate': 24.8},
    {'state': 'UTTAR PRADESH',  'GSDP_per_capita': 83000,  'Urbanization_Rate': 22.3},
    {'state': 'KERALA',         'GSDP_per_capita': 243000, 'Urbanization_Rate': 47.7},
    {'state': 'HARYANA',        'GSDP_per_capita': 247000, 'Urbanization_Rate': 34.8},
    {'state': 'WEST BENGAL',    'GSDP_per_capita': 130000, 'Urbanization_Rate': 31.9},
    {'state': 'PUNJAB',         'GSDP_per_capita': 185000, 'Urbanization_Rate': 37.5},
    {'state': 'ODISHA',         'GSDP_per_capita': 145000, 'Urbanization_Rate': 16.7},
    {'state': 'BIHAR',          'GSDP_per_capita': 57000,  'Urbanization_Rate': 11.3},
    {'state': 'CHHATTISGARH',   'GSDP_per_capita': 130000, 'Urbanization_Rate': 23.2},
])

# Bordering states dropped in the Donut Hole specification
BORDER_STATES = ["GUJARAT", "MADHYA PRADESH", "CHHATTISGARH", "TELANGANA", "KARNATAKA"]


def run_sdid(df_input: pd.DataFrame, model_name: str, fig_dir: str) -> dict:
    """
    Execute the SDiD estimator on the supplied panel dataframe.

    Parameters
    ----------
    df_input   : Prepared panel — columns: state, year_month, outcome, GSDP, Urban
    model_name : Human-readable label for logging/saving
    fig_dir    : Directory to save outcome/weight trajectory plots

    Returns
    -------
    dict with ATT, SE, CI, donor count, and significance flag.
    """
    print(f"\n{'─'*60}")
    print(f"  MODEL: {model_name}")
    print(f"  Donors (N₀): {df_input['state'].nunique() - 1}")
    print(f"{'─'*60}")

    df = df_input.copy()
    df['month'] = pd.to_datetime(df['year_month'])

    # ── Consecutive Integer Time Index ───────────────────────────────────────
    sorted_months = sorted(df['month'].unique())
    time_map      = {m: i for i, m in enumerate(sorted_months)}
    df['time']    = df['month'].map(time_map)

    # ── Merge Economic Covariates ────────────────────────────────────────────
    df = pd.merge(df, ECONOMIC_DATA, on='state', how='left')

    # ── Treatment Indicator ──────────────────────────────────────────────────
    treatment_date     = pd.to_datetime(settings.TREATMENT_DATE)
    treatment_time_idx = next(
        (time_map[m] for m in sorted_months if pd.Timestamp(m) >= treatment_date),
        None
    )
    if treatment_time_idx is None:
        raise ValueError(
            f"Treatment date {settings.TREATMENT_DATE} is beyond panel end "
            f"({df['month'].max().date()})."
        )

    df['treatment'] = (
        (df['state'] == 'MAHARASHTRA') & (df['time'] >= treatment_time_idx)
    ).astype(int)

    n_pre  = int(treatment_time_idx)
    n_post = int(df['time'].max() - treatment_time_idx + 1)
    print(f"  Pre-periods  : {n_pre}")
    print(f"  Post-periods : {n_post}")
    print(f"  Treated obs  : {df['treatment'].sum()}")

    # ── Rename to library defaults ───────────────────────────────────────────
    df = df.rename(columns={'state': 'unit'})

    # ── Initialise & Fit SDiD ────────────────────────────────────────────────
    sdid = Synthdid(
        data=df,
        unit='unit',
        time='time',
        treatment='treatment',
        outcome='outcome',
        covariates=['GSDP_per_capita', 'Urbanization_Rate']
    )
    sdid.fit(cov_method='optimized')
    tau_hat = float(sdid.att)
    print(f"  ATT (τ̂)      : {tau_hat:+.4f} pp")

    # ── Placebo Standard Errors ───────────────────────────────────────────────
    print(f"  Computing placebo SE (n_reps=200)...")
    sdid.vcov(method="placebo", n_reps=200)
    se = float(sdid.se)

    t_stat   = tau_hat / se if se > 0 else 0.0
    ci_lower = tau_hat - 1.96 * se
    ci_upper = tau_hat + 1.96 * se

    print(f"  Placebo SE   : {se:.4f}")
    print(f"  t-statistic  : {t_stat:.4f}")
    print(f"  95% CI       : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Significant  : {'YES ★' if abs(t_stat) > 1.96 else 'No'}")

    # ── Donor Weight Audit ───────────────────────────────────────────────────
    all_units   = sorted(df['unit'].unique().tolist())
    donor_units = [u for u in all_units if u != 'MAHARASHTRA']
    omega_vec   = sdid.weights["omega"][0]

    print(f"\n  Donor weight distribution (L2-regularised):")
    for unit, w in sorted(zip(donor_units, omega_vec), key=lambda x: -x[1]):
        bar = "█" * int(w * 30)
        print(f"    {unit:<25}  {w:.4f}  {bar}")

    # ── Save Plots ─────────────────────────────────────────────────────────
    label = model_name.replace(" ", "_").lower()
    try:
        sdid.plot_outcomes()
        for i, fig in enumerate(sdid.plot_outcomes):
            p = os.path.join(fig_dir, f"sdid_{label}_outcomes_{i}.png")
            fig.savefig(p, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  [PLOT] Saved: {os.path.basename(p)}")
    except Exception as e:
        print(f"  [WARNING] Outcome plot failed: {e}")

    try:
        sdid.plot_weights()
        for i, fig in enumerate(sdid.plot_weights):
            p = os.path.join(fig_dir, f"sdid_{label}_weights_{i}.png")
            fig.savefig(p, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  [PLOT] Saved: {os.path.basename(p)}")
    except Exception as e:
        print(f"  [WARNING] Weight plot failed: {e}")

    return {
        'Model':       model_name,
        'Donors (N₀)': len(donor_units),
        'Pre-periods': n_pre,
        'Post-periods': n_post,
        'ATT (pp)':    round(tau_hat, 4),
        'Placebo SE':  round(se, 4),
        't-stat':      round(t_stat, 4),
        'CI Lower':    round(ci_lower, 4),
        'CI Upper':    round(ci_upper, 4),
        'Sig. @5%':    abs(t_stat) > 1.96,
    }


def generate_latex_table(results: list[dict], output_path: str):
    """Write the dual-specification empirical results table in LaTeX."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SDiD Estimates of the Maharashtra EV Subsidy Policy (2025)}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l c c c c c}",
        r"\hline\hline",
        r"\textbf{Specification} & \textbf{Donors} & \textbf{ATT (pp)} "
        r"& \textbf{Placebo SE} & \textbf{95\% CI} & \textbf{Sig.} \\",
        r"\hline",
    ]

    for r in results:
        ci_str  = f"[{r['CI Lower']:.2f}, {r['CI Upper']:.2f}]"
        sig_str = r"Yes$^{*}$" if r['Sig. @5%'] else "No"
        lines.append(
            f"{r['Model']} & {r['Donors (N₀)']} & {r['ATT (pp)']:.2f} "
            f"& ({r['Placebo SE']:.2f}) & {ci_str} & {sig_str} \\\\"
        )

    lines += [
        r"\hline\hline",
        r"\multicolumn{6}{l}{\footnotesize \textit{Notes:} Standard errors from"
        r" 200 placebo replications. Outcome: EV penetration rate (\%).} \\",
        r"\multicolumn{6}{l}{\footnotesize $^{*}$ Significant at the 5\% level."
        r" ATT = Average Treatment Effect on the Treated.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n[LATEX] Table written → {output_path}")


def execute_pipeline():
    print("\n" + "=" * 60)
    print("  SYNTHETIC DiD — Maharashtra EV Policy 2025")
    print("  Dual-Specification: Main Model + Spatial Donut Hole")
    print("=" * 60)

    # ── Load panel ───────────────────────────────────────────────────────────
    panel_path = os.path.join(
        settings.PROCESSED_DATA_DIR, "final_state_feature_matrix_main.parquet"
    )
    if not os.path.exists(panel_path):
        print(f"[ERROR] Panel not found: {panel_path}")
        print("Run: python src/features/polars_transform.py")
        return

    df_full = pd.read_parquet(panel_path)[["state", "year_month", "outcome"]]
    print(f"[DATA] Loaded: {len(df_full)} obs, {df_full['state'].nunique()} states")
    print(f"[DATA] States: {sorted(df_full['state'].unique())}")

    # ── Output directories ────────────────────────────────────────────────────
    fig_dir     = os.path.join(settings.REPORTS_DIR, "figures")
    results_dir = os.path.join(settings.MODELS_DIR, "scm_results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    results = []

    # ── 1. Main Model (N=16) ─────────────────────────────────────────────────
    res = run_sdid(df_full, "Main Model (N=16)", fig_dir)
    results.append(res)

    # ── 2. Spatial Donut Hole (N=11) ─────────────────────────────────────────
    df_donut = df_full[~df_full['state'].isin(BORDER_STATES)].copy()
    print(f"\n[DONUT] Excluded borders: {BORDER_STATES}")
    print(f"[DONUT] Remaining states: {sorted(df_donut['state'].unique())}")
    res = run_sdid(df_donut, "Donut Hole (N=11)", fig_dir)
    results.append(res)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_df  = pd.DataFrame(results)
    csv_path    = os.path.join(results_dir, "sdid_dual_spec_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OUTPUT] Results CSV → {csv_path}")

    # ── Save LaTeX Table ──────────────────────────────────────────────────────
    tex_path = os.path.join(settings.PROJECT_ROOT, "paper", "tables", "main_empirical_results.tex")
    generate_latex_table(results, tex_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\n[SUCCESS] SDiD dual-specification pipeline complete.")


if __name__ == "__main__":
    execute_pipeline()
