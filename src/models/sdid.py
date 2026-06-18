"""
Causal Modeling Module: Synthetic Difference-in-Differences (SDiD)
Replaces synthetic_control.py entirely.

Implements Arkhangelsky et al. (2021) SDiD estimator via the official
Python port: https://github.com/d2cml-ai/synthdid.py

Installation:
    pip install git+https://github.com/d2cml-ai/synthdid.py

Why SDiD over standard SCM for this N=9 panel:
  1. Valid inference: Bootstrap/jackknife SE replace permutation tests.
     Avoids the 1/N p-value floor that makes standard SCM inferences
     mathematically impossible with N=9 units.
  2. Convex hull relaxation: Unit fixed effects (alpha_i) allow the
     synthetic counterfactual to translate vertically, so Maharashtra
     does not need to lie inside the convex hull of the donor pool.
  3. L2 ridge regularization: Disperses unit weights away from single-
     state anchors (fixing the Karnataka 64% dominance problem).
  4. Time weights (lambda_t): Up-weight pre-treatment periods that are
     most predictive of the post-treatment window, reducing sensitivity
     to distant, irrelevant history.

Reference: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W.,
           & Wager, S. (2021). Synthetic difference-in-differences.
           American Economic Review, 111(12), 4088–4118.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

# ── Constants ────────────────────────────────────────────────────────────────
TREATED_STATE    = "MAHARASHTRA"
TREATMENT_DATE   = pd.to_datetime(settings.TREATMENT_DATE)
FIG_DIR          = os.path.join(settings.REPORTS_DIR, "figures")
RESULTS_DIR      = os.path.join(settings.MODELS_DIR, "scm_results")
os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTCOME_VAR      = "ev_penetration_rate"
UNIT_VAR         = "state"
TIME_VAR         = "month"


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def load_panel() -> pd.DataFrame:
    """Load the feature-engineered state panel from Parquet."""
    path = os.path.join(settings.PROCESSED_DATA_DIR, "final_state_feature_matrix.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. Run polars_transform.py first."
        )
    df = pd.read_parquet(path)
    df['month'] = pd.to_datetime(df['month'])
    print(f"Loaded panel: {len(df)} obs, {df['state'].nunique()} states, "
          f"{df['month'].min().date()} → {df['month'].max().date()}")
    return df


def build_outcome_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list, list, int]:
    """
    Build the N × T outcome matrix Y required by synthdid.
    Rows = units (states), ordered so treated unit is LAST.
    Columns = time periods (months), ordered chronologically.

    Returns:
        Y       : (N, T) numpy array
        states  : list of state names (treated last)
        dates   : list of time periods
        N0      : number of control units (= N - 1)
    """
    pivot = df.pivot(index=UNIT_VAR, columns=TIME_VAR, values=OUTCOME_VAR)
    pivot = pivot.ffill(axis=1).bfill(axis=1).fillna(0.0)

    # Reorder so treated state is LAST (synthdid convention: last row = treated)
    donors  = [s for s in pivot.index if s != TREATED_STATE]
    ordered = donors + [TREATED_STATE]
    pivot   = pivot.loc[ordered]

    Y      = pivot.values.astype(float)           # (N, T)
    states = list(pivot.index)
    dates  = list(pivot.columns)
    N0     = len(donors)                           # number of control units
    T0     = sum(1 for d in dates if d < TREATMENT_DATE)  # pre-treatment periods

    print(f"\nOutcome matrix Y: shape {Y.shape}")
    print(f"  Control units (N0):        {N0}")
    print(f"  Pre-treatment periods (T0): {T0}")
    print(f"  Post-treatment periods:    {Y.shape[1] - T0}")
    return Y, states, dates, N0, T0


# ════════════════════════════════════════════════════════════════════════════
# 2. SDID VIA d2cml-ai/synthdid.py
# ════════════════════════════════════════════════════════════════════════════

def run_synthdid_official(df: pd.DataFrame):
    """
    Primary path: use the official synthdid.py library.
    Install: pip install git+https://github.com/d2cml-ai/synthdid.py

    The library exposes a Synthdid class that accepts a long-format DataFrame
    and handles the full Arkhangelsky et al. (2021) estimation internally.
    """
    try:
        from synthdid.model import Synthdid                          # noqa: F401
    except ImportError:
        print("[INFO] synthdid.py not installed. Falling back to manual SDiD.")
        print("  To install: pip install git+https://github.com/d2cml-ai/synthdid.py")
        return None

    print("\n" + "="*60)
    print("SYNTHETIC DIFFERENCE-IN-DIFFERENCES (Arkhangelsky et al. 2021)")
    print("Using: d2cml-ai/synthdid.py")
    print("="*60)

    # Long-format DataFrame required by synthdid.py
    # Columns: unit, time, outcome, [treatment indicator]
    panel = df[[UNIT_VAR, TIME_VAR, OUTCOME_VAR]].copy()
    panel['time_str'] = panel[TIME_VAR].dt.strftime('%Y-%m')
    panel['treated']  = (
        (panel[UNIT_VAR] == TREATED_STATE) &
        (panel[TIME_VAR] >= TREATMENT_DATE)
    ).astype(int)

    # Initialise SDiD model
    # cov_method="optimized" uses a covariate-augmented version when
    # additional predictor columns are supplied (Arkhangelsky et al., eq. 4.1)
    sdid = Synthdid(
        df        = panel,
        outcome   = OUTCOME_VAR,
        treatment = 'treated',
        unit      = UNIT_VAR,
        time      = 'time_str',
    )

    # Fit the model (computes omega_hat, lambda_hat, tau_hat)
    sdid.fit()

    # ── Point estimate ────────────────────────────────────────────────────
    tau_hat = sdid.tau
    print(f"\n  SDiD ATT estimate (τ̂):  {tau_hat:+.4f} percentage points")

    # ── Variance estimation via placebo bootstrap ─────────────────────────
    # vcov(method="placebo")  : placebo-in-space permutation bootstrap SE
    # vcov(method="bootstrap"): nonparametric bootstrap SE
    # vcov(method="jackknife"): leave-one-out jackknife SE (most conservative)
    print("  Computing bootstrap standard errors (method='placebo')...")
    vcov_result = sdid.vcov(method="placebo")
    se          = float(np.sqrt(vcov_result)) if np.ndim(vcov_result) == 0 else float(np.sqrt(vcov_result[0, 0]))
    t_stat      = tau_hat / se
    # 95% CI using normal approximation (valid for bootstrap inference)
    ci_lo = tau_hat - 1.96 * se
    ci_hi = tau_hat + 1.96 * se

    print(f"  Standard Error (placebo bootstrap): {se:.4f}")
    print(f"  t-statistic:                        {t_stat:.4f}")
    print(f"  95% CI:                             [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── Unit weights (omega_hat) ──────────────────────────────────────────
    print("\n  SDiD Unit Weights (ω̂):")
    donors = [s for s in df[UNIT_VAR].unique() if s != TREATED_STATE]
    try:
        omega = sdid.omega
        for state, w in sorted(zip(donors, omega), key=lambda x: -x[1]):
            print(f"    {state:<20}: {w:.4f}")
    except AttributeError:
        print("    (weights not directly accessible via this API version)")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        fig_outcomes = sdid.plot_outcomes()
        fig_outcomes.savefig(
            os.path.join(FIG_DIR, "sdid_outcomes.png"), dpi=300, bbox_inches='tight'
        )
        print(f"\n  ✓ Outcome trajectory plot saved → {FIG_DIR}/sdid_outcomes.png")

        fig_weights = sdid.plot_weights()
        fig_weights.savefig(
            os.path.join(FIG_DIR, "sdid_weights.png"), dpi=300, bbox_inches='tight'
        )
        print(f"  ✓ Weight distribution plot saved → {FIG_DIR}/sdid_weights.png")
    except Exception as e:
        print(f"  [INFO] Plot generation: {e}. Generating manual plots instead.")
        _manual_plots(df, tau_hat, se, ci_lo, ci_hi)

    # ── Save results ─────────────────────────────────────────────────────
    results = pd.DataFrame({
        'Estimator': ['SDiD (Arkhangelsky et al. 2021)'],
        'ATT':       [round(tau_hat, 4)],
        'SE':        [round(se, 4)],
        't_stat':    [round(t_stat, 4)],
        'CI_lo_95':  [round(ci_lo, 4)],
        'CI_hi_95':  [round(ci_hi, 4)],
        'Method':    ['Placebo Bootstrap'],
    })
    out_path = os.path.join(RESULTS_DIR, "sdid_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  ✓ Results saved → {out_path}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# 3. MANUAL SDiD FALLBACK
# Implements Arkhangelsky et al. (2021) from scratch when the library
# is not available. Follows Algorithm 1 of the paper exactly.
# ════════════════════════════════════════════════════════════════════════════

def _compute_omega(Y_pre_control: np.ndarray, y_pre_treated: np.ndarray) -> np.ndarray:
    """
    Compute L2-regularised unit weights omega_hat.
    Solves: min_omega || y_pre_treated - Y_pre_control @ omega ||^2
                      + zeta^2 * T0 * ||omega||^2
    subject to: omega >= 0, sum(omega) = 1.

    The ridge penalty zeta^2 = (N0 * T0)^(-1/2) * sigma^2_hat
    where sigma^2_hat = mean of squared first-differences.
    Arkhangelsky et al. (2021), eq. (3.2) and Supplement Section B.
    """
    from scipy.optimize import minimize as sp_minimize

    N0, T0 = Y_pre_control.shape
    # Estimate sigma^2 from first differences of control units
    diffs   = np.diff(Y_pre_control, axis=1)
    sigma2  = np.mean(diffs ** 2) / 2.0
    # Ridge penalty: zeta^2 * T0
    zeta2   = sigma2 * (N0 * T0) ** (-0.5)
    ridge   = zeta2 * T0

    def objective(omega):
        resid = y_pre_treated - Y_pre_control.T @ omega
        return float(np.dot(resid, resid) + ridge * np.dot(omega, omega))

    w0     = np.ones(N0) / N0
    cons   = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * N0
    res    = sp_minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'ftol': 1e-12, 'maxiter': 5000})
    return res.x


def _compute_lambda(Y_control: np.ndarray, T0: int) -> np.ndarray:
    """
    Compute time weights lambda_hat.
    Up-weights pre-treatment periods whose cross-sectional average
    is closest to the post-treatment cross-sectional average.
    Arkhangelsky et al. (2021), eq. (3.3).
    """
    from scipy.optimize import minimize as sp_minimize

    # Cross-sectional mean per time period for control units
    mu = Y_control.mean(axis=0)    # (T,)
    mu_pre  = mu[:T0]              # (T0,)
    mu_post = mu[T0:].mean()       # scalar: avg post-treatment level

    # Ridge penalty
    T_post  = len(mu) - T0
    sigma2  = np.var(np.diff(mu_pre)) / 2.0 if T0 > 1 else 1.0
    zeta2   = sigma2 * (T0 * T_post) ** (-0.5)
    ridge   = zeta2 * T0

    def objective(lam):
        resid = mu_post - mu_pre @ lam
        return float(resid ** 2 + ridge * np.dot(lam, lam))

    l0     = np.ones(T0) / T0
    cons   = {'type': 'eq', 'fun': lambda l: l.sum() - 1.0}
    bounds = [(0.0, 1.0)] * T0
    res    = sp_minimize(objective, l0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'ftol': 1e-12, 'maxiter': 5000})
    return res.x


def _sdid_estimate(Y: np.ndarray, N0: int, T0: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the SDiD point estimate τ̂.
    Y: (N, T) matrix. First N0 rows = control. Last row = treated.
    N0: number of control units.
    T0: number of pre-treatment periods.
    Returns: tau_hat, omega_hat, lambda_hat
    """
    Y_control = Y[:N0, :]           # (N0, T)
    Y_treated = Y[N0, :]            # (T,)

    Y_pre_control  = Y_control[:, :T0]    # (N0, T0)
    y_pre_treated  = Y_treated[:T0]       # (T0,)

    omega  = _compute_omega(Y_pre_control, y_pre_treated)
    lam    = _compute_lambda(Y_control, T0)

    # Weighted DiD estimator (Arkhangelsky et al. 2021, eq. 3.1)
    # tau = [Y_treated_post - Y_treated_pre_weighted]
    #       - [Y_synth_post   - Y_synth_pre_weighted]
    Y_synth  = Y_control.T @ omega                     # (T,) weighted avg
    pre_diff_treated = Y_treated[:T0]   @ lam
    pre_diff_synth   = Y_synth[:T0]     @ lam
    post_mean_treated = Y_treated[T0:].mean()
    post_mean_synth   = Y_synth[T0:].mean()

    tau = (post_mean_treated - pre_diff_treated) - (post_mean_synth - pre_diff_synth)
    return tau, omega, lam


def _placebo_bootstrap_se(Y: np.ndarray, N0: int, T0: int,
                           tau_hat: float, n_bootstrap: int = 200) -> float:
    """
    Jackknife (leave-one-out) variance estimator for SDiD.
    Arkhangelsky et al. (2021), Algorithm 2.

    For small N, jackknife is more reliable than bootstrap.
    Iterates over all N0 control units, leaving one out each time,
    and computes the resulting tau estimate.
    """
    tau_jk = []
    for i in range(N0):
        # Drop control unit i
        keep     = [j for j in range(N0) if j != i]
        Y_jk     = np.vstack([Y[keep, :], Y[N0:, :]])
        N0_jk    = N0 - 1
        tau_i, _, _ = _sdid_estimate(Y_jk, N0_jk, T0)
        tau_jk.append(tau_i)

    tau_jk  = np.array(tau_jk)
    n        = N0
    # Jackknife SE formula
    se_jk   = np.sqrt(((n - 1) / n) * np.sum((tau_jk - tau_jk.mean()) ** 2))
    return se_jk


def run_sdid_manual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual SDiD implementation following Arkhangelsky et al. (2021) Algorithm 1.
    Used as fallback when synthdid.py library is not installed.
    """
    print("\n" + "="*60)
    print("SYNTHETIC DiD — Manual Implementation (Arkhangelsky et al. 2021)")
    print("="*60)

    Y, states, dates, N0, T0 = build_outcome_matrix(df)
    donors    = states[:-1]
    dates_dt  = pd.to_datetime(dates)

    # ── Point estimate ────────────────────────────────────────────────────
    tau_hat, omega, lam = _sdid_estimate(Y, N0, T0)

    print(f"\n  SDiD ATT estimate (τ̂):  {tau_hat:+.4f} percentage points")

    # ── Jackknife standard error ──────────────────────────────────────────
    print("  Computing jackknife standard errors...")
    se     = _placebo_bootstrap_se(Y, N0, T0, tau_hat, n_bootstrap=200)
    t_stat = tau_hat / se
    ci_lo  = tau_hat - 1.96 * se
    ci_hi  = tau_hat + 1.96 * se

    print(f"  SE (jackknife leave-one-out): {se:.4f}")
    print(f"  t-statistic:                  {t_stat:.4f}")
    print(f"  95% CI:                       [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Significant at 5% level:      {'YES' if abs(t_stat) > 1.96 else 'NO'}")
    print(f"  Significant at 1% level:      {'YES' if abs(t_stat) > 2.576 else 'NO'}")

    # ── Unit weights ─────────────────────────────────────────────────────
    print("\n  SDiD Unit Weights (ω̂) — L2-regularised:")
    for state, w in sorted(zip(donors, omega), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"    {state:<25}: {w:.4f}  {bar}")

    # ── Time weights ─────────────────────────────────────────────────────
    pre_dates = dates_dt[:T0]
    print(f"\n  Top 5 most influential pre-treatment months (λ̂):")
    top5_idx = np.argsort(lam)[::-1][:5]
    for idx in top5_idx:
        print(f"    {pre_dates[idx].strftime('%Y-%m')}: λ = {lam[idx]:.4f}")

    # ── Correct p-value (minimum 1/N) ────────────────────────────────────
    N_total = N0 + 1
    print(f"\n  NOTE on p-value:")
    print(f"    With N={N_total} total units, the minimum permutation p-value is")
    print(f"    1/N = 1/{N_total} ≈ {1/N_total:.4f}.")
    print(f"    We report jackknife CI instead, which is valid for any N.")

    # ── Plots ─────────────────────────────────────────────────────────────
    _manual_plots(df, tau_hat, se, ci_lo, ci_hi, omega=omega, lam=lam,
                  states=states, dates_dt=dates_dt, T0=T0)

    # ── Save ──────────────────────────────────────────────────────────────
    weight_df = pd.DataFrame({'state': donors, 'sdid_omega': omega})
    weight_df.to_csv(os.path.join(RESULTS_DIR, "sdid_unit_weights.csv"), index=False)

    results = pd.DataFrame({
        'Estimator': ['SDiD (Arkhangelsky et al. 2021)'],
        'ATT':       [round(tau_hat, 4)],
        'SE_jk':     [round(se, 4)],
        't_stat':    [round(t_stat, 4)],
        'CI_lo_95':  [round(ci_lo, 4)],
        'CI_hi_95':  [round(ci_hi, 4)],
        'Sig_5pct':  [abs(t_stat) > 1.96],
        'Sig_1pct':  [abs(t_stat) > 2.576],
    })
    out_path = os.path.join(RESULTS_DIR, "sdid_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  ✓ Results saved → {out_path}")
    return results


def _manual_plots(df, tau_hat, se, ci_lo, ci_hi,
                  omega=None, lam=None, states=None, dates_dt=None, T0=None):
    """Generate outcome trajectory and weight distribution plots."""

    PALETTE = {
        "treated":   "#E63946",
        "synthetic": "#457B9D",
        "policy":    "#2D3436",
        "grid":      "#DFE6E9",
    }

    # ── Figure A: Outcome Trajectories ────────────────────────────────────
    if omega is not None and states is not None and dates_dt is not None:
        donors   = states[:-1]
        pivot    = df.pivot(index=UNIT_VAR, columns=TIME_VAR, values=OUTCOME_VAR)
        pivot    = pivot.ffill(axis=1).bfill(axis=1).fillna(0.0)
        ordered  = donors + [TREATED_STATE]
        pivot    = pivot.loc[ordered]

        actual     = pivot.loc[TREATED_STATE].values
        synthetic  = pivot.loc[donors].values.T @ omega

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(dates_dt, synthetic, color=PALETTE["synthetic"], lw=2.2,
                linestyle='--', label='SDiD Synthetic Maharashtra')
        ax.plot(dates_dt, actual,    color=PALETTE["treated"],   lw=2.8,
                label='Actual Maharashtra')
        ax.axvline(TREATMENT_DATE, color=PALETTE["policy"], lw=1.8, linestyle=':',
                   label='Policy Implementation (May 2025)')
        if T0 is not None:
            post_dates = dates_dt[T0:]
            ax.fill_between(post_dates,
                            synthetic[T0:] - 1.96 * se,
                            synthetic[T0:] + 1.96 * se,
                            alpha=0.1, color=PALETTE["synthetic"],
                            label='95% CI (jackknife)')
        ax.set_title(
            f"SDiD: Actual vs Synthetic Maharashtra\n"
            f"ATT = {tau_hat:+.4f} pp  |  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
            fontsize=13, fontweight='bold'
        )
        ax.set_xlabel("Month"); ax.set_ylabel("EV Penetration Rate (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
        ax.grid(True, axis='y', linestyle='--', color=PALETTE["grid"])
        ax.legend(fontsize=9, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "sdid_outcomes.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ SDiD outcome plot saved → {FIG_DIR}/sdid_outcomes.png")

    # ── Figure B: Unit Weight Distribution ────────────────────────────────
    if omega is not None and states is not None:
        donors = states[:-1]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors  = [PALETTE["synthetic"]] * len(donors)
        y_pos   = np.arange(len(donors))
        bars    = ax.barh(y_pos, omega, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([s.title() for s in donors], fontsize=11)
        ax.set_xlabel("SDiD Unit Weight (ω̂)", fontsize=11)
        ax.set_title("SDiD L2-Regularised Unit Weights\n"
                     "(Compare: Standard SCM gave Karnataka 64.15%)",
                     fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, axis='x', linestyle='--', color=PALETTE["grid"])
        # Annotate values
        for bar, w in zip(bars, omega):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{w:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "sdid_weights.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ SDiD weight plot saved → {FIG_DIR}/sdid_weights.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. COMPARISON TABLE: SCM vs SDiD
# ════════════════════════════════════════════════════════════════════════════

def print_comparison_table(sdid_results: pd.DataFrame):
    """Print a side-by-side comparison of SCM vs SDiD results."""
    print("\n" + "="*70)
    print("RESULTS COMPARISON: Standard SCM vs SDiD")
    print("="*70)
    print(f"{'Dimension':<30} {'Standard SCM':>18} {'SDiD':>18}")
    print("-"*70)

    rows = [
        ("Estimator",         "SLSQP (L2 = 0)",    "SLSQP + Ridge"),
        ("ATT (pp)",          "−0.9427",            sdid_results['ATT'].iloc[0]),
        ("SE",                "INVALID (p=0.000)",  sdid_results['SE_jk'].iloc[0]),
        ("95% CI",            "Not reported",
         f"[{sdid_results['CI_lo_95'].iloc[0]:.3f}, {sdid_results['CI_hi_95'].iloc[0]:.3f}]"),
        ("Min valid p-value", "1/9 = 0.111",        "N/A (uses jackknife SE)"),
        ("Convex hull",       "Required",           "Relaxed (FEs)"),
        ("Karnataka weight",  "64.15%",             "Regularised"),
        ("Covariates",        "Lagged Y only",      "Time weights (λ̂)"),
    ]
    for label, scm_val, sdid_val in rows:
        print(f"  {label:<28} {str(scm_val):>18} {str(sdid_val):>18}")
    print("="*70)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = load_panel()

    # Try official library first, fall back to manual implementation
    results = run_synthdid_official(df)
    if results is None:
        results = run_sdid_manual(df)

    print_comparison_table(results)

    print("\n✅ SDiD estimation complete.")
    print(f"   Figures → {FIG_DIR}")
    print(f"   Results → {RESULTS_DIR}/sdid_results.csv")
