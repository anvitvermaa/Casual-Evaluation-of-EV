"""
Causal Modeling Module: Difference-in-Differences Baseline (Macro-State Level)
Estimates the causal impact using a standard Two-Way Fixed Effects (TWFE) DiD model
as a robustness baseline against the SCM estimates.
"""

import os
import sys
import pandas as pd
import statsmodels.formula.api as smf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def run_did_model():
    """Runs a TWFE DiD model on the macro-state feature matrix."""
    print("Running Macro-State Difference-in-Differences Baseline Model...")

    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_state_feature_matrix.parquet")
    output_dir = os.path.join(settings.MODELS_DIR, "scm_results")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"[ERROR] Feature matrix not found at {input_path}. Run polars_transform.py first.")
        return

    df = pd.read_parquet(input_path)

    # Drop rows with missing target variable
    df = df.dropna(subset=['ev_penetration_rate', 'did_treat_post']).copy()

    # Categorical fixed effects
    df['state_cat'] = df['state'].astype('category')
    df['month_cat'] = df['month'].astype('category')

    print(f"  Observations: {len(df)}")
    print(f"  States: {sorted(df['state'].unique())}")
    print(f"  Treatment obs (Maharashtra post-May 2025): {df['did_treat_post'].sum()}")

    # TWFE formula:
    # C(state_cat)  = State Fixed Effects  (absorbs time-invariant state-level variation)
    # C(month_cat)  = Month Fixed Effects  (absorbs all state-common temporal shocks)
    # did_treat_post = beta (our causal estimate)
    formula = "ev_penetration_rate ~ did_treat_post + C(state_cat) + C(month_cat)"

    print("\nFitting TWFE DiD regression with state and month fixed effects...")
    model = smf.ols(formula, data=df)

    # Cluster standard errors at the state level for heteroscedasticity-robust inference
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['state']})

    coef  = results.params['did_treat_post']
    se    = results.bse['did_treat_post']
    p_val = results.pvalues['did_treat_post']
    ci_lo, ci_hi = results.conf_int().loc['did_treat_post']

    print(f"\n{'='*50}")
    print(f"DiD Average Treatment Effect (ATE): {coef:+.4f} percentage points")
    print(f"Standard Error                     : {se:.4f}")
    print(f"95% Confidence Interval            : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"P-value                            : {p_val:.4f}")
    print(f"{'='*50}")

    if p_val < 0.01:
        print("Result: *** Statistically significant at 1% level")
    elif p_val < 0.05:
        print("Result: **  Statistically significant at 5% level")
    elif p_val < 0.10:
        print("Result: *   Statistically significant at 10% level")
    else:
        print("Result:     Not statistically significant at conventional levels")

    # Save full summary
    summary_path = os.path.join(output_dir, "did_state_baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write(results.summary().as_text())

    print(f"\nFull summary saved → {summary_path}")

if __name__ == "__main__":
    run_did_model()
