"""
Causal Modeling Module: Difference-in-Differences Baseline
Estimates the causal impact using a standard Two-Way Fixed Effects (TWFE) DiD model
as a robustness baseline against the SCM estimates.
"""

import os
import sys
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def run_did_model():
    """Runs a TWFE DiD model on the unified feature matrix."""
    print("Running Difference-in-Differences Baseline Model...")
    
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_feature_matrix.parquet")
    output_dir = os.path.join(settings.MODELS_DIR, "scm_results")
    
    # Load data
    df = pd.read_parquet(input_path)
    
    # Drop rows with missing values in the variables used for regression
    cols_to_use = ['ev_penetration_rate', 'did_treat_post', 'gsdp_per_capita', 'urban_population_pct', 'district', 'month']
    df = df.dropna(subset=cols_to_use).copy()
    
    # Define categorical variables for Fixed Effects
    df['district_cat'] = df['district'].astype('category')
    df['month_cat'] = df['month'].astype('category')
    
    # We want to estimate:
    # ev_penetration_rate = alpha + beta * did_treat_post + gamma * X + district_FE + month_FE + error
    
    # Standard TWFE formula
    # C(district_cat) = District Fixed Effects
    # C(month_cat) = Time Fixed Effects
    formula = "ev_penetration_rate ~ did_treat_post + gsdp_per_capita + urban_population_pct + C(district_cat) + C(month_cat)"
    
    print("Fitting DiD regression with district and time fixed effects...")
    model = smf.ols(formula, data=df)
    
    # Cluster standard errors at the district level
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['district']})
    
    # Extract the treatment effect
    coef = results.params['did_treat_post']
    se = results.bse['did_treat_post']
    p_val = results.pvalues['did_treat_post']
    
    print(f"\nDiD Estimated Average Treatment Effect (ATE): {coef:.4f}")
    print(f"Standard Error: {se:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    # Save full summary to a text file
    summary_str = results.summary().as_text()
    with open(os.path.join(output_dir, "did_baseline_summary.txt"), "w") as f:
        f.write(summary_str)
        
    print("DiD estimation complete. Results saved.")

if __name__ == "__main__":
    run_did_model()
