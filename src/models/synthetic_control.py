"""
Causal Modeling Module: Synthetic Control Method (SCM) (Macro-State Level)
Constructs a synthetic control group from donor states, calculates ATE, and runs in-space placebo tests.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

warnings.filterwarnings('ignore')

def optimize_weights(X_treated, X_donor):
    n_donors = X_donor.shape[1]
    def loss(W, X_t, X_d):
        diff = X_t - X_d.dot(W)
        return np.sum(diff ** 2)
    w0 = np.ones(n_donors) / n_donors
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, 1) for _ in range(n_donors)]
    res = minimize(loss, w0, args=(X_treated, X_donor), method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def run_scm(outcome_var='ev_penetration_rate'):
    print(f"\n--- Running Macro-State SCM and Placebo Tests for: {outcome_var} ---")
    
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_state_feature_matrix.parquet")
    output_dir = os.path.join(settings.MODELS_DIR, "scm_results")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"File {input_path} not found. Ensure pipeline has run.")
        return

    df = pd.read_parquet(input_path)
    df['date'] = pd.to_datetime(df['month'])
    treatment_date = pd.to_datetime(settings.TREATMENT_DATE)
    
    treated_state = "MAHARASHTRA"
    donor_states = [s for s in df['state'].unique() if s != treated_state]
    
    # Create treated series
    df_treated = df[df['state'] == treated_state][['date', outcome_var]].copy()
    df_treated.rename(columns={outcome_var: 'Treated'}, inplace=True)
    
    # Create donor matrix
    df_donor = df[df['state'].isin(donor_states)].pivot(index='date', columns='state', values=outcome_var)
    
    panel = pd.merge(df_treated, df_donor, on='date', how='inner')
    panel.set_index('date', inplace=True)
    
    # Forward and back fill missing values for models
    panel.ffill(inplace=True)
    panel.bfill(inplace=True)
    panel.fillna(0, inplace=True) # Final safety net
    
    pre_mask = panel.index < treatment_date
    post_mask = panel.index >= treatment_date
    
    X_pre = panel[pre_mask]
    X_treated_pre = X_pre['Treated'].values
    X_donor_pre = X_pre[donor_states].values
    
    weights = optimize_weights(X_treated_pre, X_donor_pre)
    synthetic_full = panel[donor_states].values.dot(weights)
    panel['Synthetic'] = synthetic_full
    panel['Gap_Treated'] = panel['Treated'] - panel['Synthetic']
    
    ate_post = panel.loc[post_mask, 'Gap_Treated'].mean()
    rmspe_pre = np.sqrt(np.mean(panel.loc[pre_mask, 'Gap_Treated']**2))
    
    print(f"True Treated ATE: {ate_post:.4f} (Pre-treatment RMSPE: {rmspe_pre:.4f})")
    print("\nOptimal State Weights:")
    for state, weight in zip(donor_states, weights):
        if weight > 0.001:
            print(f"  {state}: {weight:.3f}")
            
    # --- Placebo Tests (In-space permutations) ---
    placebo_gaps = {}
    placebo_ates = []
    
    for donor in donor_states:
        # Pretend 'donor' is treated, rest of donors are the new donor pool
        placebo_pool = [d for d in donor_states if d != donor]
        
        X_placebo_pre = X_pre[donor].values
        X_placebo_donor_pre = X_pre[placebo_pool].values
        
        w_placebo = optimize_weights(X_placebo_pre, X_placebo_donor_pre)
        
        synthetic_placebo = panel[placebo_pool].values.dot(w_placebo)
        gap_placebo = panel[donor] - synthetic_placebo
        
        # Calculate RMSPE for placebo
        rmspe_placebo_pre = np.sqrt(np.mean(gap_placebo[pre_mask]**2))
        
        # Only keep placebos with good pre-treatment fit (e.g., RMSPE < 5 * Treated RMSPE)
        if rmspe_placebo_pre < 5 * rmspe_pre:
            placebo_gaps[donor] = gap_placebo
            ate_placebo = gap_placebo[post_mask].mean()
            placebo_ates.append(ate_placebo)
            
    # Calculate Pseudo P-value
    if outcome_var == 'ev_penetration_rate':
        # We expect a positive effect
        p_val = np.mean(np.array(placebo_ates) >= ate_post)
    else:
        # PM2.5 we expect a negative effect
        p_val = np.mean(np.array(placebo_ates) <= ate_post)
        
    print(f"\nPlacebo Pseudo P-value: {p_val:.4f} (based on {len(placebo_ates)} valid placebos out of {len(donor_states)})")
    
    # --- Plotting Gap & Placebos ---
    plt.figure(figsize=(10, 6))
    
    for donor, gap in placebo_gaps.items():
        plt.plot(panel.index, gap, color='gray', alpha=0.3, linewidth=1)
        
    plt.plot(panel.index, panel['Gap_Treated'], color='red', linewidth=3, label=f'Treated (ATE={ate_post:.2f}, p={p_val:.3f})')
    plt.axvline(x=treatment_date, color='black', linestyle=':', label='Policy Implementation')
    plt.axhline(y=0, color='black', linestyle='--')
    
    plt.title(f'Treatment Effect Gap vs Placebos: {outcome_var}')
    plt.ylabel(f'Gap in {outcome_var}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(settings.REPORTS_DIR, "figures", f"scm_state_placebo_{outcome_var}.png"))
    plt.close()
    
    # Save results dictionary
    results_df = pd.DataFrame({
        'Metric': ['ATE', 'Pre_RMSPE', 'P_Value', 'Valid_Placebos'],
        'Value': [ate_post, rmspe_pre, p_val, len(placebo_ates)]
    })
    results_df.to_csv(os.path.join(output_dir, f"scm_state_results_{outcome_var}.csv"), index=False)

if __name__ == "__main__":
    run_scm('ev_penetration_rate')
    run_scm('pm25_monthly_mean')
