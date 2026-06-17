"""
Causal Modeling Module: Causal Forest
Estimates Heterogeneous Treatment Effects (HTE) using EconML's Causal Forest
to understand which district characteristics amplify the policy's impact.
"""

import os
import sys
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import shap

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def run_causal_forest():
    """Trains a Causal Forest to estimate CATE."""
    print("Running Causal Forest for Heterogeneous Treatment Effects...")
    
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_feature_matrix.parquet")
    output_dir = os.path.join(settings.MODELS_DIR, "scm_results")
    
    # Load data
    df = pd.read_parquet(input_path)

    # Drop NaNs
    cols = ['ev_penetration_rate', 'is_treated_district', 'gsdp_per_capita', 'urban_population_pct', 'charging_station_density']
    df = df.dropna(subset=cols).copy()
    
    # We only use post-treatment data for evaluating the actual treatment effect heterogeneity
    treatment_date = pd.to_datetime(settings.TREATMENT_DATE)
    df_post = df[pd.to_datetime(df['month']) >= treatment_date]
    
    # Define variables
    Y = df_post['ev_penetration_rate'] # Outcome
    T = df_post['is_treated_district'] # Treatment
    
    # Covariates to observe heterogeneity over (X)
    X_cols = ['gsdp_per_capita', 'urban_population_pct', 'charging_station_density']
    X = df_post[X_cols]
    
    # Controls (W)
    W = None
    
    print("Fitting CausalForestDML...")
    # Initialize Causal Forest
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=settings.RANDOM_SEED),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=settings.RANDOM_SEED),
        discrete_treatment=False,
        n_estimators=500,
        random_state=settings.RANDOM_SEED
    )
    
    model.fit(Y, T, X=X, W=W)
    
    # Estimate CATE (Conditional Average Treatment Effect) for each district
    df_post_districts = df_post.groupby('district')[X_cols].mean().reset_index()
    X_test = df_post_districts[X_cols]
    
    cate = model.effect(X_test)
    df_post_districts['Estimated_CATE'] = cate
    
    # Save CATE results
    df_post_districts.sort_values('Estimated_CATE', ascending=False, inplace=True)
    df_post_districts.to_csv(os.path.join(output_dir, "causal_forest_hte.csv"), index=False)
    
    print("Causal Forest estimation complete.")
    print("\nTop 3 districts with highest treatment effect:")
    print(df_post_districts[['district', 'Estimated_CATE']].head(3).to_string(index=False))
    
    # SHAP Values for Interpretability
    print("Generating SHAP values for HTE interpretability...")
    try:
        shap_values = model.shap_values(X_test)
        # EconML shap_values returns dict[Y_name, dict[T_name, array]]
        # We index using the series names or defaults if not found
        y_name = Y.name if hasattr(Y, 'name') and Y.name in shap_values else list(shap_values.keys())[0]
        t_name = T.name if hasattr(T, 'name') and T.name in shap_values[y_name] else list(shap_values[y_name].keys())[0]
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[y_name][t_name], X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(settings.REPORTS_DIR, "figures", "causal_forest_shap.png"))
        plt.close()
        print("SHAP plot saved.")
    except Exception as e:
        print(f"Skipping SHAP plot due to error: {e}")

if __name__ == "__main__":
    # Ensure shap is installed if not already
    try:
        import shap
    except ImportError:
        print("Installing SHAP...")
        os.system("pip install shap")
        import shap
        
    run_causal_forest()
