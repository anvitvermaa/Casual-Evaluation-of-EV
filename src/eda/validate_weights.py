import os
import sys
import pandas as pd

# Absolute path handling for execution from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

def validate_sdid_weights():
    weights_path = os.path.join(settings.MODELS_DIR, "scm_results", "sdid_unit_weights.csv")
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found at {weights_path}. Did SDiD run successfully?")
        sys.exit(1)
        
    df = pd.read_csv(weights_path)
    
    # Sort weights descending to find the top anchors
    df = df.sort_values(by='Weight', ascending=False).reset_index(drop=True)
    
    # Assertions for dispersion
    karnataka_weight = df.loc[df['State'] == 'KARNATAKA', 'Weight'].values[0]
    
    print("\n" + "="*50)
    print("⚖️  SDiD WEIGHT DISPERSION VALIDATION")
    print("="*50)
    
    print("\n[EMPIRICAL WEIGHT DISTRIBUTION]")
    for idx, row in df.head(3).iterrows():
        print(f"  {idx+1}. {row['State']:<20} : {row['Weight']:.4f}")
    
    print(f"  ... and {len(df)-3} other states.")
    
    print("\n[VALIDATION CHECK]")
    print(f"Target: Karnataka weight must be < 0.5000")
    print(f"Actual: Karnataka weight = {karnataka_weight:.4f}")
    
    # Strict validation assertion
    if karnataka_weight >= 0.50:
        print("\n❌ [FAIL] The L2 Ridge Regularization failed to adequately disperse the weights.")
        print("Karnataka remains a fragile single-unit anchor.")
        sys.exit(1)
    else:
        print("\n✅ [PASS] The 64% anchor has been broken. Weights are empirically dispersed.")
        print("L2 Ridge Regularization successfully stabilized the Synthetic Control.")

if __name__ == "__main__":
    validate_sdid_weights()
