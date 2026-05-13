"""
EDA Module: Visualizations
Generates high-quality, publication-ready figures for pre/post policy trends
in EV adoption and Air Quality.
"""

import os
import sys
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

# Apply academic/publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

def generate_visualizations():
    """Generates all EDA plots."""
    print("Starting Visualization Generation...")
    
    input_path = os.path.join(settings.PROCESSED_DATA_DIR, "final_feature_matrix.parquet")
    fig_dir = os.path.join(settings.REPORTS_DIR, "figures")
    
    if not os.path.exists(input_path):
        print("Feature matrix not found. Run Phase 3 first.")
        sys.exit(1)
        
    df = pl.read_parquet(input_path)
    # Convert to Pandas for seaborn/matplotlib compatibility
    df_pd = df.to_pandas()
    
    treatment_date = datetime.strptime(settings.TREATMENT_DATE, "%Y-%m-%d")
    
    # --- Figure 1: EV Adoption Trends ---
    print("Plotting EV Adoption Trends...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate monthly averages by group
    trend_df = df_pd.groupby(['month', 'is_treated_district'])['ev_penetration_rate'].mean().reset_index()
    treated_trend = trend_df[trend_df['is_treated_district'] == 1]
    control_trend = trend_df[trend_df['is_treated_district'] == 0]
    
    ax.plot(treated_trend['month'], treated_trend['ev_penetration_rate'], 
            label='Treated Districts (VGF + Toll Waiver)', color='#1f77b4', linewidth=2)
    ax.plot(control_trend['month'], control_trend['ev_penetration_rate'], 
            label='Control Districts (Donor Pool)', color='#ff7f0e', linewidth=2, linestyle='--')
            
    # Add policy intervention line
    ax.axvline(x=treatment_date, color='red', linestyle=':', linewidth=2, label='Policy Announcement (May 2025)')
    
    ax.set_title('Average EV Penetration Rate over Time')
    ax.set_ylabel('EV Penetration Rate (%)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_ev_adoption_trends.png"))
    plt.close()
    
    # --- Figure 2: PM2.5 Trends ---
    print("Plotting PM2.5 Trends...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    trend_pm_df = df_pd.groupby(['month', 'is_treated_district'])['pm25_monthly_mean'].mean().reset_index()
    treated_pm = trend_pm_df[trend_pm_df['is_treated_district'] == 1]
    control_pm = trend_pm_df[trend_pm_df['is_treated_district'] == 0]
    
    ax.plot(treated_pm['month'], treated_pm['pm25_monthly_mean'], 
            label='Treated Districts', color='#2ca02c', linewidth=2)
    ax.plot(control_pm['month'], control_pm['pm25_monthly_mean'], 
            label='Control Districts', color='#d62728', linewidth=2, linestyle='--')
            
    ax.axvline(x=treatment_date, color='red', linestyle=':', linewidth=2)
    
    ax.set_title('Average Monthly PM2.5 Concentrations over Time')
    ax.set_ylabel('PM2.5 (μg/m³)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_pm25_trends.png"))
    plt.close()
    
    # --- Figure 3: Correlation Heatmap ---
    print("Plotting Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Select pre-treatment numeric features
    cols = ['ev_penetration_rate', 'pm25_monthly_mean', 'gsdp_per_capita', 'urban_population_pct', 'charging_station_density']
    corr = df_pd[df_pd['month'] < treatment_date][cols].corr()
    
    # Clean labels for plot
    labels = ['EV Rate', 'PM2.5', 'GSDP per Capita', 'Urbanization (%)', 'Charging Density']
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                xticklabels=labels, yticklabels=labels, ax=ax)
                
    ax.set_title('Pre-Treatment Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_correlation_heatmap.png"))
    plt.close()
    
    print(f"Visualizations saved to {fig_dir}")

if __name__ == "__main__":
    generate_visualizations()
