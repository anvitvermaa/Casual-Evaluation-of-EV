"""
Figure Generation: All publication-grade charts for the research paper.
Outputs 300 DPI PNG figures for journal submission.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import settings

warnings.filterwarnings('ignore')

# ── Aesthetic constants ──────────────────────────────────────────────────────
TREATMENT_DATE = pd.to_datetime(settings.TREATMENT_DATE)
TREATED_STATE  = "MAHARASHTRA"
FIG_DIR        = os.path.join(settings.REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    "treated":    "#E63946",   # vivid red  – Maharashtra
    "synthetic":  "#457B9D",   # steel blue – synthetic control
    "donor":      "#A8DADC",   # pale teal  – donor pool
    "policy":     "#2D3436",   # near-black – policy line
    "grid":       "#DFE6E9",
    "text":       "#2D3436",
}

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.edgecolor":     "#B2BEC3",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.color":         PALETTE["grid"],
    "grid.linewidth":     0.6,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

DONOR_COLORS = [
    "#6C5CE7","#00B894","#FDCB6E","#E17055",
    "#0984E3","#A29BFE","#55EFC4","#F9CA24"
]

STATE_LABELS = {
    "ANDHRA PRADESH": "Andhra Pradesh",
    "GUJARAT":        "Gujarat",
    "KARNATAKA":      "Karnataka",
    "MADHYA PRADESH": "Madhya Pradesh",
    "MAHARASHTRA":    "Maharashtra",
    "RAJASTHAN":      "Rajasthan",
    "TAMIL NADU":     "Tamil Nadu",
    "TELANGANA":      "Telangana",
    "UTTAR PRADESH":  "Uttar Pradesh",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def optimize_weights(X_treated, X_donor):
    n = X_donor.shape[1]
    loss = lambda W: np.sum((X_treated - X_donor.dot(W)) ** 2)
    res  = minimize(loss, np.ones(n)/n, method='SLSQP',
                    bounds=[(0,1)]*n,
                    constraints={'type':'eq','fun': lambda w: w.sum()-1})
    return res.x

def load_panel():
    path = os.path.join(settings.PROCESSED_DATA_DIR, "final_state_feature_matrix.parquet")
    df   = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['month'])
    return df

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – EV Penetration Trends: All States (2022–2026)
# ────────────────────────────────────────────────────────────────────────────

def fig1_all_state_trends(df):
    fig, ax = plt.subplots(figsize=(12, 6.5))

    donors = [s for s in df['state'].unique() if s != TREATED_STATE]
    for i, state in enumerate(donors):
        sub = df[df['state'] == state].sort_values('date')
        ax.plot(sub['date'], sub['ev_penetration_rate'],
                color=DONOR_COLORS[i % len(DONOR_COLORS)],
                linewidth=1.4, alpha=0.75, label=STATE_LABELS[state])

    mh = df[df['state'] == TREATED_STATE].sort_values('date')
    ax.plot(mh['date'], mh['ev_penetration_rate'],
            color=PALETTE["treated"], linewidth=3.2,
            label="Maharashtra (Treated)", zorder=5)

    ax.axvline(TREATMENT_DATE, color=PALETTE["policy"],
               linestyle='--', linewidth=1.6, label='Policy Implementation\n(May 2025)')

    ax.set_title("Figure 1: Monthly EV Penetration Rate by State (2022–2026)",
                 fontsize=14, fontweight='bold', pad=14, color=PALETTE["text"])
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("EV Penetration Rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.grid(True, axis='y', linestyle='--')
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9, ncol=2)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_state_trends.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 1 saved → {out}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – SCM: Actual Maharashtra vs Synthetic Control
# ────────────────────────────────────────────────────────────────────────────

def fig2_scm_counterfactual(df):
    donors = [s for s in df['state'].unique() if s != TREATED_STATE]

    df_treated = df[df['state'] == TREATED_STATE][['date','ev_penetration_rate']].copy()
    df_treated.rename(columns={'ev_penetration_rate':'Treated'}, inplace=True)

    df_donor = df[df['state'].isin(donors)].pivot(
        index='date', columns='state', values='ev_penetration_rate')

    panel = pd.merge(df_treated, df_donor, on='date', how='inner').set_index('date')
    panel.ffill(inplace=True); panel.bfill(inplace=True); panel.fillna(0, inplace=True)

    pre  = panel.index < TREATMENT_DATE
    post = panel.index >= TREATMENT_DATE

    weights    = optimize_weights(panel.loc[pre,'Treated'].values,
                                   panel.loc[pre, donors].values)
    synthetic  = panel[donors].values.dot(weights)
    panel['Synthetic'] = synthetic

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.fill_between(panel.index, panel['Treated'], panel['Synthetic'],
                    where=post, alpha=0.12, color=PALETTE["treated"], label='_nolegend_')

    ax.plot(panel.index, panel['Synthetic'],
            color=PALETTE["synthetic"], linewidth=2.2,
            linestyle='--', label='Synthetic Maharashtra')
    ax.plot(panel.index, panel['Treated'],
            color=PALETTE["treated"], linewidth=2.8,
            label='Actual Maharashtra')

    ax.axvline(TREATMENT_DATE, color=PALETTE["policy"],
               linestyle=':', linewidth=1.8, label='Policy Implementation (May 2025)')

    # Annotate the gap
    last_post = panel.loc[post].index[-1]
    actual_v  = panel.loc[last_post, 'Treated']
    synth_v   = panel.loc[last_post, 'Synthetic']
    ax.annotate('', xy=(last_post, actual_v), xytext=(last_post, synth_v),
                arrowprops=dict(arrowstyle='<->', color='#636E72', lw=1.4))
    ax.text(last_post + pd.Timedelta(days=8), (actual_v+synth_v)/2,
            f"Gap: {actual_v-synth_v:+.2f}pp",
            fontsize=9, color="#636E72", va='center')

    # Donor weights annotation
    weight_text = "Synthetic weights:\n" + "\n".join(
        f"  {STATE_LABELS[d]}: {w:.3f}"
        for d, w in zip(donors, weights) if w > 0.005)
    ax.text(0.02, 0.97, weight_text, transform=ax.transAxes,
            fontsize=8, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F0F4F8', ec='#B2BEC3', alpha=0.9))

    ate  = panel.loc[post, 'Treated'].mean() - panel.loc[post, 'Synthetic'].mean()
    rmspe = np.sqrt(np.mean((panel.loc[pre,'Treated'] - panel.loc[pre,'Synthetic'])**2))

    ax.set_title(f"Figure 2: Actual vs Synthetic Control – EV Penetration Rate\n"
                 f"Pre-treatment RMSPE: {rmspe:.4f}  |  Post-policy ATE: {ate:+.4f} pp",
                 fontsize=13, fontweight='bold', pad=12, color=PALETTE["text"])
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("EV Penetration Rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.grid(True, axis='y', linestyle='--')
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_scm_counterfactual.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 2 saved → {out}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – SCM Gap Plot with In-Space Placebos
# ────────────────────────────────────────────────────────────────────────────

def fig3_placebo_gap(df):
    donors = [s for s in df['state'].unique() if s != TREATED_STATE]

    df_treated = df[df['state'] == TREATED_STATE][['date','ev_penetration_rate']].copy()
    df_treated.rename(columns={'ev_penetration_rate':'Treated'}, inplace=True)
    df_donor   = df[df['state'].isin(donors)].pivot(
        index='date', columns='state', values='ev_penetration_rate')

    panel = pd.merge(df_treated, df_donor, on='date', how='inner').set_index('date')
    panel.ffill(inplace=True); panel.bfill(inplace=True); panel.fillna(0, inplace=True)

    pre  = panel.index < TREATMENT_DATE
    post = panel.index >= TREATMENT_DATE

    # True gap
    w_true    = optimize_weights(panel.loc[pre,'Treated'].values, panel.loc[pre,donors].values)
    synth_true = panel[donors].values.dot(w_true)
    gap_true   = panel['Treated'].values - synth_true
    rmspe_true = np.sqrt(np.mean(gap_true[pre]**2))
    ate_true   = gap_true[post].mean()

    # Placebo gaps
    placebo_gaps = {}
    placebo_ates = []
    for donor in donors:
        pool = [d for d in donors if d != donor]
        X_p  = panel.loc[pre, donor].values
        X_d  = panel.loc[pre, pool].values
        w_p  = optimize_weights(X_p, X_d)
        g    = panel[donor].values - panel[pool].values.dot(w_p)
        rmspe_p = np.sqrt(np.mean(g[pre]**2))
        if rmspe_p < 5 * rmspe_true:
            placebo_gaps[donor] = g
            placebo_ates.append(g[post].mean())

    p_val = np.mean(np.array(placebo_ates) <= ate_true)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, (donor, gap) in enumerate(placebo_gaps.items()):
        ax.plot(panel.index, gap,
                color=DONOR_COLORS[i % len(DONOR_COLORS)], alpha=0.35,
                linewidth=1.2, label=STATE_LABELS[donor])

    ax.plot(panel.index, gap_true,
            color=PALETTE["treated"], linewidth=3,
            label=f"Maharashtra  (ATE = {ate_true:+.4f} pp, p = {p_val:.3f})", zorder=5)

    ax.axvline(TREATMENT_DATE, color=PALETTE["policy"],
               linestyle=':', linewidth=1.8, label='Policy Implementation (May 2025)')
    ax.axhline(0, color='#636E72', linewidth=0.9, linestyle='--')

    ax.set_title(f"Figure 3: SCM In-Space Placebo Test\n"
                 f"Maharashtra vs {len(placebo_gaps)} Donor State Placebos  |  "
                 f"Pseudo p-value = {p_val:.3f}",
                 fontsize=13, fontweight='bold', pad=12, color=PALETTE["text"])
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Treatment Effect Gap (pp)", fontsize=11)
    ax.grid(True, axis='y', linestyle='--')
    ax.legend(fontsize=8.5, framealpha=0.9, ncol=2)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_placebo_gap.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 3 saved → {out}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – EV Penetration Rate: Maharashtra Pre vs Post (Bar)
# ────────────────────────────────────────────────────────────────────────────

def fig4_maharashtra_pre_post(df):
    mh  = df[df['state'] == TREATED_STATE].sort_values('date').copy()
    pre = mh[mh['date'] < TREATMENT_DATE]
    post= mh[mh['date'] >= TREATMENT_DATE]

    fig, ax = plt.subplots(figsize=(12, 5))

    bar_colors = [PALETTE["donor"]] * len(pre) + [PALETTE["treated"]] * len(post)
    all_vals   = pd.concat([pre['ev_penetration_rate'], post['ev_penetration_rate']])
    all_dates  = pd.concat([pre['date'], post['date']])

    ax.bar(all_dates, all_vals, color=bar_colors, width=25, edgecolor='white', linewidth=0.4)
    ax.axvline(TREATMENT_DATE, color=PALETTE["policy"],
               linestyle='--', linewidth=2, label='Policy Implementation (May 2025)')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=PALETTE["donor"],   label='Pre-Policy Period (Jan 2022 – Apr 2025)'),
        Patch(color=PALETTE["treated"], label='Post-Policy Period (May 2025 – Jun 2026)'),
    ], fontsize=10, framealpha=0.9)

    pre_mean  = pre['ev_penetration_rate'].mean()
    post_mean = post['ev_penetration_rate'].mean()
    ax.axhline(pre_mean,  color=PALETTE["donor"],    linestyle=':', linewidth=1.5,
               label=f'Pre-mean: {pre_mean:.2f}%')
    ax.axhline(post_mean, color=PALETTE["treated"],  linestyle=':', linewidth=1.5,
               label=f'Post-mean: {post_mean:.2f}%')

    ax.set_title("Figure 4: Maharashtra EV Penetration Rate – Monthly Bars Pre vs Post Policy",
                 fontsize=13, fontweight='bold', pad=12, color=PALETTE["text"])
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("EV Penetration Rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.grid(True, axis='y', linestyle='--')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_maharashtra_pre_post.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 4 saved → {out}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Covariate Balance Table (Pre-treatment mean comparison)
# ────────────────────────────────────────────────────────────────────────────

def fig5_balance_table(df):
    pre = df[df['date'] < TREATMENT_DATE].copy()
    mh_pre  = pre[pre['state'] == TREATED_STATE]
    don_pre = pre[pre['state'] != TREATED_STATE]

    metrics = {
        'EV Penetration Rate (%)': ('ev_penetration_rate', '{:.3f}'),
        'EV Registrations (000s)': ('ev_registrations',    '{:.1f}'),
        'Total Registrations (000s)': ('total_registrations', '{:.1f}'),
    }

    rows = []
    for label, (col, fmt) in metrics.items():
        mh_mean  = mh_pre[col].mean()
        don_mean = don_pre.groupby('state')[col].mean().mean()
        diff     = mh_mean - don_mean
        rows.append([label,
                     fmt.format(mh_mean if 'Reg' not in label else mh_mean/1000),
                     fmt.format(don_mean if 'Reg' not in label else don_mean/1000),
                     f"{diff/don_mean*100:+.1f}%"])

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=['Variable', 'Maharashtra', 'Donor Pool\n(Mean)', 'Relative\nDifference'],
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2D3436')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F0F4F8')
        cell.set_edgecolor('#DFE6E9')
        cell.set_linewidth(0.5)

    ax.set_title("Table 1: Pre-Treatment Covariate Balance (Jan 2022 – Apr 2025)",
                 fontsize=13, fontweight='bold', pad=16, color=PALETTE["text"])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_balance_table.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 5 (Table 1) saved → {out}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – EV Growth: All States Bar Chart (2022 vs 2026-H1 comparison)
# ────────────────────────────────────────────────────────────────────────────

def fig6_state_comparison_bar(df):
    # Compare average EV penetration: 2022 vs 2026 H1
    yr_2022 = df[df['year'] == 2022].groupby('state')['ev_penetration_rate'].mean()
    yr_2026 = df[df['year'] == 2026].groupby('state')['ev_penetration_rate'].mean()

    compare = pd.DataFrame({'2022': yr_2022, '2026 (H1)': yr_2026}).dropna().sort_values('2026 (H1)', ascending=True)
    states_clean = [STATE_LABELS[s] for s in compare.index]

    x = np.arange(len(compare))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.barh(x - w/2, compare['2022'],     w, color='#ADB5BD', label='2022 Avg.', edgecolor='white')
    b2 = ax.barh(x + w/2, compare['2026 (H1)'], w,
                 color=[PALETTE["treated"] if s == 'Maharashtra' else PALETTE["synthetic"]
                        for s in compare.index],
                 label='2026 H1 Avg.', edgecolor='white')

    ax.set_yticks(x)
    ax.set_yticklabels(states_clean, fontsize=11)
    ax.set_xlabel("Average EV Penetration Rate (%)", fontsize=11)
    ax.set_title("Figure 6: EV Penetration Rate by State — 2022 vs 2026 (H1)",
                 fontsize=13, fontweight='bold', pad=12, color=PALETTE["text"])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.grid(True, axis='x', linestyle='--')
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig6_state_bar_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 6 saved → {out}")

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating all publication-grade figures...\n")
    df = load_panel()
    fig1_all_state_trends(df)
    fig2_scm_counterfactual(df)
    fig3_placebo_gap(df)
    fig4_maharashtra_pre_post(df)
    fig5_balance_table(df)
    fig6_state_comparison_bar(df)
    print(f"\n✅ All 6 figures saved to {FIG_DIR}")
