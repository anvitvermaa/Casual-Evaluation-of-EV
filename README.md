<div align="center">
  <h1>Maharashtra EV Policy 2025: A Causal Evaluation</h1>
  <p><strong>Estimating the Short-Run Effect of the Maharashtra EV Subsidy using Synthetic Difference-in-Differences</strong></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Polars](https://img.shields.io/badge/Data_Engine-Polars-orange.svg)](https://pola.rs/)
  [![DuckDB](https://img.shields.io/badge/SQL_Joins-DuckDB-yellow.svg)](https://duckdb.org/)
  [![SynthDID](https://img.shields.io/badge/Causal_Inference-SynthDiD-success.svg)](https://github.com/synth-inference/synthdid)
</div>

---

## Results (Clean Run — Battery-EV Only Outcome)

| Specification | Donors | Pre-periods | Post-periods | ATT (pp) | Placebo SE | t-stat | 95% CI | Significant |
|--------------|--------|-------------|--------------|----------|-----------|--------|--------|-------------|
| Main Model (N=16) | 15 | 40 | 14 | +0.0347 | 1.1647 | 0.030 | [−2.25, +2.32] | No |
| Donut Hole (N=11) | 10 | 40 | 14 | +0.2889 | 1.6546 | 0.175 | [−2.95, +3.53] | No |

**Finding:** After correcting the outcome variable to include only battery-electric vehicles (ELECTRIC(BOV), ELECTRIC), the policy shows **no statistically detectable short-run effect** on EV adoption in either specification. The ATT is positive but economically negligible and statistically indistinguishable from zero.

> **Important note on a previous claim:** An earlier version of this project reported a statistically significant −0.943 pp decline. That result was produced with `STRONG HYBRID EV` included in the EV outcome. Strong hybrids surged nationally in 2024-25 due to new model launches entirely unrelated to the Maharashtra subsidy — contaminating the outcome variable right at the treatment window. Removing them corrects the measurement. The null result above is the accurate finding.

---

## What This Is

This project is a rigorous, end-to-end data engineering and econometric pipeline that evaluates whether the **Maharashtra EV Subsidy Policy (2025)** produced a genuine causal increase in battery-electric vehicle adoption across a balanced macro-state panel.

Instead of a naive before/after comparison, the project uses **Synthetic Difference-in-Differences (SDiD)**. L2 Ridge Regularization is applied to 15 donor states to construct a "Synthetic Maharashtra"—a weighted counterfactual trajectory calibrated to match Maharashtra's pre-treatment trend. The ATT isolates the true policy effect by comparing the observed post-treatment trajectory against this counterfactual.

---

## What the Null Result Actually Means

A null result is **not a failure** — it is a scientifically valid finding. The policy shows no detectable effect in the short-run (13 months post-treatment). There are several credible economic explanations:

1. **Supply-side constraints:** Subsidy demand may be limited by charging infrastructure rather than purchase price. If there are not enough public chargers, the subsidy alone will not drive additional purchases.
2. **Policy awareness lag:** Many consumers may not yet be aware of the state-level subsidy, especially with the national FAME-II expiration creating market uncertainty around the same time.
3. **Short post-window:** 13 months of post-treatment data may be insufficient for a durable behavioural response to become visible in aggregate state-level penetration rates.

The null result is honest and defensible — it is meaningfully different from "the policy failed."

---

## Panel Summary

| Metric | Value |
|--------|-------|
| Panel dimensions | N = 16 states, T = 54 months (Jan 2022 – Jun 2026) |
| Total observations | 864 (perfectly balanced — verified by hard assertion) |
| Treatment date | May 2025 (Maharashtra EV Subsidy Policy gazette notification: May 23, 2025) |
| Outcome variable | EV penetration rate (%) — battery-EVs only, strong hybrids excluded |
| Raw CSV files ingested | 80 (one per state × year, from MoRTH VAHAN dashboard) |
| Estimator | Synthdid (Arkhangelsky et al., AER 2021) with L2 Ridge Regularization |

---

## Architecture

```
data/raw/states/*.csv  (80 files)
      │
      ▼  [Stage 1 — Pandas]   src/data_ingestion/parse_vahan_data.py
      │   Unpivots cross-tabulated fuel grids. Battery-EV outcome only.
      │   Strong hybrids tracked separately as diagnostic column.
      │
vahan_fuel_panel.parquet
      │
      ▼  [Stage 2 — DuckDB]   src/data_engineering/duckdb_joins.py
      │   In-memory SQL normalization and schema enforcement.
      │
      ▼  [Stage 3 — Polars]   src/features/polars_transform.py
      │   Lazy evaluation, treatment indicators, hard balance assertion.
      │
final_state_feature_matrix_main.parquet  (864 rows, 16 × 54)
      │
      ▼  [Stage 4 — SynthDiD]  src/models/synthetic_control.py
          Dual specification: Main (N=16) + Spatial Donut Hole (N=11).
          Placebo-bootstrapped SE (200 replications). LaTeX table output.
```

**Robustness:** Two specifications run automatically. The Spatial Donut Hole drops Maharashtra's 5 bordering states (Gujarat, MP, Chhattisgarh, Telangana, Karnataka) to test for cross-border SUTVA spillovers. Both return null results with overlapping confidence intervals.

---

## Execution

### Setup
```bash
conda env create -f environment.yml
conda activate ev-policy-sdid
```

### Run the Full Pipeline
```bash
make run-sdid
```

### Key Source Files

| File | What it does |
|------|-------------|
| `config/settings.py` | Treatment date, state list, all paths |
| `src/data_ingestion/parse_vahan_data.py` | CSV parser — battery-EV extraction |
| `src/features/polars_transform.py` | Lazy Polars pipeline + balance assertion |
| `src/models/synthetic_control.py` | SDiD estimator, dual-spec, LaTeX output |
| `src/data_engineering/duckdb_joins.py` | DuckDB in-memory SQL normalization |

---

## Reference
Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088–4118.
