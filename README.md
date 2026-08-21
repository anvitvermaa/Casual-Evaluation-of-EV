<div align="center">
  <h1>Maharashtra EV Policy 2025: A Causal Evaluation</h1>
  <p><strong>Proving the "Demand Displacement Paradox" using Synthetic Difference-in-Differences</strong></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Polars](https://img.shields.io/badge/Data_Engine-Polars-orange.svg)](https://pola.rs/)
  [![DuckDB](https://img.shields.io/badge/SQL_Joins-DuckDB-yellow.svg)](https://duckdb.org/)
  [![SynthDID](https://img.shields.io/badge/Causal_Inference-SynthDiD-success.svg)](https://github.com/synth-inference/synthdid)
</div>

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Causal Finding (ATT)** | −0.943 pp (EV penetration rate) |
| **Statistical Validation** | Placebo SE, 200 replications |
| **Panel Dimensions** | N = 16 states, T = 54 months (Jan 2022 – Jun 2026) |
| **Treatment Date** | May 2025 (Maharashtra EV Subsidy Policy gazette notification) |
| **Raw CSV files ingested** | 80 fuel-type cross-tabulation files (one per state × year) |
| **Total registered vehicles represented** | ~34 million vehicles across all 80 files |
| **Final modeling matrix** | 864 rows (16 × 54 perfectly balanced panel) |
| **Estimator** | Synthdid (Arkhangelsky et al., AER 2021) with L2 Ridge Regularization |

---

## What This Is

Governments frequently announce subsidies and claim success when sales rise, conflating correlation with causation. This project is a rigorous, end-to-end data engineering and econometric pipeline that evaluates whether the **Maharashtra EV Subsidy Policy (2025)** produced a genuine causal increase in battery-electric vehicle adoption.

Instead of a naive before/after comparison, the project uses **Synthetic Difference-in-Differences (SDiD)**. L2 Ridge Regularization is applied to 15 donor states to construct a "Synthetic Maharashtra"—a weighted counterfactual trajectory calibrated to match Maharashtra's pre-treatment trend. The ATT isolates the true policy effect by comparing the observed trajectory against this counterfactual in the post-treatment window.

---

## Empirical Finding: The "Demand Displacement Paradox"

The model produces a statistically validated **Average Treatment Effect (ATT) of −0.943 percentage points** in EV penetration rate.

This is a counter-intuitive but economically coherent result. When the gazette notification was published on May 23, 2025, the market had already been anticipating a state-level subsidy for months — during which time the most price-sensitive buyers had already entered the market to capture the expiring national FAME-II subsidies. The post-announcement window was then simultaneously characterised by:

1. **Demand exhaustion** — the pre-announcement surge had pulled forward the existing eager buyer pool.
2. **Infrastructure wait-and-see** — buyers waiting for the promised public fast-charging rollout before committing.

The combined effect is a measurable short-run demand contraction at the exact moment of policy activation.

**Note on inference:** With N₀ = 15 donors and a rank-based placebo test, the minimum achievable p-value is approximately 1/16 ≈ 0.063. Statistical significance here is reported as a t-statistic derived from placebo-bootstrapped standard errors (200 replications), not a permutation p-value.

---

## Architecture

This is a multi-stage, out-of-core data engineering pipeline. Each stage is a separate process connected via the `Makefile` DAG.

```
data/raw/*.csv
      │
      ▼  [Stage 1 — Pandas]
parse_vahan_data.py        →   vahan_fuel_panel.parquet
      │
      ▼  [Stage 2 — DuckDB]
duckdb_joins.py            →   unified_state_dataset.parquet
      │
      ▼  [Stage 3 — Polars lazy eval]
polars_transform.py        →   final_state_feature_matrix_main.parquet
      │
      ▼  [Stage 4 — SynthDiD]
synthetic_control.py       →   ATT, SE, 95% CI, weight plots, LaTeX table
```

**Stage 1 (Pandas):** Parses 80 messy, cross-tabulated CSVs from the MoRTH VAHAN dashboard. Handles encoding fallbacks (UTF-8 → latin-1), strips Indian-format number commas (`1,23,456` → `123456`), and separates battery-EV registrations from Strong Hybrid registrations (the latter are tracked diagnostically but **excluded from the outcome**, as they are not policy-eligible and surged independently in 2024-25).

**Stage 2 (DuckDB):** Runs in-memory SQL joins to produce a unified state-level panel view, applying schema normalization.

**Stage 3 (Polars):** Lazy evaluation filters the panel to the 16 macro-states, computes the EV penetration rate (outcome variable), assigns treatment indicators, and runs a **hard-failing balance assertion** to guarantee the panel is exactly N × T before writing the modeling matrix.

**Stage 4 (SynthDiD):** Fits the `synthdid` estimator (Arkhangelsky et al., AER 2021) using optimized covariate projection (GSDP per capita, urbanization rate) and computes placebo-bootstrapped standard errors.

---

## Robustness Specifications

Two specifications are run in every pipeline execution:

| Specification | N (Donors) | Purpose |
|---------------|-----------|---------|
| Main Model | 15 | Full donor pool |
| Spatial Donut Hole | 10 | Bordering states (Gujarat, MP, Chhattisgarh, Telangana, Karnataka) excluded — tests for cross-border SUTVA spillovers |

---

## Execution & Replication

### Setup
```bash
conda env create -f environment.yml
conda activate ev-policy-sdid
```

### Run the Full Pipeline
```bash
make run-sdid
```

This sequentially executes all stages: ingestion → DuckDB joins → Polars transformation → SDiD estimation → figure generation.

### Explore the Source
| File | What it does |
|------|-------------|
| `config/settings.py` | Single source of truth: treatment date, state list, paths |
| `src/data_ingestion/parse_vahan_data.py` | Fuel CSV parser — the most brittle part of the pipeline |
| `src/features/polars_transform.py` | Lazy Polars pipeline + balance assertion |
| `src/models/synthetic_control.py` | SDiD estimator, dual-spec execution, LaTeX table generation |
| `src/data_engineering/duckdb_joins.py` | DuckDB in-memory SQL normalization |

---

## Reference
Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088–4118.
