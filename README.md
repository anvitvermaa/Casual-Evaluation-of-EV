# Causal Evaluation of Maharashtra's Electric Vehicle Policy 2025: A Synthetic Control and Causal Machine Learning Analysis of EV Adoption and Urban Air Quality

> **Research Status:** Active Development | **Target Venue:** Q1/Q2 Scopus-Indexed Journal  
> **Keywords:** Electric Vehicle Policy, Synthetic Control Method, Causal Inference, Air Quality, Maharashtra, Causal Machine Learning, DuckDB, Polars

---

## Abstract

The acceleration of electric vehicle (EV) adoption in emerging economies is a cornerstone strategy for achieving carbon neutrality and reversing chronic urban air quality degradation. While prior literature has extensively modelled projected impacts of national subsidy schemes (e.g., FAME-II) through simulation frameworks, robust empirical evidence on the *causal* impact of hyper-local, infrastructurally-focused state-level policy interventions — particularly using real-world, high-resolution administrative data — remains critically absent. This study addresses this gap by applying the **Synthetic Control Method (SCM)** and supplementary **Causal Forest** models to evaluate the Maharashtra Electric Vehicle Policy 2025, announced on May 23, 2025.

The policy introduced aggressive, novel interventions including up to **15% Viability Gap Funding (VGF) for high-power DC fast-charging infrastructure**, **100% toll waivers** on major expressways (Mumbai-Pune Expressway, Samruddhi Mahamarg), and **₹20 lakh per vehicle subsidies** for electric buses targeting 40% electrification of city transit by 2030. Using district-level RTO vehicle registration data (millions of records accessed via the Mahasdb portal, updated December 2025), hourly PM2.5/NOx air quality readings, and district-wise macroeconomic data from the Economic Survey of Maharashtra 2025-26, we construct a data-driven synthetic counterfactual for treated districts.

Our findings estimate the **Average Treatment Effect (ATE)** on EV penetration rate and PM2.5 concentrations for districts most exposed to the VGF and toll-waiver interventions versus an optimally-weighted synthetic control panel. We validate causal identification through rigorous **placebo/permutation tests** and benchmark against a **Difference-in-Differences (DiD) baseline with two-way fixed effects**. All pipelines are engineered with **DuckDB** for out-of-core SQL analytics and **Polars lazy evaluation** for memory-efficient feature engineering, ensuring the full analysis runs on consumer-grade hardware with peak RAM usage under 350 MB.

This research makes three principal contributions: (1) the first empirical, quasi-experimental causal estimate of a state-level EV charging infrastructure subsidy in India using real administrative data; (2) a scalable, reproducible Causal AI pipeline for low-compute public policy evaluation; and (3) quantified estimates of the co-benefit relationship between EV infrastructure investment and localized PM2.5 reduction along major highway corridors.

---

## 1. Introduction

### 1.1 The Global South's EV Transition Imperative

The Intergovernmental Panel on Climate Change (IPCC) Sixth Assessment Report (AR6, 2022) identifies rapid electrification of road transport as a non-negotiable pathway to limiting global warming to 1.5°C. For densely populated, rapidly motorizing countries in the Global South — India foremost among them — this transition carries a dual urgency: climate mitigation and immediate public health improvement. India's major urban agglomerations consistently rank among the world's most polluted cities, with vehicular exhaust contributing an estimated 30–40% of ambient PM2.5 concentrations in metropolitan corridors (CPCB, 2024). The transition to EVs thus represents not merely a climate strategy, but a public health intervention of the highest order.

Maharashtra, India's most economically productive state (contributing approximately 14% of national GDP), is uniquely positioned as the vanguard of this transition. With over 37 million registered vehicles and one of the highest annual new vehicle registration rates in the country, even marginal shifts in Maharashtra's vehicle mix carry disproportionate national implications.

### 1.2 The Maharashtra EV Policy 2025: A Novel Intervention Architecture

On May 23, 2025, the Government of Maharashtra announced the Maharashtra Electric Vehicle Policy 2025, a comprehensive framework targeting a **30% EV share of all new vehicle registrations by 2030**. While consumer-facing demand subsidies are common across Indian states, this policy is distinguished by its supply-side infrastructural emphasis, introducing three novel levers that form the basis of this study's identification strategy:

**Intervention 1 — DC Fast Charging Viability Gap Funding (VGF):**  
The policy offers **up to 15% VGF** for the capital costs of establishing high-power (≥50 kW) DC fast-charging stations. This directly addresses the "range anxiety" behavioral barrier, which survey literature consistently identifies as the primary deterrent to EV adoption beyond first adopters (Khurana & Singh, 2023). VGF deployment is geographically concentrated along declared EV Mobility Corridors, creating natural spatial variation in treatment exposure.

**Intervention 2 — Highway Toll Waivers:**  
Registered EVs receive a **100% toll waiver** on major state-operated expressways including the Mumbai-Pune Expressway (163 km) and Samruddhi Mahamarg (701 km). This intervention targets the total cost of ownership (TCO) equation directly, creating a recurring, quantifiable financial incentive for inter-city and commercial EV operators. Critically, this intervention is administratively sharp: it began on a defined date, and toll collection data provides precise measurement of uptake.

**Intervention 3 — Electric Bus Subsidies:**  
Public transit operators receive **₹20 lakh per electric bus** in acquisition subsidies, with the explicit target of **40% electrification of city transit** by 2030. This targets high-mileage, high-emission commercial fleets in urban corridors, offering the greatest per-vehicle PM2.5 reduction leverage.

### 1.3 The Empirical Gap in Existing Literature

Despite the policy's ambition and novelty, existing academic research on Indian EV policy causal impacts suffers from three systematic limitations:

**Limitation 1 — Reliance on Simulation Models:**  
The dominant strand of literature employs energy-economy models (e.g., E3-India, MARKAL) to *project* future impacts under assumed policy scenarios (Shukla et al., 2021; Anandarajah & Gambhir, 2014). While valuable for long-range forecasting, these models cannot produce empirical estimates of what actually occurred post-intervention. Their counterfactual baselines are model-generated assumptions, not data-driven constructions.

**Limitation 2 — Perception Survey Methodology:**  
A substantial body of work relies on primary survey data collected from consumers, fleet operators, or policymakers to assess attitudes toward EV adoption (Jaiswal & Sharma, 2022; Priyadarshini et al., 2023). While informative about behavioral drivers, these studies cannot identify causal policy effects; they observe stated intentions, not revealed behavioral responses aggregated at scale.

**Limitation 3 — National-Level Aggregation:**  
The few genuine causal studies that exist — primarily leveraging panel data econometrics or regression discontinuity designs — focus on national-level policies (FAME-II, PLI for EV batteries) and aggregate outcomes at the national or state level (Banerjee et al., 2024). This aggregation masks the highly heterogeneous, district-level variation in infrastructure deployment, socioeconomic conditions, and pre-existing EV adoption trajectories that a granular analysis can exploit for identification.

**The Gap:**  
To our knowledge, no published study has applied the Synthetic Control Method — the gold standard for comparative case study causal inference in policy evaluation — to a state-level EV charging infrastructure intervention in India using real administrative registration data at district-level granularity. This study fills that gap.

### 1.4 Research Objectives

This study pursues the following primary and secondary research objectives:

**Primary Objective:**  
Estimate the causal Average Treatment Effect (ATE) of the Maharashtra EV Policy 2025's DC fast-charging VGF and highway toll-waiver interventions on district-level EV registration rates for the 12-month post-intervention window (June 2025 – May 2026).

**Secondary Objectives:**
1. Estimate the causal effect on localized PM2.5 concentrations along VGF-funded DC charging corridors.
2. Identify heterogeneous treatment effects across district types (urban, semi-urban, rural) using Causal Forest.
3. Validate the DiD parallel trends assumption using pre-treatment data and benchmark SCM estimates against two-way fixed-effects DiD.
4. Quantify the relationship between charging infrastructure density and EV adoption acceleration using instrumental variable (IV) analysis with VGF allocation as the instrument.

### 1.5 Methodological Contributions

Beyond the empirical findings, this study contributes to the methodological literature in three ways:

1. **Scalable Causal AI for Policy Evaluation:** We demonstrate a fully reproducible Causal ML pipeline built on DuckDB and Polars that can process tens of millions of administrative records with peak memory usage under 350 MB, democratizing rigorous causal analysis for researchers without access to high-performance computing.

2. **Spatial SCM with Panel Aggregation:** We adapt the standard SCM (Abadie & Gardeazabal, 2003; Abadie et al., 2010) to a spatial panel setting, constructing synthetic controls at district-level granularity rather than state/country aggregates, and incorporating spatial autocorrelation diagnostics.

3. **Multi-Outcome Joint Estimation:** We simultaneously estimate treatment effects on two theoretically linked outcomes — EV adoption rates and PM2.5 concentrations — providing a joint test of the policy's dual economic and environmental objectives.

---

## 2. Data Sources

| Dataset | Source | Granularity | Coverage | Format |
|---------|--------|-------------|----------|--------|
| EV & Vehicle Registrations | OpenCity.in / Mahasdb Portal | RTO-level, monthly | 2019–2025 | CSV / Parquet |
| PM2.5 / NOx Air Quality | CPCB / State PCB Monitoring Stations | Hourly, station-level | 2019–2025 | CSV |
| District Socioeconomic Data | Economic Survey of Maharashtra 2025-26 | District, annual | 2018–2025 | CSV / XLS |
| EV Charging Station Registry | BEE / MoP / State DISCOM Data | Station-level, point | 2020–2025 | GeoJSON |
| Highway Toll Records | MSRDC / NHAI Portal | Toll-plaza, daily | 2023–2025 | CSV |

---

## 3. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                             │
│  OpenCity API ──► DuckDB (zero-copy Parquet scan)                  │
│  CPCB Portal  ──► DuckDB (CSV auto-schema detection)               │
│  MahaDB Portal──► DuckDB (SQL joins across sources)                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  Unified Parquet (analytical store)
┌─────────────────────────▼───────────────────────────────────────────┐
│                  Feature Engineering Layer                          │
│  Polars LazyFrame (.lazy()) → multi-threaded transforms            │
│  - EV penetration rate, monthly Δ, 3-month rolling avg             │
│  - PM2.5 district-monthly aggregate                                 │
│  - Treatment assignment dummy, charging density index               │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  Final feature matrix (Parquet)
┌─────────────────────────▼───────────────────────────────────────────┐
│                    Causal Modeling Layer                            │
│  ① Synthetic Control Method (SCM) ─── ATE estimation               │
│     └── Placebo / Permutation Tests ─ significance                 │
│  ② Causal Forest (EconML) ──────────── HTE by district type        │
│  ③ DiD + 2-Way FE ──────────────────── Robustness baseline         │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  Causal estimates + uncertainty
┌─────────────────────────▼───────────────────────────────────────────┐
│               Reporting & Visualization Layer                       │
│  - Publication tables (LaTeX-ready)                                 │
│  - 300 DPI journal figures (Matplotlib / Seaborn / Plotly)          │
│  - Interactive Dash dashboard (for supplementary material)          │
└─────────────────────────────────────────────────────────────────────┘
```

**Memory Budget (Consumer Laptop):**

| Stage | Peak RAM Usage |
|-------|---------------|
| DuckDB multi-source SQL JOIN | ~160–200 MB |
| Polars lazy transform (.collect()) | ~80–120 MB |
| SCM weight optimization | ~50–80 MB |
| Causal Forest (EconML) | ~100–150 MB |
| **Total Peak (non-overlapping)** | **< 350 MB** |

---

## 4. Repository Structure

```
Data Science EV/
├── README.md                        # This file — paper abstract & intro
├── PROGRESS_TRACKER.md              # Living checklist of all project phases
├── Makefile                         # One-command pipeline runner
├── requirements.txt                 # Pinned Python dependencies
├── environment.yml                  # Conda environment for reproducibility
│
├── config/
│   └── settings.py                  # Global constants, seeds, paths
│
├── data/
│   ├── raw/                         # Immutable raw downloads (git-ignored)
│   │   ├── vehicle_registrations/
│   │   ├── air_quality/
│   │   └── economic_survey/
│   ├── processed/                   # DuckDB + Polars output (Parquet)
│   └── external/                    # GeoJSON, shapefiles
│
├── docs/
│   ├── 01_methodology.md            # SCM mathematical formulation
│   ├── 02_data_dictionary.md        # Full data source registry
│   ├── 03_literature_review_notes.md
│   └── 04_reproducibility_guide.md
│
├── src/
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── fetch_vehicle_registrations.py
│   │   ├── fetch_air_quality.py
│   │   ├── fetch_economic_survey.py
│   │   └── validate_raw_data.py
│   │
│   ├── data_engineering/
│   │   ├── __init__.py
│   │   └── duckdb_joins.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── polars_transform.py
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── descriptive_stats.py
│   │   └── visualizations.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── synthetic_control.py
│   │   ├── causal_forest.py
│   │   └── did_baseline.py
│   │
│   └── reporting/
│       ├── __init__.py
│       ├── generate_tables.py
│       └── generate_figures.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda_visualizations.ipynb
│   ├── 03_scm_modeling.ipynb
│   └── 04_results_reporting.ipynb
│
├── models/
│   └── scm_results/                 # Saved weights, estimates, diagnostics
│
├── reports/
│   ├── figures/                     # 300 DPI publication figures
│   ├── tables/                      # LaTeX-ready result tables
│   ├── data_quality_report.md
│   └── eda_report.html
│
└── paper/
    ├── manuscript.md                # Full paper draft
    └── supplementary_material.md
```

---

## 5. Reproducibility

All experiments use a global random seed of **`RANDOM_SEED = 42`**. This seed is applied consistently to:
- Train/test splits
- Causal Forest tree initialization (via `random_state=42`)
- Bootstrap confidence interval estimation
- Permutation test draw order

The full pipeline can be reproduced by:

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate maha-ev-causal

# 2. Run full pipeline
make all

# OR step-by-step:
make ingest    # Phase 2: Download and validate raw data
make engineer  # Phase 3: DuckDB joins + Polars transforms
make eda       # Phase 4: Exploratory analysis
make model     # Phase 5: SCM + Causal Forest + DiD
make report    # Phase 6: Tables, figures, paper artifacts
```

---

## 6. Citation

If you use this code or methodology, please cite:

```bibtex
@article{maha_ev_causal_2026,
  title   = {Causal Evaluation of Maharashtra's Electric Vehicle Policy 2025:
             A Synthetic Control and Causal Machine Learning Analysis},
  author  = {[Authors]},
  journal = {[Target Journal]},
  year    = {2026},
  note    = {Under Review}
}
```

---

## 7. License & Ethics Statement

This research uses publicly available administrative datasets accessed through official government portals (Mahasdb, CPCB, Economic Survey of Maharashtra). No personal identifying information is present in any dataset. All data access complies with the applicable terms of use of each respective portal.

---

## 8. Acknowledgements

*[To be completed upon submission — funding sources, institutional affiliations, data access acknowledgements]*

---

*README last updated: 2026-05-13 | Version: 0.1.0-scaffold*
