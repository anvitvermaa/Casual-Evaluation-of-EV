<div align="center">
  <h1>⚡ Maharashtra EV Policy 2025: A Causal Evaluation</h1>
  <p><strong>Proving the "Demand Displacement Paradox" using Synthetic Difference-in-Differences</strong></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Polars](https://img.shields.io/badge/Data_Engine-Polars-orange.svg)](https://pola.rs/)
  [![DuckDB](https://img.shields.io/badge/Compute-DuckDB-yellow.svg)](https://duckdb.org/)
  [![SynthDID](https://img.shields.io/badge/Causal_Inference-SynthDiD-success.svg)](https://github.com/synth-inference/synthdid)
</div>

---

## 📊 The TL;DR (By the Numbers)
* **The Finding:** A **−0.943 percentage point drop** in short-term EV adoption ($p < 0.001$).
* **The Data:** Nearly **100 Million** raw vehicle registrations ingested.
* **The Scale:** A perfectly balanced macro-panel of **16 top Indian states** over **54 continuous months**.
* **The Math:** **Synthetic Difference-in-Differences (SDiD)** using L2 Ridge Regularization.

---

## 📖 What is this?
Governments often announce massive subsidies and claim success when sales naturally go up, confusing correlation with causation. This project is a rigorous, end-to-end data engineering and econometric pipeline built to evaluate whether the **Maharashtra EV Subsidy Policy (2025)** *actually* worked.

Instead of a basic A/B test, this project uses **Synthetic Difference-in-Differences (SDiD)**. By applying L2 Ridge Regularization to a massive dataset of 15 "donor" states, the algorithm mathematically constructs a "Synthetic Maharashtra"—a simulated baseline reality where the policy never existed. We then compare real Maharashtra to Synthetic Maharashtra to isolate the true causal effect.

## 🤯 What did I find? (The "Demand Displacement Paradox")
Most people assume subsidies immediately increase sales. This model mathematically proved the exact opposite happened in the short term. 

The policy resulted in a **statistically significant −0.943 pp *decline* in EV adoption**. 

**Why?** Because of the **Demand Displacement Paradox**. When the government published the gazette on May 23, 2025, consumers anticipated it. The most eager buyers rushed to purchase *before* the national FAME-II subsidies expired, while others decided to *wait* for the promised state-level charging infrastructure to actually be built. The result was a massive short-term demand crash exactly when the policy launched.

---

## 🏗️ The Tech Stack & Architecture

This isn't just an Excel spreadsheet. This is a highly optimized, out-of-core data pipeline built to handle macroeconomic scale without crashing your laptop's RAM.

* **Data Ingestion (`Pandas`):** A custom python scraper and parser that rips through highly unstructured, cross-tabulated government CSVs (from the MoRTH VAHAN dashboard) and unpivots them into a standardized, long-format SQL-ready structure.
* **Feature Engineering (`Polars` & `DuckDB`):** Uses Polars' multi-threaded Rust backend and lazy evaluation (`pl.scan_parquet`) to filter the raw data down to a perfectly balanced 16-state, 54-month macro-panel. It dynamically hacks datetime strings into continuous integer matrices required for complex math solvers.
* **Causal Modeling (`SynthDID`):** Feeds the perfect matrix into a Python port of the `synthdid` estimator, which leverages `scipy.optimize` to solve convex weighting problems via Ridge Regressions.
* **Validation:** Validated using "in-space placebo permutation tests" (pretending donor states received the treatment to ensure the model isn't just picking up noise).

---

## 🚀 How to use it

If you want to run the pipeline yourself and generate the ATE estimates and Plotly trajectory graphs:

### 1. Setup the Environment
You need `conda` installed. Run the following to install the exact dependencies (Polars, DuckDB, synthdid, etc):
```bash
conda env create -f environment.yml
conda activate ev-policy-sdid
```

### 2. Run the Full DAG Pipeline
The entire pipeline is connected via a single entry point. Running this command will sequentially trigger the ingestion parser, the Polars transformation engine, and the Causal model:
```bash
make run-sdid
```

### 3. Explore the Code
* `src/data_ingestion/parse_vahan_data.py`: See how messy CSVs are coerced and unpivoted.
* `src/features/polars_transform.py`: Check out the Polars lazy-evaluation query plans.
* `src/models/synthetic_control.py`: Read the econometric math and placebo permutation tests.
