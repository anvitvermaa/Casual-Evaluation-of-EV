# Causal Evaluation of EV Policy

This repository contains the data engineering and causal modeling pipeline for evaluating the impact of state-level EV subsidy policies using advanced causal inference methods (Synthetic Difference-in-Differences).

## Overview

**Project Context:** A rigorous quasi-experimental causal evaluation of the Maharashtra EV Subsidy Policy 2025 across a balanced macro-state panel of top vehicle-registering Indian states ($N=16$, $T=54$ months).

* **Uncovered the "Demand Displacement Paradox":** Mathematically isolated a null short-run demand signal using quasi-experimental causal analysis, driven by volatile national FAME-II subsidy expirations.
* **High-Performance Data Engineering:** Engineered an out-of-core ETL pipeline utilizing a Python AJAX scraper and Polars to lazily ingest, transform, and evaluate nearly 100 million API-sourced vehicle registrations across 54 months of macroscopic Vahan data.
* **Advanced Causal Architecture:** Pioneered a rigorous dual-specification causal architecture utilizing the Synthetic Difference-in-Differences (SDiD) estimator with L2 Ridge Regularization to construct unconfounded baseline counterfactuals.
* **Mathematical Robustness:** Designed advanced spatial robustness checks ("Donut Hole" specifications) and placebo bootstrap permutation tests to mathematically validate SUTVA compliance against cross-border arbitrage spillovers.

## Setup

1. Create the conda environment:
```bash
conda env create -f environment.yml
```
2. Activate the environment:
```bash
conda activate ev-policy-sdid
```

## Structure

*   `src/`: Contains the pipeline source code.
    *   `data/`: Data extraction and ingestion scripts.
    *   `features/`: DuckDB and Polars based feature engineering.
    *   `models/`: Causal modeling implementations (SDiD).
*   `Makefile`: Defines the build automation.

## Running the Pipeline

To run the pipeline locally (requires data):
```bash
make run-sdid
```
