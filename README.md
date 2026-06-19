# Causal Evaluation of EV Policy

This repository contains the data engineering and causal modeling pipeline for evaluating the impact of state-level EV subsidy policies using advanced causal inference methods (Synthetic Difference-in-Differences).

## Overview

The repository hosts the automated data ingestion, DuckDB-powered data processing, and causal inference modeling scripts used to construct a macro-state panel and estimate causal effects of policy interventions.

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
