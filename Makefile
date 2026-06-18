.PHONY: all setup ingest engineer eda model report clean

all: setup ingest engineer eda model report

setup:
	python config/settings.py
	@echo "Directory structure verified."

ingest: setup
	python src/data_ingestion/fetch_vehicle_registrations.py
	python src/data_ingestion/fetch_air_quality.py
	python src/data_ingestion/fetch_economic_survey.py
	python src/data_ingestion/validate_raw_data.py
	@echo "Phase 2: Data Ingestion completed."

engineer: ingest
	python src/data_engineering/duckdb_joins.py
	python src/features/polars_transform.py
	@echo "Phase 3: Data Engineering completed."

eda: engineer
	python src/eda/descriptive_stats.py
	python src/eda/visualizations.py
	@echo "Phase 4: EDA completed."

model: eda
	python src/models/synthetic_control.py
	python src/models/causal_forest.py
	python src/models/did_baseline.py
	@echo "Phase 5: Modeling completed."

report: model
	python src/reporting/generate_tables.py
	python src/reporting/generate_figures.py
	@echo "Phase 6: Reporting completed."

clean:
	rm -rf data/processed/*
	rm -rf models/scm_results/*
	rm -rf reports/figures/*
	rm -rf reports/tables/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleaned processed data, models, reports, and caches."

run-sdid:
	bash run_pipeline.sh
