# End-to-End ML Pipeline

A clean, reproducible machine learning pipeline covering the full data science workflow — from raw data generation through EDA, model training, and REST API serving.

Built as a portfolio project to demonstrate practical ML engineering skills.

---

## Architecture

```
projects/
├── config.yaml              ← single source of truth for all parameters
├── run_pipeline.py          ← runs the full pipeline with one command
│
├── 00_dataset_build/
│   └── dataset_build.py     ← generates synthetic regression data
│
├── 01_eda/
│   ├── eda.py               ← feature selection, VIF, outlier treatment
│   └── eda_visualizations/  ← correlation heatmap, scatter plots, distributions
│
├── 02_ml_pipeline/
│   └── pipeline.py          ← cross-validation, model selection, evaluation
│
└── 03_mlops/
    └── serve.py             ← FastAPI inference endpoint
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/norenlivingston/datascienceportfolio.git
cd datascienceportfolio/projects
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py
```

Sample output:

```
2026-04-09 12:00:00  INFO     ───────────────────────────────────────────────────────
2026-04-09 12:00:00  INFO       STAGE 1 / 3 — Dataset Build
2026-04-09 12:00:01  INFO     Saved raw dataset → data/synthetic_regression_dataset.csv
2026-04-09 12:00:01  INFO       STAGE 2 / 3 — Exploratory Data Analysis
2026-04-09 12:00:03  INFO     Top 8 features selected: ['Feature_5', ...]
2026-04-09 12:00:04  INFO       STAGE 3 / 3 — Model Training & Evaluation
2026-04-09 12:00:08  INFO       RandomForest          R² = 0.9991 ± 0.0002
2026-04-09 12:00:08  INFO       LinearRegression      R² = 0.9988 ± 0.0003
2026-04-09 12:00:08  INFO     Selected: RandomForest
2026-04-09 12:00:09  INFO     Hold-out → MAE=1.8  RMSE=2.4  R²=0.9991
```

---

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| **Dataset Build** | `00_dataset_build/dataset_build.py` | `data/synthetic_regression_dataset.csv` |
| **EDA** | `01_eda/eda.py` | `data/cleaned_synthetic_regression_dataset.csv`, plots |
| **Model Training** | `02_ml_pipeline/pipeline.py` | `best_model.pkl`, `metrics_log.json`, feature importance plot |
| **Serving** | `03_mlops/serve.py` | REST API at `localhost:8000` |

Each stage can be run independently as well as part of the full pipeline.

---

## Configuration

All parameters are in `projects/config.yaml`. No code changes needed to tune the pipeline:

```yaml
pipeline:
  cv_folds: 5           # k-fold cross-validation
  test_size: 0.2
  models:
    random_forest:
      n_estimators: 100
    linear_regression: {}

eda:
  top_n_features: 8
  outlier_method: "iqr"
  outlier_strategy: "iqr_bound"
```

---

## Inference API

Start the server (requires the pipeline to have been run first):

```bash
cd projects
python 03_mlops/serve.py
```

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Feature_3": 1.2, "Feature_7": -0.5, ...}}'
```

Interactive docs available at `http://localhost:8000/docs`.

---

## Skills Demonstrated

| Area | Tools / Techniques |
|---|---|
| **Data Engineering** | pandas, NumPy, reproducible data generation |
| **EDA** | Correlation analysis, VIF (multicollinearity), IQR outlier treatment, Seaborn / Matplotlib |
| **Machine Learning** | scikit-learn Pipelines, k-fold cross-validation, model comparison, feature importance |
| **MLOps** | Model serialisation (joblib), metrics logging, FastAPI + Uvicorn serving |
| **Software Engineering** | Config-driven design, modular functions, structured logging, clean repo |

---

## Stack

Python · scikit-learn · pandas · NumPy · FastAPI · Uvicorn · Pydantic · Matplotlib · Seaborn · statsmodels · PyYAML

---

Noren Livingston, M.S.
[norenlivingston@gmail.com](mailto:norenlivingston@gmail.com)
