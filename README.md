# Bankruptcy Prediction System

This project has been upgraded from a notebook-only ML experiment into a deployable bankruptcy risk platform with:

- reusable source code under `src/bankruptcy_prediction`
- a training pipeline with SMOTE, scaling, multiple model candidates, threshold tuning, and model selection
- generated model artifacts in `models/`
- generated model reports in `reports/`
- a browser dashboard in `app/`
- CSV scoring for new companies
- a production HTTP server with `/api/health`, `/api/report`, `/api/sample`, and `/api/predict`
- schema validation, upload limits, numeric input checks, and prediction audit logging
- drift monitoring against the training baseline and downloadable scoring CSVs
- model governance docs, API docs, runbook, and CI workflow
- Render, Vercel, Docker, and Procfile deployment assets
- unit tests for dataset and inference contracts

## Dataset

The default dataset is `data/data.csv`. It contains 6,819 companies, 95 financial features, and the target column `Bankrupt?`.

## Train the Model

For a fast run:

```powershell
.\scripts\train_quick.ps1
```

For a full run with permutation importance:

```powershell
.\scripts\train.ps1
```

Training writes:

- `models/bankruptcy_risk_pipeline.joblib`
- `models/model_metadata.json`
- `reports/model_report.json`
- `reports/model_report.csv`
- `data/prediction_template.csv`

## Score New Data

Use the generated template to see the required 95 feature columns:

```powershell
.\scripts\predict.ps1 -InputCsv data\prediction_template.csv -OutputCsv reports\predictions.csv
```

The output includes `bankruptcy_probability`, `risk_label`, and `prediction`.

## Open Dashboard

After training:

```powershell
$env:PYTHONPATH="venv\Lib\site-packages;src"
python start.py
```

Then open:

```text
http://127.0.0.1:8000
```

The dashboard includes model metrics, a leaderboard, feature drivers, dataset profile, sample scoring, and CSV upload scoring.

## Industry-Level Operations

The app now includes production-facing safeguards:

- `/api/schema` publishes the exact 95-column scoring contract
- `/api/audit/summary` summarizes prediction traffic and risk distribution
- `/api/model-card` exposes model purpose, metrics, limitations, and monitoring recommendations
- `/api/drift` compares incoming scoring batches against the training baseline
- `/api/predict.csv` exports scored rows as CSV
- request size and row count limits protect the API from oversized uploads
- input validation rejects missing, non-numeric, missing-value, and infinite-value records
- `.github/workflows/ci.yml` trains a quick artifact, runs tests, and compiles the app on every push

Supporting docs:

- [MODEL_CARD.md](MODEL_CARD.md)
- [docs/API.md](docs/API.md)
- [docs/RUNBOOK.md](docs/RUNBOOK.md)

## Run Tests

```powershell
$env:PYTHONPATH="venv\Lib\site-packages;src"
python -m unittest discover -s tests
```

## Deploy

Render is the recommended host for the full ML app:

```text
render.yaml
```

Vercel support is included for a portfolio/demo deployment:

```text
vercel.json
api/index.py
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for exact deployment steps.

## Project Layout

```text
app/                         Browser dashboard
api/                         Vercel Python API adapter
data/                        Raw dataset and generated prediction template
models/                      Trained pipeline and metadata
reports/                     Model leaderboard and dashboard report
scripts/                     PowerShell workflow commands
src/bankruptcy_prediction/   Production ML package
tests/                       Dataset and inference tests
ML Project Final.ipynb       Original notebook
render.yaml                  Render deployment blueprint
vercel.json                  Vercel deployment config
Dockerfile                   Container deployment option
```
