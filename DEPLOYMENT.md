# Deployment Guide

This project is ready for Render first and Vercel as a lighter alternative.

## Recommended: Render

Render is the best fit because the app serves a Python ML model and keeps the API warm as a web service.

1. Push this folder to a GitHub repository.
2. In Render, choose **New Web Service**.
3. Connect the repository.
4. Render will detect `render.yaml`.
5. Confirm these commands:

```bash
pip install -r requirements.txt && PYTHONPATH=src python -m bankruptcy_prediction.train --quick
```

```bash
PYTHONPATH=src python start.py
```

6. Deploy.

After deploy, open the Render URL. The root path redirects to:

```text
/app/dashboard.html
```

Useful production endpoints:

```text
/api/health
/api/report
/api/sample?limit=5
/api/predict
```

## Vercel

Vercel can serve the dashboard and Python serverless API through `api/index.py`, but ML packages can make cold starts and build size more fragile. Use it if you want a portfolio/demo deployment. Use Render if you want the more reliable ML app.

1. Push the project to GitHub.
2. Import the repo in Vercel.
3. Vercel will use `vercel.json`.
4. Keep the Python runtime on 3.11 when prompted.

## Local Production Smoke Test

```powershell
$env:PYTHONPATH="venv\Lib\site-packages;src"
$env:PORT="8000"
python start.py
```

Then open:

```text
http://127.0.0.1:8000
```

## CSV Prediction Contract

The uploaded CSV must contain the same 95 feature columns as:

```text
data/prediction_template.csv
```

The response includes:

- `bankruptcy_probability`
- `risk_label`
- `prediction`
