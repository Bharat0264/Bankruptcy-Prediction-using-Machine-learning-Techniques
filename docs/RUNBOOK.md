# Production Runbook

## Start Locally

```powershell
$env:PYTHONPATH="venv\Lib\site-packages;src"
$env:PORT="8000"
python start.py
```

## Validate Health

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/health -UseBasicParsing
```

## Retrain

```powershell
$env:PYTHONPATH="venv\Lib\site-packages;src"
python -m bankruptcy_prediction.train --quick
```

## Check Audit Summary

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/audit/summary -UseBasicParsing
```

## Check Drift

```powershell
$sample = Invoke-WebRequest http://127.0.0.1:8000/api/sample?limit=5 -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/api/drift -Method Post -ContentType "application/json" -Body $sample.Content -UseBasicParsing
```

## Common Issues

- Missing model artifact: run training or verify Render build command completed.
- CSV upload rejected: compare the file columns against `/api/schema`.
- Drift baseline missing: retrain with `python -m bankruptcy_prediction.train --quick`.
- Slow cold start: prefer Render over Vercel for the full ML app.
