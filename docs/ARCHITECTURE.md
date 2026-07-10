# Architecture

## Runtime

The app is a Python web service that serves a static dashboard and JSON/CSV API from one process.

```text
Browser Dashboard
  -> /api/report
  -> /api/predict
  -> /api/predict.csv
  -> /api/drift
  -> /api/audit/summary

Python Server
  -> validation
  -> model pipeline
  -> drift baseline
  -> audit log
```

## Core Modules

- `server.py`: HTTP routes and production entrypoint
- `validation.py`: input contract, row limits, numeric checks
- `modeling.py`: model candidates, SMOTE pipeline, threshold tuning
- `inference.py`: feature alignment and prediction
- `drift.py`: training baseline and batch drift comparison
- `audit.py`: JSONL prediction audit summaries
- `reporting.py`: model, metadata, report, and baseline artifacts

## Deployment

Render is preferred for the full ML application because it keeps a persistent Python process warm. Vercel is supported for a demo-style serverless deployment.
