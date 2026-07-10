# API Reference

Base URL locally:

```text
http://127.0.0.1:8000
```

## GET `/api/health`

Returns service status and whether the model artifact is ready.

## GET `/api/report`

Returns the generated model report used by the dashboard.

## GET `/api/schema`

Returns the scoring schema, including the 95 required numeric feature columns.

## GET `/api/sample?limit=5`

Returns sample scoring rows from `data/prediction_template.csv`.

## GET `/api/audit/summary`

Returns prediction volume and risk-label counts from the local audit log.

## POST `/api/drift`

Compares a JSON or CSV scoring batch against the training baseline and returns the most shifted features.

## POST `/api/analyze-any`

Accepts a general company CSV and returns a best-effort business-risk analysis. This endpoint supports common sales, revenue, profit, expense, cash, debt, asset, liability, customer, and date columns. It does not require the strict 95-column bankruptcy schema, but confidence depends on the financial signals available.

## GET `/api/model-card`

Returns the project model card as Markdown.

## POST `/api/predict`

Scores JSON rows or an uploaded CSV.

## POST `/api/predict.csv`

Scores JSON rows or an uploaded CSV and returns a downloadable CSV.

JSON body:

```json
{
  "rows": [
    {
      "ROA(C) before interest and depreciation before interest": 0.37
    }
  ]
}
```

CSV upload:

```bash
curl -F "file=@data/prediction_template.csv" http://127.0.0.1:8000/api/predict
```

Response fields:

- `bankruptcy_probability`
- `risk_label`
- `prediction`
- `risk_counts`
- `scored_at`
