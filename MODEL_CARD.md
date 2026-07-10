# Model Card: Bankruptcy Risk Platform

## Intended Use

This model supports early financial distress screening for companies. It should be used as a decision-support signal, not as the only basis for credit denial, investment decisions, lending actions, or legal conclusions.

## Training Data

- Source file: `data/data.csv`
- Rows: 6,819
- Features: 95 numeric financial indicators
- Target: `Bankrupt?`
- Positive class rate: approximately 3.23%

## Model Selection

The training pipeline compares multiple candidate models and selects the best model using average precision, recall, and balanced accuracy. The current quick-training artifact selected XGBoost.

## Current Validation Metrics

- ROC-AUC: 0.947
- Recall: 0.727
- Average precision: 0.505
- Tuned decision threshold: 0.15

## Operational Safeguards

- Strict schema validation for all prediction requests
- Numeric-only feature enforcement
- Missing and infinite value rejection
- Maximum request size and row-count limits
- Prediction audit log at `reports/prediction_audit.jsonl`
- Health, schema, report, and audit API endpoints
- Drift baseline checks for incoming scoring batches
- Downloadable CSV scoring output for analyst review
- Flexible business-health analysis for general company CSVs when trained-model inputs are unavailable

## Limitations

The dataset is highly imbalanced and may not represent every market, region, time period, company size, or accounting regime. Model outputs are probabilities from historical patterns, not guarantees of bankruptcy.

The generic CSV analyzer is heuristic and should be treated as a business-health screening layer. It is useful for sales or accounting CSVs, but its confidence depends on the available columns and it is not a substitute for the trained 95-feature bankruptcy model.

## Monitoring Recommendations

- Track prediction volume and risk-label distribution weekly
- Compare production feature distributions with training data
- Investigate high drift flags before trusting large batch scoring jobs
- Retrain when financial reporting regimes or macroeconomic conditions shift materially
- Review false positives and false negatives with domain experts
