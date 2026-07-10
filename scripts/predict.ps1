param(
  [Parameter(Mandatory=$true)][string]$InputCsv,
  [string]$OutputCsv = "reports\predictions.csv"
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "venv\Lib\site-packages;src"
python -m bankruptcy_prediction.predict --input $InputCsv --output $OutputCsv
