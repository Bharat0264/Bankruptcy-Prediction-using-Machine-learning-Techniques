$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "venv\Lib\site-packages;src"
python -m bankruptcy_prediction.train --quick
