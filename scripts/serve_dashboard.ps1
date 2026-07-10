$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "venv\Lib\site-packages;src"
$env:PORT = "8000"
python start.py
