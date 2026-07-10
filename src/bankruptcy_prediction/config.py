from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
APP_DIR = ROOT_DIR / "app"

DEFAULT_DATA_PATH = DATA_DIR / "data.csv"
TARGET_COLUMN = "Bankrupt?"
RANDOM_STATE = 42

MODEL_PATH = MODEL_DIR / "bankruptcy_risk_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORT_PATH = REPORT_DIR / "model_report.json"
DRIFT_BASELINE_PATH = MODEL_DIR / "drift_baseline.json"
PREDICTION_TEMPLATE_PATH = DATA_DIR / "prediction_template.csv"
AUDIT_LOG_PATH = REPORT_DIR / "prediction_audit.jsonl"

MAX_ROWS_PER_REQUEST = 500
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
