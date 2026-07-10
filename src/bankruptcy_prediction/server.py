from __future__ import annotations

import cgi
import csv
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .audit import append_prediction_audit, audit_summary
from .config import MAX_UPLOAD_BYTES, METADATA_PATH, REPORT_PATH, ROOT_DIR
from .drift import drift_report
from .inference import load_model, predict_frame
from .train import main as train_main
from .validation import schema_payload


class BankruptcyRiskHandler(SimpleHTTPRequestHandler):
    server_version = "BankruptcyRiskServer/2.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def log_message(self, format, *args):
        if os.environ.get("QUIET_SERVER_LOGS") != "1":
            super().log_message(format, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._redirect("/app/dashboard.html")
            return
        if parsed.path == "/api/health":
            self._json({"status": "ok", "service": "bankruptcy-risk", "model_ready": self._model_ready()})
            return
        if parsed.path == "/api/report":
            self._send_report()
            return
        if parsed.path == "/api/schema":
            self._send_schema()
            return
        if parsed.path == "/api/audit/summary":
            self._json(audit_summary())
            return
        if parsed.path == "/api/model-card":
            self._send_model_card()
            return
        if parsed.path == "/api/sample":
            self._send_sample(parse_qs(parsed.query))
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            self._predict()
            return
        if parsed.path == "/api/predict.csv":
            self._predict(csv_response=True)
            return
        if parsed.path == "/api/drift":
            self._drift()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route")

    def _redirect(self, target: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", target)
        self.end_headers()

    def _json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_report(self) -> None:
        if not REPORT_PATH.exists():
            self._json({"error": "Report not found. Train the model first."}, HTTPStatus.NOT_FOUND)
            return
        self._json(json.loads(REPORT_PATH.read_text(encoding="utf-8")))

    def _send_schema(self) -> None:
        if not METADATA_PATH.exists():
            self._json({"error": "Model metadata not found. Train the model first."}, HTTPStatus.NOT_FOUND)
            return
        metadata = pd.read_json(METADATA_PATH, typ="series").to_dict()
        self._json(schema_payload(metadata["feature_names"]))

    def _send_model_card(self) -> None:
        card_path = ROOT_DIR / "MODEL_CARD.md"
        if not card_path.exists():
            self._json({"error": "MODEL_CARD.md not found."}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        body = card_path.read_bytes()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sample(self, query: dict[str, list[str]]) -> None:
        limit = int(query.get("limit", ["5"])[0])
        df = pd.read_csv(ROOT_DIR / "data" / "prediction_template.csv").head(max(1, min(limit, 25)))
        self._json({"rows": df.to_dict(orient="records")})

    def _predict(self, csv_response: bool = False) -> None:
        try:
            df = self._request_to_dataframe()
            pipeline, metadata = load_model()
            predictions = predict_frame(df, pipeline, metadata)
            columns = ["bankruptcy_probability", "risk_label", "prediction"]
            risk_counts = predictions["risk_label"].value_counts().to_dict()
            append_prediction_audit(len(predictions), metadata["model_name"], metadata["threshold"], risk_counts)
            if csv_response:
                self._csv("bankruptcy_predictions.csv", predictions.to_dict(orient="records"))
                return
            response = {
                "count": int(len(predictions)),
                "threshold": float(metadata["threshold"]),
                "model_name": metadata["model_name"],
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "risk_counts": risk_counts,
                "predictions": predictions[columns].to_dict(orient="records"),
                "preview": predictions.head(20).to_dict(orient="records"),
            }
            self._json(response)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _drift(self) -> None:
        try:
            df = self._request_to_dataframe()
            _, metadata = load_model()
            self._json(drift_report(df, metadata["feature_names"]))
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _csv(self, filename: str, rows: list[dict]) -> None:
        if not rows:
            self._json({"error": "No rows available to export."}, HTTPStatus.BAD_REQUEST)
            return
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        body = output.getvalue().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _request_to_dataframe(self) -> pd.DataFrame:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("No request body was provided.")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError(f"Request body exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")

        if content_type.startswith("application/json"):
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            rows = payload.get("rows", payload)
            return pd.DataFrame(rows)

        if content_type.startswith("multipart/form-data"):
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            file_item = form["file"] if "file" in form else None
            if file_item is None or not getattr(file_item, "file", None):
                raise ValueError("Upload a CSV file with form field name 'file'.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
                temp.write(file_item.file.read())
                temp_path = Path(temp.name)
            try:
                return pd.read_csv(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

        body = self.rfile.read(content_length)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            temp.write(body)
            temp_path = Path(temp.name)
        try:
            return pd.read_csv(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _model_ready(self) -> bool:
        return (ROOT_DIR / "models" / "bankruptcy_risk_pipeline.joblib").exists()


def ensure_artifacts() -> None:
    model_path = ROOT_DIR / "models" / "bankruptcy_risk_pipeline.joblib"
    metadata_path = ROOT_DIR / "models" / "model_metadata.json"
    baseline_path = ROOT_DIR / "models" / "drift_baseline.json"
    if model_path.exists() and metadata_path.exists() and REPORT_PATH.exists() and baseline_path.exists():
        return
    os.environ["PYTHONPATH"] = f"{ROOT_DIR / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    train_main()


def create_server(host: str = "0.0.0.0", port: int | None = None) -> ThreadingHTTPServer:
    ensure_artifacts()
    selected_port = port or int(os.environ.get("PORT", "8000"))
    return ThreadingHTTPServer((host, selected_port), BankruptcyRiskHandler)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    server = create_server(host=host, port=port)
    print(f"Bankruptcy Risk app listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
