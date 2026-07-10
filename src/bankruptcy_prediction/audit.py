from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import AUDIT_LOG_PATH


def append_prediction_audit(
    count: int,
    model_name: str,
    threshold: float,
    risk_counts: dict[str, int],
    audit_path: Path = AUDIT_LOG_PATH,
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": int(count),
        "model_name": model_name,
        "threshold": float(threshold),
        "risk_counts": risk_counts,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def audit_summary(audit_path: Path = AUDIT_LOG_PATH, limit: int = 25) -> dict:
    if not audit_path.exists():
        return {"total_requests": 0, "total_rows_scored": 0, "recent": []}

    records = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    total_rows = sum(int(record.get("count", 0)) for record in records)
    aggregate_risk: dict[str, int] = {}
    for record in records:
        for label, value in record.get("risk_counts", {}).items():
            aggregate_risk[label] = aggregate_risk.get(label, 0) + int(value)

    return {
        "total_requests": len(records),
        "total_rows_scored": total_rows,
        "aggregate_risk_counts": aggregate_risk,
        "recent": records[-limit:],
    }
