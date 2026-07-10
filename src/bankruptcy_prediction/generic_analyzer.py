from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnRole:
    name: str
    role: str


ROLE_PATTERNS = {
    "revenue": ("revenue", "sales", "turnover", "income", "gross sales"),
    "profit": ("profit", "net income", "earnings", "ebit", "ebitda", "margin"),
    "expense": ("expense", "cost", "cogs", "spend", "operating expense"),
    "asset": ("asset", "assets"),
    "liability": ("liability", "liabilities", "payable"),
    "debt": ("debt", "loan", "borrow", "borrowing"),
    "cash": ("cash", "bank balance", "liquidity"),
    "equity": ("equity", "net worth", "capital"),
    "date": ("date", "month", "quarter", "year", "period"),
    "customer": ("customer", "client", "account"),
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _find_columns(columns: Iterable[str]) -> list[ColumnRole]:
    roles: list[ColumnRole] = []
    for column in columns:
        normalized = _normalize(column)
        for role, patterns in ROLE_PATTERNS.items():
            if any(pattern in normalized for pattern in patterns):
                roles.append(ColumnRole(name=column, role=role))
                break
    return roles


def _first_numeric(df: pd.DataFrame, roles: list[ColumnRole], role: str) -> pd.Series | None:
    for item in roles:
        if item.role == role:
            series = pd.to_numeric(df[item.name], errors="coerce")
            if series.notna().sum() > 0:
                return series
    return None


def _date_series(df: pd.DataFrame, roles: list[ColumnRole]) -> pd.Series | None:
    for item in roles:
        if item.role == "date":
            series = pd.to_datetime(df[item.name], errors="coerce")
            if series.notna().sum() > 1:
                return series
    return None


def _trend(values: pd.Series, dates: pd.Series | None = None) -> float | None:
    cleaned = pd.to_numeric(values, errors="coerce")
    mask = cleaned.notna()
    if dates is not None:
        mask = mask & dates.notna()
    cleaned = cleaned[mask]
    if len(cleaned) < 3:
        return None
    if dates is not None:
        ordered = pd.DataFrame({"date": dates[mask], "value": cleaned}).sort_values("date")
        cleaned = ordered["value"]
    x = np.arange(len(cleaned), dtype=float)
    y = cleaned.to_numpy(dtype=float)
    mean_abs = float(np.mean(np.abs(y))) or 1.0
    return float(np.polyfit(x, y, 1)[0] / mean_abs)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def _metric(label: str, value, status: str = "info") -> dict:
    return {"label": label, "value": value, "status": status}


def _recommendations(risk_score: float, confidence: str, metrics: dict) -> list[str]:
    tips: list[str] = []
    if confidence == "Low":
        tips.append("Upload balance sheet, income statement, or cash-flow columns to improve prediction confidence.")
    if metrics.get("profit_margin") is not None and metrics["profit_margin"] < 0.05:
        tips.append("Profit margin is thin; review pricing, cost of goods, and operating expenses.")
    if metrics.get("revenue_trend") is not None and metrics["revenue_trend"] < -0.02:
        tips.append("Revenue trend is declining; investigate customer churn, seasonality, and sales pipeline health.")
    if metrics.get("debt_to_assets") is not None and metrics["debt_to_assets"] > 0.6:
        tips.append("Debt load appears high relative to assets; monitor refinancing and interest coverage risk.")
    if metrics.get("cash_to_expense") is not None and metrics["cash_to_expense"] < 0.15:
        tips.append("Cash buffer looks low against expenses; review runway and short-term liquidity.")
    if not tips and risk_score < 0.35:
        tips.append("No major red flags detected from available columns; add richer financial data for a stronger risk read.")
    return tips


def analyze_company_csv(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("No rows were provided for analysis.")
    if len(df) > 5000:
        raise ValueError("Generic company analysis is limited to 5,000 rows per request.")

    roles = _find_columns(df.columns)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_columns = [column for column in numeric_df.columns if numeric_df[column].notna().sum() > 0]
    if not numeric_columns:
        raise ValueError("No numeric columns were found. Add revenue, profit, expense, cash, debt, or similar numeric fields.")

    revenue = _first_numeric(df, roles, "revenue")
    profit = _first_numeric(df, roles, "profit")
    expense = _first_numeric(df, roles, "expense")
    assets = _first_numeric(df, roles, "asset")
    liabilities = _first_numeric(df, roles, "liability")
    debt = _first_numeric(df, roles, "debt")
    cash = _first_numeric(df, roles, "cash")
    equity = _first_numeric(df, roles, "equity")
    dates = _date_series(df, roles)

    total_revenue = float(revenue.sum()) if revenue is not None else None
    total_profit = float(profit.sum()) if profit is not None else None
    total_expense = float(expense.sum()) if expense is not None else None
    total_assets = float(assets.mean()) if assets is not None else None
    total_liabilities = float(liabilities.mean()) if liabilities is not None else None
    total_debt = float(debt.mean()) if debt is not None else None
    total_cash = float(cash.mean()) if cash is not None else None
    total_equity = float(equity.mean()) if equity is not None else None

    metrics = {
        "profit_margin": _safe_ratio(total_profit, total_revenue),
        "expense_ratio": _safe_ratio(total_expense, total_revenue),
        "debt_to_assets": _safe_ratio(total_debt or total_liabilities, total_assets),
        "liability_to_equity": _safe_ratio(total_liabilities, total_equity),
        "cash_to_expense": _safe_ratio(total_cash, total_expense),
        "revenue_trend": _trend(revenue, dates) if revenue is not None else None,
        "profit_trend": _trend(profit, dates) if profit is not None else None,
    }

    risk_score = 0.35
    evidence = 0
    if metrics["profit_margin"] is not None:
        evidence += 1
        risk_score += 0.22 if metrics["profit_margin"] < 0 else 0.12 if metrics["profit_margin"] < 0.05 else -0.08
    if metrics["expense_ratio"] is not None:
        evidence += 1
        risk_score += 0.14 if metrics["expense_ratio"] > 0.95 else 0.07 if metrics["expense_ratio"] > 0.8 else -0.04
    if metrics["debt_to_assets"] is not None:
        evidence += 1
        risk_score += 0.18 if metrics["debt_to_assets"] > 0.7 else 0.09 if metrics["debt_to_assets"] > 0.5 else -0.04
    if metrics["liability_to_equity"] is not None:
        evidence += 1
        risk_score += 0.14 if metrics["liability_to_equity"] > 2 else 0.07 if metrics["liability_to_equity"] > 1 else -0.03
    if metrics["cash_to_expense"] is not None:
        evidence += 1
        risk_score += 0.12 if metrics["cash_to_expense"] < 0.1 else -0.04 if metrics["cash_to_expense"] > 0.4 else 0
    if metrics["revenue_trend"] is not None:
        evidence += 1
        risk_score += 0.14 if metrics["revenue_trend"] < -0.03 else 0.06 if metrics["revenue_trend"] < 0 else -0.04
    if metrics["profit_trend"] is not None:
        evidence += 1
        risk_score += 0.12 if metrics["profit_trend"] < -0.03 else -0.03 if metrics["profit_trend"] > 0 else 0

    risk_score = float(min(max(risk_score, 0.02), 0.98))
    confidence = "High" if evidence >= 5 else "Medium" if evidence >= 3 else "Low"
    risk_label = "Critical" if risk_score >= 0.75 else "High" if risk_score >= 0.55 else "Watch" if risk_score >= 0.35 else "Low"

    displayed_metrics = [
        _metric("Rows analyzed", int(len(df))),
        _metric("Numeric columns", int(len(numeric_columns))),
        _metric("Detected financial columns", int(len(roles))),
    ]
    for key, label in [
        ("profit_margin", "Profit margin"),
        ("expense_ratio", "Expense ratio"),
        ("debt_to_assets", "Debt to assets"),
        ("liability_to_equity", "Liability to equity"),
        ("cash_to_expense", "Cash to expense"),
        ("revenue_trend", "Revenue trend"),
        ("profit_trend", "Profit trend"),
    ]:
        value = metrics[key]
        if value is not None:
            displayed_metrics.append(_metric(label, float(value)))

    return {
        "mode": "generic_company_analysis",
        "risk_score": risk_score,
        "risk_label": risk_label,
        "confidence": confidence,
        "detected_roles": [role.__dict__ for role in roles],
        "metrics": displayed_metrics,
        "recommendations": _recommendations(risk_score, confidence, metrics),
        "note": "This is a flexible business-risk estimate. Bankruptcy-model predictions require the strict 95-column financial schema.",
    }
