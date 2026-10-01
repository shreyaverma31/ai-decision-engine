"""Data loading, validation and KPI calculation.

This module is the deterministic foundation of the decision engine: every
number an insight or recommendation cites should be computable from these
functions, so it can be traced back to the source rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "region", "product", "units", "revenue", "cost", "returns"]
NUMERIC_COLUMNS = ["units", "revenue", "cost", "returns"]
TEXT_COLUMNS = ["region", "product"]


class DataValidationError(ValueError):
    """Raised when a dataset fails validation. `issues` lists every problem found."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Invalid dataset:\n- " + "\n- ".join(issues))


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a sales CSV and return a validated, correctly typed DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return validate_data(df)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema and basic business rules; return a cleaned copy.

    Collects all problems before raising so the user sees everything at once.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError([f"Missing required columns: {', '.join(missing)}"])
    if df.empty:
        raise DataValidationError(["Dataset has no rows."])

    df = df.copy()
    issues: list[str] = []

    dates = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(dates.isna().sum() - df["date"].isna().sum())
    if bad_dates:
        issues.append(f"'date' has {bad_dates} unparseable value(s).")
    df["date"] = dates

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = int(converted.isna().sum() - df[col].isna().sum())
        if bad:
            issues.append(f"'{col}' has {bad} non-numeric value(s).")
        df[col] = converted

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    for col, count in null_counts[null_counts > 0].items():
        issues.append(f"'{col}' has {int(count)} missing value(s).")

    for col in NUMERIC_COLUMNS:
        negatives = int((df[col] < 0).sum())
        if negatives:
            issues.append(f"'{col}' has {negatives} negative value(s).")

    over_returned = int((df["returns"] > df["units"]).sum())
    if over_returned:
        issues.append(f"{over_returned} row(s) have more returns than units sold.")

    if issues:
        raise DataValidationError(issues)
    return df


def _return_rate(returns: float, units: float) -> float:
    return float(returns / units) if units else 0.0


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Headline KPIs for a (possibly filtered) dataset.

    Revenue is gross sales; return rate is returned units / units sold.
    """
    revenue = float(df["revenue"].sum())
    cost = float(df["cost"].sum())
    units = int(df["units"].sum())
    returns = int(df["returns"].sum())
    profit = revenue - cost
    return {
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "profit_margin": profit / revenue if revenue else 0.0,
        "units": units,
        "returns": returns,
        "return_rate": _return_rate(returns, units),
    }


def kpis_by(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """The same KPIs as calculate_kpis, broken down by one or more columns."""
    grouped = df.groupby(by, observed=True)[NUMERIC_COLUMNS].sum().reset_index()
    grouped["profit"] = grouped["revenue"] - grouped["cost"]
    grouped["profit_margin"] = np.where(
        grouped["revenue"] > 0, grouped["profit"] / grouped["revenue"].where(grouped["revenue"] > 0), 0.0
    )
    grouped["return_rate"] = np.where(
        grouped["units"] > 0, grouped["returns"] / grouped["units"].where(grouped["units"] > 0), 0.0
    )
    return grouped
