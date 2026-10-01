"""Synthetic e-commerce sales data generator.

Produces one row per (date, region, product) with daily units, revenue, cost
and returns. The data is deliberately seeded with a realistic business problem
that the analysis engine should be able to discover on its own:

* From ANOMALY_START onward, the South region softens across all products.
* Laptop Pro in the South declines much more sharply (units fall ~55% by year
  end, with deeper discounting eroding revenue per unit).
* The Laptop Pro return rate in the South climbs from ~4% to ~16%.

Everything is driven by a fixed seed, so the same arguments always produce the
same dataset.

Usage:
    python data_generator.py                      # writes data/sales.csv
    python data_generator.py --seed 7 --out x.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "sales.csv"

# The planted problem. Exposed so tests (and later, demo scripts) can refer to it.
ANOMALY_START = "2025-07-01"
ANOMALY_REGION = "South"
ANOMALY_PRODUCT = "Laptop Pro"

# Region demand multipliers.
REGIONS = {"North": 1.10, "South": 1.00, "East": 0.95, "West": 1.05}

# price, unit cost, baseline daily units per region, baseline return rate
PRODUCTS = {
    "Laptop Pro": {"price": 1500.0, "unit_cost": 1050.0, "base_units": 12, "return_rate": 0.040},
    "Laptop Air": {"price": 1000.0, "unit_cost": 720.0, "base_units": 18, "return_rate": 0.035},
    "Tablet": {"price": 600.0, "unit_cost": 400.0, "base_units": 22, "return_rate": 0.030},
    "Smartwatch": {"price": 300.0, "unit_cost": 180.0, "base_units": 30, "return_rate": 0.025},
    "Headphones": {"price": 150.0, "unit_cost": 80.0, "base_units": 45, "return_rate": 0.020},
}

# Severity of the planted problem at full strength (end of the date range).
SOUTH_REGIONWIDE_UNIT_DROP = 0.12   # all South products lose up to 12% of units
LAPTOP_PRO_UNIT_DROP = 0.55         # South Laptop Pro loses up to 55% of units
LAPTOP_PRO_EXTRA_RETURN_RATE = 0.12 # return rate rises by up to 12 points
LAPTOP_PRO_EXTRA_DISCOUNT = 0.06    # up to 6% deeper discounting


def _seasonality(dates: pd.DatetimeIndex) -> np.ndarray:
    """Mild annual cycle, weekend bump, and a Nov/Dec holiday lift."""
    day_of_year = dates.dayofyear.to_numpy()
    annual = 1 + 0.08 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    weekend = np.where(dates.dayofweek.to_numpy() >= 5, 1.10, 1.0)
    holiday = np.where(dates.month.to_numpy() >= 11, 1.20, 1.0)
    return annual * weekend * holiday


def _anomaly_progress(dates: pd.DatetimeIndex, anomaly_start: str) -> np.ndarray:
    """0 before the anomaly starts, ramping linearly to 1 at the last date."""
    start = pd.Timestamp(anomaly_start)
    end = dates.max()
    if end <= start:
        return np.zeros(len(dates))
    progress = (dates - start) / (end - start)
    return np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)


def generate_sales_data(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    seed: int = SEED,
    anomaly_start: str = ANOMALY_START,
) -> pd.DataFrame:
    """Generate the synthetic sales dataset as a DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Full cartesian grid: every date x region x product.
    grid = pd.MultiIndex.from_product(
        [dates, list(REGIONS), list(PRODUCTS)], names=["date", "region", "product"]
    ).to_frame(index=False)

    row_dates = pd.DatetimeIndex(grid["date"])
    product_info = pd.DataFrame.from_dict(PRODUCTS, orient="index")
    info = product_info.loc[grid["product"]].reset_index(drop=True)

    progress = _anomaly_progress(row_dates, anomaly_start)
    is_south = (grid["region"] == ANOMALY_REGION).to_numpy()
    is_target = is_south & (grid["product"] == ANOMALY_PRODUCT).to_numpy()

    # --- Units -------------------------------------------------------------
    demand = (
        info["base_units"].to_numpy()
        * grid["region"].map(REGIONS).to_numpy()
        * _seasonality(row_dates)
    )
    demand *= np.where(is_south, 1 - SOUTH_REGIONWIDE_UNIT_DROP * progress, 1.0)
    demand *= np.where(is_target, 1 - LAPTOP_PRO_UNIT_DROP * progress, 1.0)
    units = rng.poisson(demand)

    # --- Returns -----------------------------------------------------------
    return_rate = info["return_rate"].to_numpy() + np.where(
        is_target, LAPTOP_PRO_EXTRA_RETURN_RATE * progress, 0.0
    )
    returns = rng.binomial(units, return_rate)

    # --- Revenue & cost ----------------------------------------------------
    # Revenue is gross sales (before returns); discounting varies day to day.
    discount = rng.uniform(0.0, 0.08, size=len(grid))
    discount += np.where(is_target, LAPTOP_PRO_EXTRA_DISCOUNT * progress, 0.0)
    revenue = units * info["price"].to_numpy() * (1 - discount)
    cost = units * info["unit_cost"].to_numpy() * rng.normal(1.0, 0.02, size=len(grid))

    df = grid.assign(
        units=units.astype(int),
        revenue=np.round(revenue, 2),
        cost=np.round(np.maximum(cost, 0.0), 2),
        returns=returns.astype(int),
    )
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "region", "product", "units", "revenue", "cost", "returns"]]


def save_dataset(df: pd.DataFrame, path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write the dataset to CSV, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce sales data.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    df = generate_sales_data(args.start, args.end, args.seed)
    path = save_dataset(df, args.out)
    print(f"Wrote {len(df):,} rows to {path}")


if __name__ == "__main__":
    main()
