import pandas as pd
import pytest

from analyzer import DataValidationError, calculate_kpis, kpis_by, load_data, validate_data
from data_generator import (
    ANOMALY_PRODUCT,
    ANOMALY_REGION,
    ANOMALY_START,
    generate_sales_data,
    save_dataset,
)


@pytest.fixture
def sample_df():
    return validate_data(
        pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
                "region": ["North", "South", "North", "South"],
                "product": ["Tablet", "Tablet", "Laptop Pro", "Laptop Pro"],
                "units": [10, 20, 5, 0],
                "revenue": [6000.0, 12000.0, 7500.0, 0.0],
                "cost": [4000.0, 8000.0, 5250.0, 0.0],
                "returns": [1, 2, 1, 0],
            }
        )
    )


# --- KPI calculations --------------------------------------------------------

def test_kpi_totals(sample_df):
    kpis = calculate_kpis(sample_df)
    assert kpis["revenue"] == pytest.approx(25500.0)
    assert kpis["cost"] == pytest.approx(17250.0)
    assert kpis["units"] == 35
    assert kpis["returns"] == 4


def test_profit_and_margin(sample_df):
    kpis = calculate_kpis(sample_df)
    assert kpis["profit"] == pytest.approx(25500.0 - 17250.0)
    assert kpis["profit_margin"] == pytest.approx(8250.0 / 25500.0)


def test_return_rate(sample_df):
    assert calculate_kpis(sample_df)["return_rate"] == pytest.approx(4 / 35)


def test_zero_units_gives_zero_return_rate(sample_df):
    empty_sales = sample_df[sample_df["units"] == 0]
    kpis = calculate_kpis(empty_sales)
    assert kpis["return_rate"] == 0.0
    assert kpis["profit_margin"] == 0.0


def test_kpis_by_region(sample_df):
    by_region = kpis_by(sample_df, "region").set_index("region")
    north, south = by_region.loc["North"], by_region.loc["South"]
    assert north["revenue"] == pytest.approx(13500.0)
    assert north["profit"] == pytest.approx(13500.0 - 9250.0)
    assert north["return_rate"] == pytest.approx(2 / 15)
    assert south["return_rate"] == pytest.approx(2 / 20)


def test_grouped_kpis_sum_to_totals(sample_df):
    totals = calculate_kpis(sample_df)
    by_product = kpis_by(sample_df, ["region", "product"])
    for col in ["revenue", "cost", "profit", "units", "returns"]:
        assert by_product[col].sum() == pytest.approx(totals[col])


# --- Validation --------------------------------------------------------------

def test_missing_column_rejected(sample_df):
    with pytest.raises(DataValidationError, match="returns"):
        validate_data(sample_df.drop(columns=["returns"]))


def test_negative_and_over_returned_rows_rejected(sample_df):
    bad = sample_df.copy()
    bad.loc[0, "revenue"] = -5.0
    bad.loc[1, "returns"] = 999
    with pytest.raises(DataValidationError) as exc:
        validate_data(bad)
    assert len(exc.value.issues) == 2


def test_non_numeric_and_bad_date_rejected(sample_df):
    bad = sample_df.astype({"units": object, "date": object})
    bad.loc[0, "units"] = "ten"
    bad.loc[1, "date"] = "not a date"
    with pytest.raises(DataValidationError) as exc:
        validate_data(bad)
    assert any("units" in issue for issue in exc.value.issues)
    assert any("date" in issue for issue in exc.value.issues)


def test_load_data_round_trip(tmp_path, sample_df):
    path = tmp_path / "sales.csv"
    sample_df.to_csv(path, index=False)
    loaded = load_data(path)
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    assert calculate_kpis(loaded) == pytest.approx(calculate_kpis(sample_df))


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


# --- Synthetic dataset -------------------------------------------------------

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    path = save_dataset(generate_sales_data(), tmp_path_factory.mktemp("data") / "sales.csv")
    return load_data(path)


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_sales_data(seed=1), generate_sales_data(seed=1))
    assert not generate_sales_data(seed=1).equals(generate_sales_data(seed=2))


def test_generated_data_passes_validation(generated):
    assert len(generated) == 365 * 4 * 5


def _before_after(df):
    cutoff = pd.Timestamp(ANOMALY_START)
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def _daily_units(df):
    return df["units"].sum() / df["date"].nunique()


def test_planted_laptop_pro_south_decline(generated):
    target = generated[
        (generated["region"] == ANOMALY_REGION) & (generated["product"] == ANOMALY_PRODUCT)
    ]
    before, after = _before_after(target)
    assert _daily_units(after) < 0.8 * _daily_units(before)
    assert calculate_kpis(after)["return_rate"] > 1.5 * calculate_kpis(before)["return_rate"]


def test_decline_is_specific_to_south(generated):
    """Other regions' Laptop Pro sales should not show the same drop."""
    control = generated[
        (generated["region"] != ANOMALY_REGION) & (generated["product"] == ANOMALY_PRODUCT)
    ]
    before, after = _before_after(control)
    assert _daily_units(after) > 0.95 * _daily_units(before)
    assert calculate_kpis(after)["return_rate"] < 1.2 * calculate_kpis(before)["return_rate"]
