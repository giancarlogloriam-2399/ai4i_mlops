import pytest

from ai4i_mlops.config import FEATURES, LEAKAGE_COLUMNS, TARGET, TYPE_VALUES
from ai4i_mlops.data import load_data, validate_data


def test_data_shape():
    df = load_data()
    assert len(df) == 10000
    assert list(df.columns) == FEATURES + [TARGET]


def test_no_leakage_columns():
    df = load_data()
    for columna in LEAKAGE_COLUMNS:
        assert columna not in df.columns


def test_type_only_valid_values():
    df = load_data()
    assert set(df["type"]) <= set(TYPE_VALUES)


def test_validate_rejects_nulls():
    df = load_data().head(10).copy()
    df.loc[0, "torque"] = None
    with pytest.raises(ValueError):
        validate_data(df)


def test_validate_rejects_out_of_range():
    df = load_data().head(10).copy()
    df.loc[0, "air_temperature"] = 500.0
    with pytest.raises(ValueError):
        validate_data(df)
