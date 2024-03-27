# pylint: disable=missing-docstring
# pylint: disable=invalid-name
import numpy as np
import pandas as pd

from doors.features import (
    categorical_to_frequency,
    days_since_result,
    days_to_first_event,
    ema,
    grouped_days_since_result,
    grouped_ema,
    grouped_lagged_decay,
    grouped_lagged_ema,
    lagged_ema,
)
from doors.np import nan_allclose


def test_categorical_to_frequency():
    df = pd.DataFrame({"cat": [1, 1, 2, 3, 4, 4, 4]})
    result = categorical_to_frequency(df, "cat")
    expected = [2, 2, 1, 1, 3, 3, 3]
    assert np.allclose(expected, result)


def test_grouped_lagged_decay():
    df = pd.DataFrame({"ticker_name": ["x", "x", "x", "x"], "win_flag": [1, 0, 1, 0]})

    result = grouped_lagged_decay(df, "ticker_name", "win_flag")
    expected = [0, 1, (0 + 1 * np.e**-1), (1 + 0 * np.e**-1 + 1 * np.e**-2)]
    assert np.allclose(expected, result)


def test_grouped_lagged_decay_with_nans():
    df = pd.DataFrame(
        {"ticker_name": ["x", "x", "x", "x"], "win_flag": [1, np.nan, 0, np.nan]}
    )

    result = grouped_lagged_decay(df, "ticker_name", "win_flag")
    expected = [0, 1, (0 + 1 * np.e**-1), (0 + 0 * np.e**-1 + 1 * np.e**-2)]
    assert np.allclose(expected, result)


def test_days_to_first_event():
    df = pd.DataFrame(
        {
            "scheduled_time": np.array(
                ["2011-01-01", "2011-01-02", "2011-01-12"]
            ).astype("datetime64[ns]"),
            "ticker_name": ["a", "a", "a"],
        }
    )
    result = days_to_first_event(df, "ticker_name", "scheduled_time")
    expected = [0, 1, 11]
    assert np.allclose(expected, result)


def test_grouped_days_since_result():
    df = pd.DataFrame(
        {
            "runner_id": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            "win_flag": np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0]),
            "scheduled_time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-06",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-06",
                ]
            ),
        }
    )
    expected = [-1, 1, 2, 1, 3, -1, 1, 2, 1, 3]
    # result = features.days_to_previous_result(df, col='win_flag', value=1)
    result = grouped_days_since_result(df, groupby="runner_id", col="win_flag")
    assert np.allclose(expected, result)


def test_days_since_result():
    series = pd.Series([10, 20, 10, 0, 8])
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-04",
            "2024-01-10",
            "2024-01-20",
        ]
    )
    result = days_since_result(series, dates, value=5)
    expected = [np.nan, 1, 2, 6, 16]
    assert nan_allclose(result, expected)


def test_ema():
    array = pd.Series(np.array([10, 0, 0, 0, 0, 0]))
    assert np.allclose(array, ema(array, n_period=1))
    expected = [10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
    assert np.allclose(expected, ema(array, n_period=3))


def test_lagged_ema():
    array = pd.Series(np.array([10, 0, 0, 0, 0, 0]))
    expected = [-1, 10, 0, 0, 0, 0]
    assert np.allclose(expected, lagged_ema(array, n_period=1, init=-1))
    expected = [-2, -2, 10.0, 5.0, 2.5, 1.25]
    assert np.allclose(expected, lagged_ema(array, n_period=3, shift=2, init=-2))


def test_grouped_ema():
    df = pd.DataFrame(
        {
            "group": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            "price": np.array([10, 0, 10, 0, 0] * 2),
        }
    )
    expected = [10, 0, 10, 0, 0] * 2
    assert np.allclose(df["price"], grouped_ema(df, "price", 1, "group"))
    expected = [10.0, 5.0, 7.5, 3.75, 1.875] * 2
    assert np.allclose(expected, grouped_ema(df, "price", 3, "group"))


def test_grouped_lagged_ema():
    df = pd.DataFrame(
        {
            "group": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            "price": np.array([10, 0, 10, 0, 0] * 2),
        }
    )
    expected = [-1, 10, 0, 10, 0] * 2
    assert np.allclose(
        expected, grouped_lagged_ema(df, "price", 1, "group", shift=1, init=-1)
    )
    expected = [-2.0, -2.0, 10.0, 5.0, 7.5] * 2
    assert np.allclose(
        expected, grouped_lagged_ema(df, "price", 3, "group", shift=2, init=-2)
    )
