# pylint: disable=missing-docstring
# pylint: disable=invalid-name
import numpy as np
import pandas as pd
import pytest

from doors import np as utils_np


def test_moving_average():
    array = np.arange(100)
    result = utils_np.moving_average(array, 10)
    assert np.allclose(result[-1:], (94.5))
    assert len(result) == len(array)


def test_get_str_columns():
    df = pd.DataFrame({"MSZoning": ["FV"], "MiscFeature": [np.nan], "MoSold": [2]})
    result = utils_np.get_str_columns(df)
    for col in result:
        assert col in ["MSZoning", "MiscFeature"]


def test_nan_ffill():
    v = np.array([np.nan, 1, np.nan, np.nan, 2, np.nan, 3, np.nan])
    expected = np.array([np.nan, 1, 1, 1, 2, 2, 3, 3])
    assert utils_np.nan_allclose(utils_np.ffill(v), expected)
    v = np.array([np.nan, "S", np.nan, np.nan, 2, np.nan, 3, np.nan], dtype="O")
    expected = np.array([np.nan, "S", "S", "S", 2, 2, 3, 3], dtype="O")
    assert all(utils_np.nan_equality(utils_np.ffill(v), expected))


losses = [utils_np.bin_ent, utils_np.abs_loss, utils_np.sq_loss]


@pytest.mark.parametrize("loss", losses)
def test_losses_raise_with_wrong_shapes(loss):
    predictions = 1.0 / np.arange(2, 12).reshape((-1, 1))
    bad_flags = np.array([0] * 5 + [1] * 5)

    with pytest.raises(AssertionError):
        loss(bad_flags, predictions)


def test_nan_equality():
    # scalar
    assert utils_np.nan_equality(10, 10)
    assert utils_np.nan_equality(np.nan, np.nan)
    assert not utils_np.nan_equality(10, 11)
    assert not utils_np.nan_equality(10, np.nan)
    # vector-vector
    tens = np.repeat(10, 5)
    nans = np.repeat(np.nan, 5)
    assert all(utils_np.nan_equality(tens, tens))
    assert all(utils_np.nan_equality(nans, nans))
    assert not all(utils_np.nan_equality(tens, tens + 1))
    assert not all(utils_np.nan_equality(tens, nans))
    # vector-scalar
    tens = np.repeat(10, 5)
    nans = np.repeat(np.nan, 5)
    strings = np.repeat("a", 5)
    objects = np.array(["a", np.nan], dtype="O")
    assert all(utils_np.nan_equality(strings, "a"))
    assert all(utils_np.nan_equality(tens, 10))
    assert all(utils_np.nan_equality(nans, np.nan))
    assert any(utils_np.nan_equality(objects, np.nan))
    assert not all(utils_np.nan_equality(strings, "b"))
    assert not all(utils_np.nan_equality(tens, 11))
    assert not all(utils_np.nan_equality(tens, np.nan))
    assert not all(utils_np.nan_equality(objects, np.nan))


def test_loss_regression():
    test_data = [
        (utils_np.bin_ent, 1.2714816103091837),
        (utils_np.abs_loss, 0.5880122655122656),
        (utils_np.sq_loss, 0.44182775042217681),
    ]
    for loss, expected_value in test_data:
        predictions = 1.0 / np.arange(2, 12).reshape((-1, 1))
        good_flags = np.array([0] * 5 + [1] * 5).reshape((-1, 1))
        loss_value = np.mean(loss(good_flags, predictions))
        assert np.isclose(loss_value, expected_value)


def test_get_new_value_flag():
    values = [1, 2, 3, 1, 1, 2, 2, 3, 4]
    result = utils_np.get_new_value_flags(values)
    expected = [True, True, True, False, False, False, False, False, True]
    assert np.allclose(result, expected)


def test_nan_allclose():
    x = np.array([np.nan, 1])
    x = np.vstack([x, x]).T
    y = np.array([np.nan, 1 - 1e-12])
    y = np.vstack([y, y]).T
    assert utils_np.nan_allclose(x, y)


def test_fillna():
    x = np.array([1, 2, 3, np.nan])
    y = np.array([0, 0, 0, 1])
    expected = np.array([1, 2, 3, 1])
    output = utils_np.fillna(x, y)
    assert np.array_equal(expected, output)


def test_vector_group_apply():
    test_data = [
        # values, ids, expected
        (
            np.array([1, 0, 0, 1, 1, 0]),
            np.array([1, 2, 1, 2, 1, 2]),
            np.array([1, 0, 1, 1, 2, 1]),
        ),
        (
            np.array([1, 0, 0, 1, 1, 0]),
            np.array(["A", "B", "A", "B", "A", "B"]),
            np.array([1, 0, 1, 1, 2, 1]),
        ),
    ]
    for values, ids, expected in test_data:
        output = utils_np.group_apply(values, ids, np.cumsum)
        assert np.all(output == expected)


def test_vector_group_apply_works_with_tuple_ids():
    values = np.array([1, 1, 1, 2, 2, 2])
    all_ids = [
        np.array(list(zip(np.array([1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 2, 2])))),
        np.array(
            list(
                zip(
                    np.array([1, 1, 1, 2, 2, 2]),
                    np.array(["A", "A", "A", "B", "C", "C"]),
                )
            )
        ),
        np.array(
            list(
                zip(
                    np.array(["a", "a", "a", "b", "b", "b"]),
                    np.array(["A", "A", "A", "B", "C", "C"]),
                )
            )
        ),
    ]
    expected = np.array([3, 3, 3, 2, 4, 4])
    for ids in all_ids:
        output = utils_np.group_apply(values, ids, np.sum)
        assert np.all(output == expected)


def test_vector_group_apply_works_with_2dims():
    values = np.array([1, 1, 1, 2, 2, 2])
    apply_func = np.sum
    ids = np.array([[1, 0], [1, 0], [1, 0], [1, 1], [2, 2], [2, 2]])
    expected = np.array([3, 3, 3, 2, 4, 4])
    output = utils_np.group_apply(values, ids, apply_func)
    assert np.all(output == expected)


def test_vector_group_apply_works_with_3dims():
    values = np.array([1, 1, 1, 2, 2, 2])
    apply_func = np.sum
    ids = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [2, 2, 1], [2, 2, 1]])
    expected = np.array([2, 2, 1, 2, 4, 4])
    output = utils_np.group_apply(values, ids, apply_func)
    assert np.all(output == expected)


def test_lag_func():
    values = np.array([1, 1, 2, 2, 3, 3])
    expected = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, 3.0])
    output = utils_np.lag(values, np.nan)
    assert all(utils_np.nan_equality(expected, output))


def test_lag_func_with_shift():
    values = np.array([1, 1, 2, 2, 3, 3])
    expected = np.array([999, 999, 999, 1.0, 1.0, 2.0])
    output = utils_np.lag(values, 999, shift=3)
    assert all(utils_np.nan_equality(expected, output))


def test_lagged_cumsum_works_with_matrices():
    v = np.arange(10).reshape(5, 2)
    res = utils_np.lagged_cumsum(v, np.nan)
    assert res.shape == (5, 2)


def test_is_npdatetime():
    v = np.array(["2001-01-01", "NaT"], dtype="datetime64[ns]")
    assert utils_np.is_npdatetime(v)


def test_get_unique_values_in_order():
    values = [-1, -1, 2, 0, 5, 5, 5]
    unique = utils_np.get_unique_values_in_order(values)
    expected = np.array([-1, 2, 0, 5])
    assert np.array_equal(unique, expected)


def test_ix_to_bool():
    rows = [2, 3, 4]
    bools = utils_np.ix_to_bool(rows, 10)
    assert not any(bools[0:2])
    assert not any(bools[5:])
    assert all(bools[2:5])
    assert len(bools) == 10


def test_replace():
    v = np.array([1, 1, 2, 2, -1, -1, np.nan])
    new = utils_np.replace(v, {-1: 10, np.nan: 0})
    expected = np.array([1, 1, 2, 2, 10, 10, 0])
    assert np.array_equal(new, expected)


def test_change_flag():
    values = np.array([1, 1, 1, 2, 3, 4, 1])
    result = utils_np.change_flag(values, init=0)
    expected = np.array([0, 0, 0, 1, 1, 1, 1])
    assert np.array_equal(result, expected)


def test_cross_entropy_with_unnormalised_predictions():
    predictions = np.array([[0.1, 0.8, 0.1], [0.8, 0.5, 0.3]])
    flags = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(AssertionError):
        utils_np.x_ent(flags, predictions)
