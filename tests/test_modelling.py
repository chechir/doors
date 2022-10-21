# pylint: disable=invalid-name
# pylint: disable=missing-docstring

from collections import OrderedDict

import numpy as np
from doors import modelling


def test_get_id_fold_ixs():
    np.random.seed(0)
    ids = np.array([1, 1, 1, 2, 2, 3])
    ixs = modelling.get_id_fold_ixs(ids, n_fold=2)
    expected = [
        ([3, 4, 5], [0, 1, 2]),
        ([0, 1, 2], [3, 4, 5]),
    ]

    for i, (trn_ixs, val_ixs) in enumerate(ixs):
        assert np.all(np.array(expected[i][0]) == trn_ixs)
        assert np.all(np.array(expected[i][1]) == val_ixs)


def test_time_series_cv_ixs_start_and_stop():
    dates = np.arange(100)
    decimal_ixs = modelling.get_time_series_cv_ixs(dates, folds=1, start=0.5, stop=0.75)
    other_ixs = modelling.get_time_series_cv_ixs(dates, folds=1, start=50, stop=75)
    for _, ixs in enumerate([decimal_ixs, other_ixs]):
        ixs = ixs[0]["val"]
        expected = np.repeat(False, 100)
        expected[50:75] = True
        assert np.array_equal(expected, ixs)


def test_time_series_cv_ixs():
    dates = np.arange(9)
    ixs = modelling.get_time_series_cv_ixs(dates, folds=2, start=0.5)
    expected = _get_expected_ixs()

    for key, fold_dict in ixs.items():
        for fold_name, fold_ixs in fold_dict.items():
            assert np.array_equal(expected[key][fold_name], fold_ixs)


def _get_expected_ixs():
    expected = OrderedDict()
    expected[0] = {
        "train": np.array([True, True, True, True, True, False, False, False, False])
    }
    expected[0]["val"] = np.array(
        [False, False, False, False, False, True, True, False, False]
    )
    expected[1] = {
        "train": np.array([True, True, True, True, True, True, True, False, False])
    }
    expected[1]["val"] = np.array(
        [False, False, False, False, False, False, False, True, True]
    )
    return expected
