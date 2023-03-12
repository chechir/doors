from collections import OrderedDict

import pandas as pd

from doors.venn import (
    get_all_combinations,
    get_elements_for_each_combination,
    get_venn_values,
)

# pylint: disable=invalid-name
# pylint: disable=missing-docstring


def test_get_venn_values():
    df = pd.DataFrame(
        {
            "pair": ["primary", "primary", "secondary", "secondary"],
            "collect_id": [1, 2, 1, 3],
        }
    )
    expected = OrderedDict(
        [(("primary", "secondary"), {1}), (("primary",), {2}), (("secondary",), {3})]
    )
    result = get_venn_values(df, "pair", "collect_id")
    assert dict(result) == dict(expected)


def test_get_elements_for_each_combination():
    df = pd.DataFrame(
        {
            "pair": ["primary", "primary", "secondary", "secondary"],
            "collect_id": [1, 2, 1, 3],
        }
    )
    expected = OrderedDict(
        [
            (("primary", "secondary"), {1}),
            (("primary",), {1, 2}),
            (("secondary",), {1, 3}),
        ]
    )
    result = get_elements_for_each_combination(df, "pair", "collect_id")
    assert dict(result) == dict(expected)


def test_get_all_combinations():
    circle_ids = ["m1a1", "m1a2", "m1a3"]
    venn_elements = get_all_combinations(circle_ids)
    expected = [
        [("m1a1", "m1a2", "m1a3")],
        [("m1a2", "m1a3"), ("m1a1", "m1a2"), ("m1a1", "m1a3")],
        [("m1a1",), ("m1a2",), ("m1a3",)],
    ]

    for i, _ in enumerate(expected):
        assert set(expected[i]) == set(venn_elements[i])
