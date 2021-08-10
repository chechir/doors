# pylint: disable=missing-docstring
from doors.dicts import flatten_dict


def test_flatten_dict():
    nested_dict = {"a": {"b": {"c": "d"}, "e": "f"}, "g": "h"}
    expected = {"a_b_c": "d", "a_e": "f", "g": "h"}
    result = flatten_dict(nested_dict)
    assert result == expected
