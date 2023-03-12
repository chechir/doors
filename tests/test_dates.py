import datetime

import numpy as np

import doors as dd


def test_get_season():
    assert dd.dates.get_season(1) == "winter"
    assert dd.dates.get_season(90) == "spring"
    assert dd.dates.get_season(175) == "summer"
    assert dd.dates.get_season(270) == "autumn"


def test_get_day_of_year():
    date = np.datetime64("2017-01-01T20:00:00.000000000")
    res = dd.dates.get_day_of_year([date])
    assert res[0] == 1


def test_get_day_of_week():
    date = np.datetime64("2017-05-15T20:00:00.000000000")
    res = dd.dates.get_day_of_week([date])
    assert res[0] == 0


def test_get_hour():
    date = np.datetime64("2017-05-15T20:00:00.000000000")
    res = dd.dates.get_hour([date])
    assert res[0] == 20


def test_get_month():
    date = np.datetime64("2017-05-15T20:00:00.000000000")
    res = dd.dates.get_month([date])
    assert res[0] == 5


def test_get_week():
    date = np.datetime64("2017-01-02T20:00:00.000000000")
    res = dd.dates.get_week([date])
    assert res[0] == 1


def test_get_all_dates_in_range():
    start_date = datetime.date(2016, 1, 1)
    end_date = datetime.date(2016, 1, 4)
    expected_dates = [
        datetime.date(2016, 1, 1),
        datetime.date(2016, 1, 2),
        datetime.date(2016, 1, 3),
        datetime.date(2016, 1, 4),
    ]
    all_days = dd.dates.get_all_dates_in_range(start_date, end_date)
    assert all_days == expected_dates

    # if end_date is missing it assumed today
    start_date = dd.dates.get_datetime_yesterday_morning().date()
    end_date = dd.dates.get_datetime_this_morning().date()
    all_days = dd.dates.get_all_dates_in_range(start_date)
    assert all_days == [start_date, end_date]


def test_get_diff_days():
    dates = [
        datetime.date(2019, 12, 1),
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 3),
        datetime.date(2020, 1, 4),
    ]
    res = dd.dates.get_diff_in_days(dates, add_init=True)
    expected = [np.nan, 31, 2, 1]
    assert dd.np.nan_allclose(res, expected)
