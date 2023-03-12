# pylint: disable=invalid-name
import copy
from functools import partial

import numpy as np

from doors import np as wnp


def categorical_to_numeric(df, column):
    """convert text column into numeric using the character codes"""

    def char_to_numeric(char):
        return str(ord(char))

    def text_to_numeric(text):
        text = str(text).strip()
        text = text[:10]
        text = text.lower()
        numeric_chars = map(char_to_numeric, text)
        result = "".join(numeric_chars)
        result = float(result)
        return result

    result = map(text_to_numeric, df[column])
    result = np.log(np.array(result))
    return result


def categorical_to_frequency(df, column):
    """convert categorical column using the frequency of elements"""
    ixs = wnp.get_group_ixs(df[column].values)
    res = np.zeros(len(df))
    for ix in ixs.values():
        res[ix] = len(ix)
    return res.astype(np.int64)


def grouped_lagged_decay(df, groupby, col, fillna=0, decay=1):
    """Grouped lagged decay"""
    values = wnp.fillna(df[col].values, 0)
    f = partial(lagged_decay, decay=decay)
    result = wnp.group_apply(values, df[groupby].values, f)
    result = wnp.fillna(result, fillna)
    return result


def lagged_decay(ordered_values, decay=1):
    """lagged decay"""
    result = np.nan * ordered_values
    previous_value = np.nan
    historic_score = np.nan
    current_score = 0
    for i, value in enumerate(ordered_values):
        if i > 0:
            current_score = previous_value + historic_score * np.exp(-decay)
            result[i] = current_score
        previous_value = value
        historic_score = current_score
    return result


def days_to_first_event(df, groupby, time_col):
    """Calculate days to the first date for each group, in a Time series"""
    dates = df[time_col].astype("datetime64[ns]").values
    ids = df[groupby].values
    result = wnp.group_apply(dates, ids, _time_to_min_date)
    result = _convert_ns_to_days(result)
    return result


def _time_to_min_date(v):
    min_date = np.min(v)
    return v - min_date


def _convert_ns_to_days(values):
    return (((values / 1000000000) / 60) / 60) / 24


def grouped_days_since_result(
    df, groupby, col="win_flag", value=1, fillna=-1, coldate="scheduled_time"
):
    func = partial(days_since_result, value=1)
    result = wnp.group_apply(
        df[[col, coldate]].values, df[groupby].values, func, multiarg=True
    )
    result = wnp.fillna(result, fillna)
    return result


def days_since_result(v, dates, value=1):
    dates = dates.astype("datetime64[ms]")
    date_of_last_win = copy.deepcopy(dates)
    win_ix = v >= value
    date_of_last_win[~win_ix] = np.datetime64("NaT")
    # just to shift: shove a nat to start, drop the last value
    date_of_last_win = np.r_[np.datetime64("NaT"), date_of_last_win[:-1]]
    date_of_last_win = wnp.ffill(date_of_last_win)
    diffs = (dates - date_of_last_win).astype("timedelta64[D]")
    nan_ix = wnp.isnull(diffs)
    diffs = diffs.astype(float)
    diffs[nan_ix] = np.nan
    return diffs


# def grouped_lagged_ema(df, calc_colname, alpha, groupby):
#     v = df[calc_colname]
#     func = partial(lagged_ema, alpha=alpha)
#     result = wnp.group_apply(v, df[groupby], func)
#     return result
#
# def grouped_lagged_dema(df, calc_colname, span, beta, groupby):
#     v = df[calc_colname]
#     func = partial(lagged_dema, span=span, beta=beta)
#     result = ss.np.group_apply(v, df[groupby], func)
#     return result
#
# def grouped_lagged_ema(df, calc_colname, alpha, groupby):
#     # This is an improvement of pesky_quiz function
#     func = partial(wg.lagged_ema, alpha=alpha)
#     result = ss.np.group_apply(df[calc_colname], df[groupby], func)
#     return result


def grouped_ema(df, col, alpha, groupby):
    v = df[col].values
    func = partial(ema, alpha=alpha)
    result = wnp.group_apply(v, df[groupby].values, func)
    return result


def ema(v, alpha=0.2):
    result = np.nan * v
    result[0] = v[0]
    for i in range(1, len(v)):
        result[i] = alpha * v[i] + (1 - alpha) * result[i - 1]
    return result


def lagged_ema(v, alpha):
    emas = ema(v, alpha)
    emas = wnp.lag(emas, init=0)
    return emas


def dema(v, span, beta):
    """needs test"""
    intercept = v * np.nan
    slope = v * np.nan
    intercept[0] = v[0]
    slope[0] = 0
    alpha = 2.0 / (1 + span)
    for i in range(1, len(v)):
        intercept[i] = alpha * v[i] + (1 - alpha) * (intercept[i - 1] + slope[i - 1])
        slope[i] = beta * (intercept[i] - intercept[i - 1]) + (1 - beta) * slope[i - 1]
    return intercept


def lagged_dema(v, span, beta):
    """needs test"""
    demas = dema(v, span, beta)
    demas = wnp.lag(demas, init=0)
    return demas
