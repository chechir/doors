""" Functions to help with Exploratory data analysis """
import pandas as pd


def get_correlations_for_col(
    data: pd.DataFrame, col: str, method="pearson"
) -> pd.DataFrame:
    corr = data.corr(numeric_only=True, method=method)[col]
    corr = pd.DataFrame(
        {
            "abs": corr.abs(),
            "corr": corr,
        }
    )
    return corr
