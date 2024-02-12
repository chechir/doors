""" Functions to help with Exploratory data analysis """
import pandas as pd
from IPython.display import display
from matplotlibe import pyplot as plt

from doors.setup_logger import get_logger

logger = get_logger(__name__)


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
    corr.sort_values("abs", ascending=False, inplace=True)
    return corr


def vc(df, column):
    vc_df = df.reset_index().groupby([column]).size().to_frame("count")
    vc_df["percentage (%)"] = vc_df["count"].div(sum(vc_df["count"])).mul(100)
    logger.INFO(f'STATUS: Value counts for "{column}"...')
    display(vc_df)


def pie(df, column):
    (
        df[column]
        .value_counts()
        .plot(kind="pie", autopct="%1.1f%%", title=f'Pie chart of "{column}"')
    )
    return plt.show()
