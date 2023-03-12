""" Functions to deal with big Query"""
import pandas as pd

PROJECT_ID = "data-analytics-platform-206914"


def read_bq_data(query):
    """Read a table from BQ"""
    result_df = pd.read_gbq(
        query=query,
        project_id=PROJECT_ID,
        # private_key=GOOGLE_CREDENTIALS,
        dialect="standard",
    )
    return result_df
