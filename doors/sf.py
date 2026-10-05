"""Functions to deal with Snowflake"""

import json
from typing import Optional, cast

import pandas as pd
import snowflake.connector


def read_sf_data(
    query: str,
    secret_name: str,
    project_id: str,
    credentials: str,
    warehouse: Optional[str] = None,
) -> pd.DataFrame:
    """Read a query result from Snowflake."""
    secret_parsed = json.loads(credentials)
    if warehouse:
        secret_parsed["warehouse"] = warehouse

    conn = snowflake.connector.connect(**secret_parsed)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        return cast(pd.DataFrame, cursor.fetch_pandas_all())
    finally:
        conn.close()
