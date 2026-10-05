"""Functions to deal with Snowflake"""

import json
from typing import Optional

import pandas as pd
import snowflake.connector


def read_sf_data(
    query: str,
    secret_name: str,
    project_id: str,
    credentials: str,
    warehouse: Optional[str] = None,
) -> pd.DataFrame:
    """Read a query result from Snowflake.

    `secret_name` is a GCP Secret Manager secret id, not a Snowflake user name — the
    secret holds a JSON blob (account, user, private_key, role, ...) that gets unpacked
    straight into snowflake.connector.connect, same pattern as luminate_data's own
    Snowflake secrets. Pass `warehouse` to override whichever warehouse is baked into
    the secret (e.g. "CLAUDE_WH").
    """
    secret_parsed = json.loads(credentials)
    if warehouse:
        secret_parsed["warehouse"] = warehouse

    conn = snowflake.connector.connect(**secret_parsed)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetch_pandas_all()
    finally:
        conn.close()
