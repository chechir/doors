"""Functions to deal with Snowflake"""

import json
from typing import Optional

import pandas as pd
import snowflake.connector
from google.cloud import secretmanager_v1

# TODO: confirm this is the exact secret id in GCP Secret Manager for the report_bot
# Snowflake user. luminate_data follows this same pattern (see
# luminate_data.utils.general.get_secret + luminate_data.integration) with secrets
# like "luminate_snowflake" holding the snowflake.connector.connect kwargs as JSON
# (account, user, private_key, role, warehouse, ...).
DEFAULT_SF_SECRET_NAME = "report_bot"


def _get_secret(
    secret_id: str, project_id: str = "firebird-data-project", version_id: str = "latest"
) -> str:
    """Fetch a secret payload from GCP Secret Manager."""
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    client = secretmanager_v1.SecretManagerServiceClient()
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")


def read_sf_data(
    query: str,
    secret_name: str = DEFAULT_SF_SECRET_NAME,
    project_id: str = "firebird-data-project",
    warehouse: Optional[str] = None,
) -> pd.DataFrame:
    """Read a query result from Snowflake, using the report_bot service user.

    Credentials (account, user, private_key, role, ...) are stored as a JSON blob in
    GCP Secret Manager and unpacked straight into snowflake.connector.connect — same
    pattern as luminate_data's own Snowflake secrets. Pass `warehouse` to override
    whichever warehouse is baked into the secret (e.g. "CLAUDE_WH").
    """
    secret_parsed = json.loads(_get_secret(secret_name, project_id=project_id))
    if warehouse:
        secret_parsed["warehouse"] = warehouse

    conn = snowflake.connector.connect(**secret_parsed)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetch_pandas_all()
    finally:
        conn.close()
