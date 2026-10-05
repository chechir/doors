# doors

These are common functions used in machine learning and data analysis work

## Optional dependencies

`snowflake-connector-python`, `google-cloud-secret-manager`, and `pandas-gbq` are
commented out in `pyproject.toml`. They're only needed for `doors.sf` (Snowflake),
`doors.gcp` (Secret Manager), and `doors.bq` (BigQuery) respectively, and we don't want
every repo that depends on `doors` to pull them in by default. If you need Snowflake,
Secret Manager, or BigQuery access, uncomment the relevant line (or install it directly)
in your own project.
