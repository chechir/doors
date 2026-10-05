# doors

These are common functions used in machine learning and data analysis work

## Optional dependencies

`snowflake-connector-python` and `google-cloud-secret-manager` are commented out in
`pyproject.toml`. They're only needed for `doors.sf` / `doors.gcp` secret helpers, and we
don't want every repo that depends on `doors` to pull them in by default. If you need
Snowflake or Secret Manager access, uncomment those lines (or install them directly) in
your own project.
