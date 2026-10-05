from typing import Any

from google.cloud import secretmanager_v1


def get_secret(secret_id: str, project_id: str, version_id: str = "latest") -> Any:
    """Fetch a secret payload from GCP Secret Manager."""
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    client = secretmanager_v1.SecretManagerServiceClient()
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")
