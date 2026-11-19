"""Azure Blob Storage provider for immutable source and run artifacts."""

from __future__ import annotations

from pathlib import Path

from azure.core.credentials import TokenCredential
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings


class AzureBlobArtifactStore:
    """Store source inputs and manifests using passwordless Azure identity."""

    def __init__(self, *, endpoint: str, credential: TokenCredential) -> None:
        self._client = BlobServiceClient(account_url=endpoint, credential=credential)

    def upload_bytes(
        self,
        *,
        container_name: str,
        blob_name: str,
        data: bytes,
        content_type: str,
        overwrite: bool = False,
    ) -> str:
        """Upload bytes and return the stable Blob URI."""

        blob = self._client.get_blob_client(container=container_name, blob=blob_name)
        try:
            blob.upload_blob(
                data,
                overwrite=overwrite,
                content_settings=ContentSettings(content_type=content_type),
            )
        except ResourceExistsError:
            if overwrite:
                raise
        return blob.url

    def upload_local_file(
        self,
        *,
        container_name: str,
        blob_name: str,
        local_path: Path,
        content_type: str,
        overwrite: bool = False,
    ) -> str:
        """Upload a local file without reading it as source code."""

        return self.upload_bytes(
            container_name=container_name,
            blob_name=blob_name,
            data=local_path.read_bytes(),
            content_type=content_type,
            overwrite=overwrite,
        )

    def upload_text(
        self,
        *,
        container_name: str,
        blob_name: str,
        text: str,
        overwrite: bool = False,
    ) -> str:
        return self.upload_bytes(
            container_name=container_name,
            blob_name=blob_name,
            data=text.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            overwrite=overwrite,
        )

    def close(self) -> None:
        self._client.close()
