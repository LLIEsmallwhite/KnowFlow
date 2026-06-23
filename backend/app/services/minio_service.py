"""
MinIO Object Storage Service

Provides file upload/download/delete for document storage.
"""

import io
import logging
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger(__name__)


class MinioService:
    """MinIO client wrapper for document storage."""

    def __init__(self):
        self._client: Optional[Minio] = None

    @property
    def client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
        return self._client

    def ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            found = self.client.bucket_exists(settings.MINIO_BUCKET)
            if not found:
                self.client.make_bucket(settings.MINIO_BUCKET)
                logger.info("Created MinIO bucket: %s", settings.MINIO_BUCKET)
            else:
                logger.debug("Bucket '%s' already exists", settings.MINIO_BUCKET)
        except S3Error as e:
            logger.error("MinIO bucket check failed: %s", e)

    def upload_file(
        self,
        file_bytes: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload file bytes to MinIO. Returns the object path."""
        self.ensure_bucket()
        self.client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
        )
        logger.info("Uploaded to MinIO: %s/%s (%d bytes)",
                     settings.MINIO_BUCKET, object_name, len(file_bytes))
        return f"{settings.MINIO_BUCKET}/{object_name}"

    def download_file(self, object_name: str) -> bytes:
        """Download file from MinIO by object name."""
        response = self.client.get_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
        )
        data = response.read()
        response.close()
        response.release_conn()
        return data

    def delete_file(self, object_name: str) -> None:
        """Delete a file from MinIO."""
        self.client.remove_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
        )
        logger.info("Deleted from MinIO: %s/%s", settings.MINIO_BUCKET, object_name)

    def object_exists(self, object_name: str) -> bool:
        try:
            self.client.stat_object(settings.MINIO_BUCKET, object_name)
            return True
        except S3Error:
            return False


# Singleton
minio_service = MinioService()
