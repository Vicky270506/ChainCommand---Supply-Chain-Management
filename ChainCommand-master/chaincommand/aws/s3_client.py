"""S3 operations for ChainCommand data persistence."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import settings
from ..utils.logging_config import get_logger

log = get_logger(__name__)


class S3Client:
    """Encapsulates S3 upload/download operations."""

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        import boto3

        self._bucket = bucket or settings.aws_s3_bucket
        self._prefix = (prefix or settings.aws_s3_prefix).rstrip("/")
        self._client = boto3.client("s3", region_name=region or settings.aws_region)

    def upload_dataframe(self, df: pd.DataFrame, key: str) -> str:
        """Upload a DataFrame as Parquet to S3."""
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        log.info("s3_upload_parquet", bucket=self._bucket, key=key, rows=len(df))
        return f"s3://{self._bucket}/{key}"

    def upload_jsonl(self, records: List[Dict[str, Any]], key: str) -> str:
        """Upload a list of dicts as JSONL to S3."""
        if not records:
            log.warning("s3_upload_jsonl_empty_records", key=key)
            return f"s3://{self._bucket}/{key}"
        lines = "\n".join(json.dumps(r, default=str) for r in records) + "\n"
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=lines.encode("utf-8"),
            ContentType="application/jsonl",
        )
        log.info("s3_upload_jsonl", bucket=self._bucket, key=key, records=len(records))
        return f"s3://{self._bucket}/{key}"

    def upload_json(self, data: Any, key: str) -> str:
        """Upload a single JSON object to S3."""
        body = json.dumps(data, default=str, indent=2)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        log.info("s3_upload_json", bucket=self._bucket, key=key)
        return f"s3://{self._bucket}/{key}"

    def list_objects(self, prefix: str) -> List[Dict[str, Any]]:
        """List all objects under a given S3 prefix, handling pagination.

        S3 ``list_objects_v2`` returns at most 1 000 keys per response.
        This method follows ``ContinuationToken`` pages until all objects
        have been collected.
        """
        full_prefix = f"{self._prefix}/{prefix}"
        all_objects: List[Dict[str, Any]] = []
        continuation_token: Optional[str] = None

        while True:
            kwargs: Dict[str, Any] = {
                "Bucket": self._bucket,
                "Prefix": full_prefix,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            resp = self._client.list_objects_v2(**kwargs)
            contents = resp.get("Contents", [])
            all_objects.extend(
                {"key": obj["Key"], "size": obj["Size"], "last_modified": obj["LastModified"]}
                for obj in contents
            )

            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

        return all_objects

    def download_json(self, key: str) -> Any:
        """Download and parse a JSON object from S3."""
        resp = self._client.get_object(Bucket=self._bucket, Key=key)
        stream = resp["Body"]
        try:
            body = stream.read().decode("utf-8")
        finally:
            stream.close()
        return json.loads(body)
