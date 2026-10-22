"""AWS-specific configuration helpers."""

from __future__ import annotations

from ..config import settings


def get_aws_config() -> dict:
    """Return AWS-related settings as a plain dict."""
    return {
        "enabled": settings.aws_enabled,
        "region": settings.aws_region,
        "s3_bucket": settings.aws_s3_bucket,
        "s3_prefix": settings.aws_s3_prefix,
        "redshift_host": settings.aws_redshift_host,
        "redshift_port": settings.aws_redshift_port,
        "redshift_db": settings.aws_redshift_db,
        "redshift_user": settings.aws_redshift_user,
        "redshift_password_set": bool(settings.aws_redshift_password.get_secret_value()),
        "redshift_iam_role": settings.aws_redshift_iam_role,
        "athena_database": settings.aws_athena_database,
        "athena_output": settings.aws_athena_output,
        "quicksight_account_id": settings.aws_quicksight_account_id,
    }
