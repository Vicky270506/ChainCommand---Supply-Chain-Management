"""Tests for AWS config helper."""

from __future__ import annotations

from unittest.mock import patch

from pydantic import SecretStr


class TestGetAwsConfig:
    def test_redshift_password_is_not_exposed(self):
        from chaincommand.aws.config import get_aws_config

        with patch("chaincommand.aws.config.settings") as mock_settings:
            mock_settings.aws_enabled = True
            mock_settings.aws_region = "us-east-1"
            mock_settings.aws_s3_bucket = "bucket"
            mock_settings.aws_s3_prefix = "prefix/"
            mock_settings.aws_redshift_host = "host"
            mock_settings.aws_redshift_port = 5439
            mock_settings.aws_redshift_db = "db"
            mock_settings.aws_redshift_user = "user"
            mock_settings.aws_redshift_password = SecretStr("super-secret")
            mock_settings.aws_redshift_iam_role = "role"
            mock_settings.aws_athena_database = "athena_db"
            mock_settings.aws_athena_output = "s3://bucket/out/"
            mock_settings.aws_quicksight_account_id = "123456789012"

            config = get_aws_config()

        assert "redshift_password" not in config
        assert config["redshift_password_set"] is True
