"""QuickSight operations for ChainCommand dashboard creation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..config import settings
from ..utils.logging_config import get_logger

log = get_logger(__name__)

# ── Default dataset definitions ─────────────────────────

DEFAULT_DATASETS = [
    {
        "name": "KPI Trends",
        "description": "Historical KPI metrics from Redshift",
        "source_type": "redshift",
        "sql": (
            "SELECT cycle, timestamp, otif, fill_rate, mape, dsi, "
            "stockout_count, total_inventory_value, carrying_cost, "
            "order_cycle_time, perfect_order_rate, inventory_turnover, "
            "backorder_rate, supplier_defect_rate "
            "FROM kpi_snapshots ORDER BY cycle"
        ),
    },
    {
        "name": "Demand Patterns",
        "description": "Demand history analytics from Athena on S3",
        "source_type": "athena",
        "sql": (
            "SELECT date, product_id, quantity, is_promotion, day_of_week, month "
            "FROM demand_history ORDER BY date"
        ),
    },
    {
        "name": "Event Analytics",
        "description": "Supply chain event analytics from Athena on S3",
        "source_type": "athena",
        "sql": (
            "SELECT event_type, severity, source_agent, timestamp, description "
            "FROM events ORDER BY timestamp DESC"
        ),
    },
]


class QuickSightClient:
    """Encapsulates AWS QuickSight dataset and dashboard operations."""

    def __init__(
        self,
        account_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        import boto3

        self._account_id = account_id or settings.aws_quicksight_account_id
        self._region = region or settings.aws_region
        self._client = boto3.client(
            "quicksight", region_name=self._region
        )

    def _ensure_data_source_arn(self, source_ref: str) -> str:
        """Accept either a data source ARN or ID."""
        if source_ref.startswith("arn:"):
            return source_ref
        return f"arn:aws:quicksight:{self._region}:{self._account_id}:datasource/{source_ref}"

    def _ensure_dataset_arn(self, dataset_ref: str) -> str:
        """Accept either a dataset ARN or ID."""
        if dataset_ref.startswith("arn:"):
            return dataset_ref
        return f"arn:aws:quicksight:{self._region}:{self._account_id}:dataset/{dataset_ref}"

    def create_data_source(
        self, name: str, source_type: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a QuickSight data source (Athena or Redshift)."""
        ds_id = f"cc-{name.lower().replace(' ', '-')}-{uuid4()}"

        _SUPPORTED_TYPES = {"athena", "redshift"}
        if source_type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported source_type '{source_type}'. Must be one of {_SUPPORTED_TYPES}"
            )

        params: Dict[str, Any] = {}
        kwargs: Dict[str, Any] = {}
        if source_type == "athena":
            params["AthenaParameters"] = {"WorkGroup": config.get("workgroup", "primary")}
        elif source_type == "redshift":
            params["RedshiftParameters"] = {
                "Host": config.get("host", settings.aws_redshift_host),
                "Port": config.get("port", settings.aws_redshift_port),
                "Database": config.get("database", settings.aws_redshift_db),
            }
            # Redshift data sources require credentials — read password from
            # settings directly to avoid leaking secrets through config dicts.
            kwargs["Credentials"] = {
                "CredentialPair": {
                    "Username": config.get("user", settings.aws_redshift_user),
                    "Password": settings.aws_redshift_password.get_secret_value(),
                }
            }

        resp = self._client.create_data_source(
            AwsAccountId=self._account_id,
            DataSourceId=ds_id,
            Name=name,
            Type=source_type.upper(),
            DataSourceParameters=params,
            **kwargs,
        )
        log.info("quicksight_data_source_created", name=name, ds_id=ds_id)
        return {"data_source_id": ds_id, "arn": resp.get("Arn", ""), "status": resp.get("CreationStatus", "")}

    def create_dataset(
        self, name: str, source_id: str, sql: str,
        columns: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Create a QuickSight dataset backed by a custom SQL query.

        Args:
            columns: List of ``{"Name": "col", "Type": "STRING"}`` dicts.
                     Required by the QuickSight API for CustomSql.
                     Defaults to a single placeholder column if not provided.
        """
        dataset_id = f"cc-ds-{name.lower().replace(' ', '-')}-{uuid4()}"
        source_arn = self._ensure_data_source_arn(source_id)

        if columns is None:
            # Provide a sensible default so the API call doesn't fail
            columns = [{"Name": "id", "Type": "STRING"}]

        resp = self._client.create_data_set(
            AwsAccountId=self._account_id,
            DataSetId=dataset_id,
            Name=name,
            PhysicalTableMap={
                "main": {
                    "CustomSql": {
                        "DataSourceArn": source_arn,
                        "Name": name,
                        "SqlQuery": sql,
                        "Columns": columns,
                    }
                }
            },
            ImportMode="DIRECT_QUERY",
        )
        log.info("quicksight_dataset_created", name=name, dataset_id=dataset_id)
        return {"dataset_id": dataset_id, "arn": resp.get("Arn", ""), "status": resp.get("Status", "")}

    def create_dashboard(
        self, name: str, dataset_ids: List[str]
    ) -> Dict[str, Any]:
        """Create a QuickSight dashboard from datasets (template-based)."""
        dashboard_id = f"cc-dash-{name.lower().replace(' ', '-')}-{uuid4()}"

        # Build dataset references
        dataset_refs = [
            {
                "DataSetPlaceholder": f"ds-{i}",
                "DataSetArn": self._ensure_dataset_arn(ds_id),
            }
            for i, ds_id in enumerate(dataset_ids)
        ]

        resp = self._client.create_dashboard(
            AwsAccountId=self._account_id,
            DashboardId=dashboard_id,
            Name=name,
            SourceEntity={
                "SourceTemplate": {
                    "DataSetReferences": dataset_refs,
                    "Arn": f"arn:aws:quicksight:{settings.aws_region}:{self._account_id}:template/cc-default-template",
                }
            },
        )
        log.info("quicksight_dashboard_created", name=name, dashboard_id=dashboard_id)
        return {
            "dashboard_id": dashboard_id,
            "arn": resp.get("Arn", ""),
            "status": resp.get("CreationStatus", ""),
        }

    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all QuickSight dashboards in the account (handles pagination)."""
        all_summaries: List[Dict[str, Any]] = []
        kwargs: Dict[str, Any] = {"AwsAccountId": self._account_id}

        while True:
            resp = self._client.list_dashboards(**kwargs)
            summaries = resp.get("DashboardSummaryList", [])
            all_summaries.extend(
                {
                    "dashboard_id": d.get("DashboardId", ""),
                    "name": d.get("Name", ""),
                    "arn": d.get("Arn", ""),
                    "published_version": d.get("PublishedVersionNumber", 0),
                    "last_updated": str(d.get("LastUpdatedTime", "")),
                }
                for d in summaries
            )
            next_token = resp.get("NextToken")
            if not next_token:
                break
            kwargs["NextToken"] = next_token

        return all_summaries
