"""Data quality validation using Great Expectations.

Uses the modern v0.18+ ``get_context()`` / ``sources.pandas_default`` API
consistently throughout.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
    from great_expectations.core.batch_definition import BatchDefinition

    HAS_GX = True
except ImportError:
    HAS_GX = False

EXPECTATIONS_DIR = Path(__file__).parent / "expectations"


def validate_dataframe(df: Any, expectation_suite: str) -> dict[str, Any]:
    """Validate a pandas DataFrame against an expectation suite.

    Uses the modern GX v0.18+ fluent datasource API:
      context.sources.pandas_default → add_dataframe_asset → build_batch_request
      context.add_expectation_suite (from JSON) → checkpoint.run()
    """
    if not HAS_GX:
        logger.warning("great_expectations not installed — skipping validation")
        return {
            "success": True,
            "skipped": True,
            "reason": "great_expectations not installed",
        }

    suite_path = EXPECTATIONS_DIR / f"{expectation_suite}.json"
    if not suite_path.exists():
        return {"success": False, "error": f"Suite not found: {expectation_suite}"}

    context = gx.get_context()

    # --- Datasource (fluent API) --------------------------------
    datasource = context.sources.add_or_update_pandas("pandas_source")
    asset = datasource.add_dataframe_asset(name="validation_asset")
    batch_request = asset.build_batch_request(dataframe=df)

    # --- Expectation suite from JSON ----------------------------
    with open(suite_path) as fh:
        suite_dict = json.load(fh)
    suite = context.add_or_update_expectation_suite(
        expectation_suite=ExpectationSuite(**suite_dict),
    )

    # --- Checkpoint (modern replacement for validation_operator) -
    checkpoint = context.add_or_update_checkpoint(
        name=f"ck_{expectation_suite}",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite.expectation_suite_name,
            },
        ],
    )
    results = checkpoint.run()

    return {
        "success": results.success,
        "statistics": results.statistics if hasattr(results, "statistics") else {},
        "suite": expectation_suite,
    }


def validate_demand_history(df: Any) -> dict[str, Any]:
    """Validate demand history data."""
    return validate_dataframe(df, "demand_history")


def validate_inventory_status(df: Any) -> dict[str, Any]:
    """Validate inventory status data."""
    return validate_dataframe(df, "inventory_status")
