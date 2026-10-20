"""Prometheus metrics for ChainCommand."""
from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# --- Metrics Definitions ---

if HAS_PROMETHEUS:
    REQUEST_COUNT = Counter(
        "chaincommand_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "chaincommand_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    SIMULATION_CYCLES = Counter(
        "chaincommand_simulation_cycles_total",
        "Total simulation cycles completed",
    )
    KPI_GAUGE = Gauge(
        "chaincommand_kpi_value",
        "Current KPI metric value",
        ["metric_name"],
    )
    RL_REWARD = Gauge(
        "chaincommand_rl_mean_reward",
        "RL inventory policy mean reward",
    )
    RISK_HIGH_COUNT = Gauge(
        "chaincommand_high_risk_suppliers",
        "Number of high-risk suppliers",
    )
    CTB_CLEAR_PCT = Gauge(
        "chaincommand_ctb_clear_percentage",
        "CTB clear percentage",
        ["assembly_id"],
    )
    CPSAT_SOLVE_TIME = Histogram(
        "chaincommand_cpsat_solve_time_ms",
        "CP-SAT solver time in milliseconds",
        buckets=[10, 50, 100, 500, 1000, 5000, 10000],
    )
    FORECAST_MAPE = Gauge(
        "chaincommand_forecast_mape",
        "Current forecast MAPE",
        ["product_id"],
    )
    ERROR_COUNT = Counter(
        "chaincommand_errors_total",
        "Total errors",
        ["error_type", "component"],
    )
    APP_INFO = Info("chaincommand", "Application info")


def get_metrics_response() -> tuple[bytes, str]:
    """Generate Prometheus metrics response."""
    if not HAS_PROMETHEUS:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


def track_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """Record an HTTP request metric."""
    if not HAS_PROMETHEUS:
        return
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def track_kpi(metric_name: str, value: float) -> None:
    """Update a KPI gauge."""
    if not HAS_PROMETHEUS:
        return
    KPI_GAUGE.labels(metric_name=metric_name).set(value)


def track_error(error_type: str, component: str) -> None:
    """Record an error."""
    if not HAS_PROMETHEUS:
        return
    ERROR_COUNT.labels(error_type=error_type, component=component).inc()


def set_app_info(version: str) -> None:
    """Set application info."""
    if not HAS_PROMETHEUS:
        return
    APP_INFO.info({"version": version})
