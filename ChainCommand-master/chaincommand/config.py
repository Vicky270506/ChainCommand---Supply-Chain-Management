"""Central configuration using Pydantic Settings."""

from __future__ import annotations

import logging

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings

_config_log = logging.getLogger(__name__)

_DEFAULT_API_KEY = "dev-key-change-me"


class Settings(BaseSettings):
    """Application-wide settings loaded from env / .env file."""

    model_config = {"env_file": ".env", "env_prefix": "CC_"}

    # ── Environment mode ──────────────────────────────────────
    # Set CC_ENV=production to enforce a real API key at startup.
    env: str = "development"

    # ── Training ──────────────────────────────────────────────
    max_train_products: int = 20

    # ── Simulation ───────────────────────────────────────
    num_products: int = 50
    num_suppliers: int = 20
    history_days: int = 365
    simulation_speed: float = 1.0
    simulation_speed_min: float = 0.1
    simulation_speed_max: float = 100.0

    # ── Event engine ─────────────────────────────────────
    event_tick_seconds: float = 5.0
    enable_proactive_monitoring: bool = True

    # ── KPI thresholds ───────────────────────────────────
    otif_target: float = 0.95
    fill_rate_target: float = 0.97
    mape_threshold: float = 15.0
    dsi_max: float = 60.0
    dsi_min: float = 10.0
    stockout_tolerance: int = 3

    # ── Escalation / HITL ────────────────────────────────
    cost_escalation_threshold: float = 50_000.0
    inventory_change_pct_threshold: float = 25.0
    auto_approve_below: float = 10_000.0

    # ── Security ─────────────────────────────────────────
    # WARNING: The default API key is for local development only.
    # In production (CC_ENV=production), set CC_API_KEY to a real secret.
    api_key: SecretStr = SecretStr(_DEFAULT_API_KEY)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    rate_limit_per_minute: int = 60

    # ── Reproducibility ──────────────────────────────────
    random_seed: int = 42

    # ── Server ───────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ── AWS ──────────────────────────────────────────────
    aws_enabled: bool = False
    aws_region: str = "ap-northeast-1"
    aws_s3_bucket: str = "chaincommand-data"
    aws_s3_prefix: str = "supply-chain/"
    aws_redshift_host: str = ""
    aws_redshift_port: int = 5439
    aws_redshift_db: str = "chaincommand"
    aws_redshift_user: str = ""
    aws_redshift_password: SecretStr = SecretStr("")  # Prefer IAM-based auth; see aws_redshift_iam_role
    aws_redshift_iam_role: str = ""  # Recommended: set this instead of password
    aws_athena_database: str = "chaincommand"
    aws_athena_output: str = "s3://chaincommand-data/athena-results/"
    aws_quicksight_account_id: str = ""

    # ── KPI engine ─────────────────────────────────────────
    kpi_max_history: int = 1000

    # ── ML model params ─────────────────────────────────
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_seq_length: int = 30
    lstm_epochs: int = 50
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    isolation_contamination: float = 0.05
    ga_population_size: int = 50
    ga_generations: int = 100
    dqn_hidden_size: int = 128
    dqn_episodes: int = 200
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.01
    dqn_epsilon_decay: float = 0.995

    # ── CP-SAT Optimization ─────────────────────────────
    ortools_time_limit_ms: int = 10_000
    ortools_risk_lambda: float = 0.3
    ortools_max_suppliers: int = 5

    # ── RL Inventory Policy ──────────────────────────────
    rl_total_timesteps: int = 50_000
    rl_episode_length: int = 90
    rl_holding_cost: float = 0.5
    rl_stockout_cost: float = 10.0
    rl_ordering_cost_fixed: float = 50.0

    # ── BOM Management ───────────────────────────────────
    bom_default_assemblies: int = 5
    bom_long_lead_threshold_days: int = 14

    # ── CTB (Clear-to-Build) ─────────────────────────────
    ctb_default_build_qty: float = 100.0


    @field_validator("env", mode="before")
    @classmethod
    def _normalize_env(cls, v: str) -> str:
        """Strip whitespace and lowercase the environment value."""
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: object) -> list[str]:
        """Accept a comma-separated string from env and split into a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v  # type: ignore[return-value]

    @model_validator(mode="after")
    def _validate_api_key(self) -> "Settings":
        """Reject the default API key in production/staging; warn in development."""
        if self.api_key.get_secret_value() == _DEFAULT_API_KEY:
            env_lower = self.env  # already normalized by _normalize_env
            if env_lower == "production":
                raise ValueError(
                    "CC_API_KEY must be changed from the default in production. "
                    "Set CC_API_KEY to a strong, unique secret."
                )
            if env_lower == "staging":
                raise ValueError(
                    "CC_API_KEY must be changed from the default in staging. "
                    "Set CC_API_KEY to a strong, unique secret before deploying to staging."
                )
            if env_lower == "development":
                _config_log.warning(
                    "Using default API key in development — "
                    "set CC_API_KEY to a real secret before deploying.",
                )
            else:
                _config_log.warning(
                    "Using default API key in '%s' environment — "
                    "set CC_API_KEY before deploying.",
                    self.env,
                )
        return self

    @model_validator(mode="after")
    def _validate_redshift_password(self) -> "Settings":
        """Warn when Redshift password is used without IAM role in non-dev envs.

        Recommended: Use IAM-based authentication instead of plaintext passwords
        for Redshift connections. Set CC_AWS_REDSHIFT_IAM_ROLE to an IAM role ARN
        and leave CC_AWS_REDSHIFT_PASSWORD empty.
        """
        env_lower = self.env  # already normalized by _normalize_env
        if self.aws_enabled and self.aws_redshift_password.get_secret_value():
            if not self.aws_redshift_iam_role:
                if env_lower == "production":
                    _config_log.warning(
                        "SECURITY: Redshift password is set in production without an "
                        "IAM role. Plaintext passwords are insecure. Set "
                        "CC_AWS_REDSHIFT_IAM_ROLE and remove CC_AWS_REDSHIFT_PASSWORD.",
                    )
                elif env_lower != "development":
                    _config_log.warning(
                        "Redshift password is set in '%s' without an IAM role. "
                        "Consider using IAM-based auth (CC_AWS_REDSHIFT_IAM_ROLE) "
                        "instead of plaintext passwords.",
                        self.env,
                    )
        if self.aws_enabled and env_lower == "production":
            if not self.aws_redshift_iam_role and not self.aws_redshift_password.get_secret_value():
                _config_log.info(
                    "Neither Redshift IAM role nor password configured. "
                    "Redshift features will be unavailable.",
                )
        return self


settings = Settings()
