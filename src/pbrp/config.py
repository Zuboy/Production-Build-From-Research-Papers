
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when required application configuration is invalid or missing."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration required by the Azure RAG path."""

    azure_ai_project_endpoint: str
    azure_ai_model_deployment_name: str
    azure_search_endpoint: str
    azure_search_index_name: str
    semantic_configuration_name: str = "rag-rp-v1-semantic-configuration"
    vector_field_name: str = "text_vector"
    retrieval_limit: int = 5
    vector_candidate_count: int = 50

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings without accepting secrets or API keys."""

        settings = cls(
            azure_ai_project_endpoint=_required_env("AZURE_AI_PROJECT_ENDPOINT"),
            azure_ai_model_deployment_name=_required_env("AZURE_AI_MODEL_DEPLOYMENT_NAME"),
            azure_search_endpoint=_required_env("AZURE_SEARCH_ENDPOINT"),
            azure_search_index_name=_required_env("AZURE_SEARCH_INDEX_NAME"),
            semantic_configuration_name=os.getenv(
                "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
                "rag-rp-v1-semantic-configuration",
            ).strip(),
            vector_field_name=os.getenv("AZURE_SEARCH_VECTOR_FIELD", "text_vector").strip(),
            retrieval_limit=_positive_int_env("PBRP_RETRIEVAL_LIMIT", 5),
            vector_candidate_count=_positive_int_env("PBRP_VECTOR_CANDIDATE_COUNT", 50),
        )
        _validate_https_url("AZURE_AI_PROJECT_ENDPOINT", settings.azure_ai_project_endpoint)
        _validate_https_url("AZURE_SEARCH_ENDPOINT", settings.azure_search_endpoint)
        if settings.vector_candidate_count < settings.retrieval_limit:
            raise ConfigurationError(
                "PBRP_VECTOR_CANDIDATE_COUNT must be greater than or equal to PBRP_RETRIEVAL_LIMIT."
            )
        return settings


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Required environment variable {name} is not set.")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error
    if value < 1:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


def _validate_https_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigurationError(f"{name} must be a valid HTTPS URL.")
