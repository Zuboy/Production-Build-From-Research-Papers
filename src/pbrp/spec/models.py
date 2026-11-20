"""Immutable contracts for evidence-cited production requirements."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evidence(ImmutableModel):
    """One exact, validated source excerpt supporting a requirement."""

    source_id: str = Field(min_length=2)
    source_type: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=200)
    search_score: float | None = None
    reranker_score: float | None = None


class TextRequirement(ImmutableModel):
    """A scalar fact that is either evidenced or explicitly missing."""

    value: str | None
    evidence: tuple[Evidence, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    gap: str | None = None

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> TextRequirement:
        if self.value is None:
            if self.evidence or self.confidence != 0 or not self.gap:
                raise ValueError("Missing text values require zero confidence and an explicit gap.")
        elif not self.value.strip() or not self.evidence or self.confidence <= 0 or self.gap:
            raise ValueError("Populated text values require evidence and positive confidence.")
        return self


class ListRequirement(ImmutableModel):
    """A list fact that is either evidenced or explicitly missing."""

    value: tuple[str, ...] | None
    evidence: tuple[Evidence, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    gap: str | None = None

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> ListRequirement:
        if self.value is None:
            if self.evidence or self.confidence != 0 or not self.gap:
                raise ValueError("Missing list values require zero confidence and an explicit gap.")
        elif (
            not self.value
            or any(not item.strip() for item in self.value)
            or not self.evidence
            or self.confidence <= 0
            or self.gap
        ):
            raise ValueError("Populated list values require evidence and positive confidence.")
        return self


class SourceFingerprint(ImmutableModel):
    """Directly observed identities for the paper and repository inputs."""

    paper_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_url: str = Field(pattern=r"^https://github\.com/")
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")


class ProductionSpec(ImmutableModel):
    """Minimal build contract produced before planning or execution."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: SourceFingerprint
    task: TextRequirement
    language: TextRequirement
    framework: TextRequirement
    python_version: TextRequirement
    dependencies: ListRequirement
    install_commands: ListRequirement
    smoke_commands: ListRequirement
    entry_points: ListRequirement
    datasets: ListRequirement
    metrics: ListRequirement
    hardware: ListRequirement
    serving_interface: TextRequirement
    gaps: tuple[str, ...] = ()

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"


class DraftEvidence(ImmutableModel):
    """Untrusted citation proposed by the model for code-side validation."""

    source_id: str
    quote: str = Field(max_length=200)


class DraftTextRequirement(ImmutableModel):
    value: str | None
    evidence: tuple[DraftEvidence, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    gap: str | None


class DraftListRequirement(ImmutableModel):
    value: tuple[str, ...] | None
    evidence: tuple[DraftEvidence, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    gap: str | None


class ProductionSpecDraft(ImmutableModel):
    """Structured model output that cannot become a spec without validation."""

    task: DraftTextRequirement
    language: DraftTextRequirement
    framework: DraftTextRequirement
    python_version: DraftTextRequirement
    dependencies: DraftListRequirement
    install_commands: DraftListRequirement
    smoke_commands: DraftListRequirement
    entry_points: DraftListRequirement
    datasets: DraftListRequirement
    metrics: DraftListRequirement
    hardware: DraftListRequirement
    serving_interface: DraftTextRequirement
