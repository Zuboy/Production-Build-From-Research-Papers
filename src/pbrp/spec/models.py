"""Validated contracts shared by specification extraction and experiment planning."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

RunId = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40}$")]


class RunSource(BaseModel):
    """The input identity recorded by the existing ingestion service."""

    model_config = ConfigDict(frozen=True)

    run_id: RunId
    run_mode: str
    paper_sha256: Sha256
    paper_blob_uri: str
    repository_url: str
    commit_sha: CommitSha
    archive_sha256: Sha256
    archive_blob_uri: str
    knowledge_index_name: str
    paper_chunk_count: int = Field(gt=0)
    repository_chunk_count: int = Field(gt=0)


class DraftFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    source_id: str
    quote: str


class SpecDraft(BaseModel):
    """Structured output for the Foundry model. An absent fact is explicitly null."""

    model_config = ConfigDict(extra="forbid")

    task: DraftFact | None
    framework: DraftFact | None
    python_version: DraftFact | None
    dependencies: DraftFact | None
    install_command: DraftFact | None
    experiment_command: DraftFact | None
    dataset: DraftFact | None
    metric: DraftFact | None
    hardware: DraftFact | None
    serving_interface: DraftFact | None


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=200)
    source_type: str
    search_score: float | None
    reranker_score: float | None


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str | None
    evidence: Evidence | None
    confidence: float = Field(ge=0, le=1)
    gap: str | None

    @model_validator(mode="after")
    def check_grounding(self) -> Fact:
        if self.value is None:
            if not self.gap or self.evidence is not None or self.confidence != 0:
                raise ValueError("A missing value requires an explicit gap and zero confidence.")
        elif (
            not self.value.strip()
            or self.evidence is None
            or self.value not in self.evidence.quote
            or self.confidence == 0
            or self.gap is not None
        ):
            raise ValueError("A populated value requires matching evidence and confidence.")
        return self


class ProductionSpec(BaseModel):
    """An immutable, versioned specification for a pinned source pair."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    source: RunSource
    task: Fact
    framework: Fact
    python_version: Fact
    dependencies: Fact
    install_command: Fact
    experiment_command: Fact
    dataset: Fact
    metric: Fact
    hardware: Fact
    serving_interface: Fact
