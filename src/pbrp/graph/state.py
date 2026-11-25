"""Typed workflow state and persisted smoke-plan contract."""

from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from pbrp.rag.models import RetrievedChunk
from pbrp.spec.models import ProductionSpec, RunSource

PlanId = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{32}$")]


class BuildPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    plan_id: PlanId
    spec: ProductionSpec
    preset: Literal["lora-cpu-v1"] | None
    dataset: Literal["synthetic"] | None
    timeout_seconds: Literal[300] = 300
    replicas: Literal[1] = 1
    retries: Literal[0] = 0
    state: Literal["awaiting_approval", "blocked"]
    gaps: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def sha256(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()


class PBRPState(MessagesState, total=False):
    run_id: str
    source: RunSource
    chunks: list[RetrievedChunk]
    spec: ProductionSpec
    plan: BuildPlan
    plan_uri: str
    result_text: str
    error: str | None
