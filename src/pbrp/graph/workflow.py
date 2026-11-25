"""Acquire, retrieve, specify, and publish a plan for human approval."""

from __future__ import annotations

import logging
from typing import Literal, Protocol
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import TypeAdapter, ValidationError

from pbrp.graph.state import BuildPlan, PBRPState, PlanId
from pbrp.smoke.contract import LORA_COMMIT, LORA_PAPER, LORA_REPOSITORY, PRESET
from pbrp.spec.models import ProductionSpec, RunId, RunSource, SpecDraft
from pbrp.spec.service import SpecModel, SpecRetriever, build_spec, retrieve_spec_evidence

_LOGGER = logging.getLogger(__name__)
_RUN_ID = TypeAdapter(RunId)
_PLAN_ID = TypeAdapter(PlanId)


class PlanningBackend(Protocol):
    @property
    def search(self) -> SpecRetriever: ...

    @property
    def llm(self) -> SpecModel: ...

    def load_source(self, run_id: str) -> RunSource: ...

    def publish_plan(self, plan_id: str, payload: str) -> str: ...

    def read_result(self, plan_id: str) -> str | None: ...


def make_graph(
    backend: PlanningBackend, default_run: str
) -> CompiledStateGraph[PBRPState, None, PBRPState, PBRPState]:
    """Build a deterministic graph. User text can select a run but cannot start a job."""

    def acquire(state: PBRPState) -> dict[str, object]:
        user = next(
            (
                item
                for item in reversed(state.get("messages", []))
                if isinstance(item, HumanMessage)
            ),
            None,
        )
        content = user.content if user is not None else ""
        parts = content.strip().split() if isinstance(content, str) else []
        if len(parts) == 2 and parts[0].lower() == "result":
            try:
                plan_id = _PLAN_ID.validate_python(parts[1])
                result = backend.read_result(plan_id)
            except ValidationError:
                return {"error": "Provide the 32-character plan ID exactly as shown."}
            except Exception:
                _LOGGER.exception("The smoke result could not be retrieved")
                return {"error": "The smoke result could not be retrieved from Blob Storage."}
            return {
                "result_text": result or "No completed experiment result exists for this plan yet."
            }
        if not parts or parts[0].lower() != "plan" or len(parts) > 2:
            return {"error": "Send `plan`, `plan <ingestion-run-id>`, or `result <plan-id>`."}
        try:
            run_id = _RUN_ID.validate_python(parts[1] if len(parts) == 2 else default_run)
        except ValidationError:
            return {"error": "The ingestion run ID is invalid."}
        try:
            return {"run_id": run_id, "source": backend.load_source(run_id)}
        except Exception:
            _LOGGER.exception("Unable to load the indexed source manifest")
            return {"error": "The ingestion manifest could not be loaded for this run."}

    def retrieve(state: PBRPState) -> dict[str, object]:
        try:
            chunks = retrieve_spec_evidence(backend.search, state["source"])
            if {chunk.source_type for chunk in chunks} >= {"paper", "repository"}:
                return {"chunks": chunks}
            return {"error": "The index did not return evidence for both pinned sources."}
        except Exception:
            _LOGGER.exception("Azure AI Search could not retrieve run evidence")
            return {"error": "Azure AI Search could not retrieve evidence for this run."}

    def specify(state: PBRPState) -> dict[str, object]:
        try:
            return {
                "spec": build_spec(state["source"], state["chunks"], backend.llm),
                "chunks": [],
            }
        except Exception:
            _LOGGER.exception("Foundry could not prepare the structured specification")
            return {"error": "Foundry could not prepare the structured specification."}

    def plan(state: PBRPState) -> dict[str, object]:
        spec = state["spec"]
        eligible = (
            spec.source.repository_url == LORA_REPOSITORY
            and spec.source.commit_sha == LORA_COMMIT
            and spec.source.paper_sha256 == LORA_PAPER
            and spec.source.run_mode == "smoke"
        )
        gaps = _gaps(spec)
        if not eligible:
            gaps += ("The run does not qualify for the configured LoRA CPU smoke preset.",)
        artifact = BuildPlan(
            plan_id=uuid4().hex,
            spec=spec,
            preset=PRESET if eligible else None,
            dataset="synthetic" if eligible else None,
            state="awaiting_approval" if eligible else "blocked",
            gaps=gaps,
        )
        try:
            uri = backend.publish_plan(artifact.plan_id, artifact.model_dump_json())
        except Exception:
            _LOGGER.exception("The experiment plan could not be saved")
            return {"error": "The experiment plan could not be saved to Blob Storage."}
        return {"plan": artifact, "plan_uri": uri}

    def report(state: PBRPState) -> dict[str, object]:
        if "result_text" in state:
            return {"messages": [AIMessage(content=state["result_text"])]}
        error = state.get("error")
        if error:
            return {"messages": [AIMessage(content=f"**Blocked.** {error}")]}

        artifact = state["plan"]
        lines = [
            f"**Status:** `{artifact.state}`",
            f"Ingestion run: `{artifact.spec.source.run_id}`",
            f"Plan ID: `{artifact.plan_id}`",
            f"Plan SHA-256: `{artifact.sha256}`",
            f"[Open the saved plan and evidence]({state['plan_uri']})",
        ]
        if artifact.preset:
            lines.append("Preset: LoRA on CPU with synthetic data, one replica, five-minute limit.")
            lines.append(
                "After reviewing the saved plan, submit it with the Azure-signed approval CLI."
            )
        if artifact.gaps:
            lines.extend(("", "Reproduction gaps:", *(f"- {gap}" for gap in artifact.gaps)))
        return {"messages": [AIMessage(content="\n".join(lines))]}

    def after_acquire(state: PBRPState) -> Literal["retrieve", "report"]:
        return "report" if state.get("error") or "result_text" in state else "retrieve"

    def after_retrieve(state: PBRPState) -> Literal["specify", "report"]:
        return "report" if state.get("error") else "specify"

    def after_specify(state: PBRPState) -> Literal["plan", "report"]:
        return "report" if state.get("error") else "plan"

    workflow = StateGraph(PBRPState)
    workflow.add_node("acquire", acquire)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("specify", specify)
    workflow.add_node("plan", plan)
    workflow.add_node("report", report)
    workflow.add_edge(START, "acquire")
    workflow.add_conditional_edges("acquire", after_acquire)
    workflow.add_conditional_edges("retrieve", after_retrieve)
    workflow.add_conditional_edges("specify", after_specify)
    workflow.add_edge("plan", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def _gaps(spec: ProductionSpec) -> tuple[str, ...]:
    return tuple(gap for field in SpecDraft.model_fields if (gap := getattr(spec, field).gap))
