"""Build a ProductionSpec from scoped retrieval and verified quotations."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from pbrp.rag.models import RetrievedChunk
from pbrp.spec.models import DraftFact, Evidence, Fact, ProductionSpec, RunSource, SpecDraft

QUERIES = (
    ("paper", "What task, dataset, evaluation metric and hardware does the paper report?"),
    ("repository", "What Python version, framework and dependencies are documented?"),
    ("repository", "Which complete commands install dependencies or run the experiment?"),
    ("repository", "Which dataset, metric, hardware and serving interface are documented?"),
)


class SpecRetriever(Protocol):
    def retrieve_source(
        self, query: str, source: RunSource, source_type: str
    ) -> list[RetrievedChunk]: ...


class SpecModel(Protocol):
    def extract(self, chunks: list[RetrievedChunk]) -> SpecDraft: ...


def retrieve_spec_evidence(retriever: SpecRetriever, source: RunSource) -> list[RetrievedChunk]:
    chunks = {
        chunk.chunk_id: chunk
        for source_type, query in QUERIES
        for chunk in retriever.retrieve_source(query, source, source_type)
    }
    return [replace(chunk, source_id=f"S{i}") for i, chunk in enumerate(chunks.values(), start=1)]


def build_spec(source: RunSource, chunks: list[RetrievedChunk], model: SpecModel) -> ProductionSpec:
    draft = model.extract(chunks) if chunks else None
    by_label = {key: chunk for chunk in chunks for key in (chunk.source_id, chunk.chunk_id)}
    fields = {
        name: _verify_fact(name, getattr(draft, name, None), by_label)
        for name in SpecDraft.model_fields
    }
    return ProductionSpec.model_validate({"source": source, **fields})


def _verify_fact(name: str, draft: DraftFact | None, chunks: dict[str, RetrievedChunk]) -> Fact:
    gap = f"No verifiable {name.replace('_', ' ')} was found in the retrieved evidence."
    if draft is not None:
        chunk = chunks.get(draft.source_id)
        value, quote = draft.value.strip(), draft.quote.strip()
        is_command = name in {"install_command", "experiment_command"}
        if (
            chunk is not None
            and chunk.source_uri
            and chunk.source_type
            and value
            and 0 < len(quote) <= 200
            and value in quote
            and quote in chunk.content
            and (
                not is_command
                or (
                    chunk.source_type == "repository"
                    and value == quote
                    and not value.endswith("\\")
                    and any(line.strip() == value for line in chunk.content.splitlines())
                )
            )
        ):
            return Fact(
                value=value,
                evidence=Evidence(
                    chunk_id=chunk.chunk_id,
                    locator=chunk.source_uri,
                    quote=quote,
                    source_type=chunk.source_type,
                    search_score=chunk.search_score,
                    reranker_score=chunk.reranker_score,
                ),
                confidence=0.9,
                gap=None,
            )
    return Fact(value=None, evidence=None, confidence=0, gap=gap)
