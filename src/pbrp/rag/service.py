"""Provider-independent RAG application service."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pbrp.rag.models import RAGAnswer, RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_INSUFFICIENT_PREFIX = "INSUFFICIENT_EVIDENCE:"


class VectorStore(Protocol):
    """Retrieval boundary consumed by the RAG service."""

    def retrieve(self, question: str, limit: int = 5) -> list[RetrievedChunk]: ...


class LanguageModel(Protocol):
    """Grounded-generation boundary consumed by the RAG service."""

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str: ...


class RAGService:
    """Coordinate retrieval, generation, and citation validation."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        language_model: LanguageModel,
        retrieval_limit: int = 5,
    ) -> None:
        if retrieval_limit < 1:
            raise ValueError("The retrieval limit must be greater than zero.")
        self._vector_store = vector_store
        self._language_model = language_model
        self._retrieval_limit = retrieval_limit

    def answer(self, question: str) -> RAGAnswer:
        """Answer a question only when retrieved evidence supports the response."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("The question cannot be empty.")

        chunks = self._vector_store.retrieve(normalized_question, limit=self._retrieval_limit)
        if not chunks:
            return _gap_answer(
                normalized_question,
                "Azure AI Search returned no evidence chunks.",
            )

        generated_answer = self._language_model.generate(normalized_question, chunks)
        if generated_answer.startswith(_INSUFFICIENT_PREFIX):
            return _gap_answer(
                normalized_question,
                generated_answer.removeprefix(_INSUFFICIENT_PREFIX).strip(),
            )

        cited_source_ids = _ordered_unique_citations(generated_answer)
        sources_by_id = {chunk.source_id: chunk for chunk in chunks}
        invalid_citations = [
            source_id for source_id in cited_source_ids if source_id not in sources_by_id
        ]
        if invalid_citations:
            return _gap_answer(
                normalized_question,
                "The generated response cited evidence that was not retrieved: "
                + ", ".join(invalid_citations),
            )
        if not cited_source_ids:
            return _gap_answer(
                normalized_question,
                "The generated response contained no verifiable evidence citations.",
            )

        cited_sources = tuple(sources_by_id[source_id] for source_id in cited_source_ids)
        return RAGAnswer(
            question=normalized_question,
            answer=generated_answer,
            sources=cited_sources,
        )


def _ordered_unique_citations(answer: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for number in _CITATION_PATTERN.findall(answer):
        source_id = f"S{number}"
        if source_id not in seen:
            seen.add(source_id)
            ordered.append(source_id)
    return ordered


def _gap_answer(question: str, gap: str) -> RAGAnswer:
    normalized_gap = gap.strip() or "The available evidence is insufficient."
    return RAGAnswer(
        question=question,
        answer=f"Evidence is insufficient: {normalized_gap}",
        sources=(),
        gaps=(normalized_gap,),
    )


def cited_source_ids(answer: str) -> Sequence[str]:
    """Expose parsed citations for later ProductionSpec evidence validation."""

    return tuple(_ordered_unique_citations(answer))
