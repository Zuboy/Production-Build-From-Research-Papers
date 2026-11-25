"""Typed contracts shared by retrieval, generation, and orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A human-readable evidence chunk returned by the retrieval provider."""

    source_id: str
    chunk_id: str
    parent_id: str | None
    title: str | None
    content: str
    search_score: float | None = None
    reranker_score: float | None = None
    source_uri: str | None = None
    source_type: str | None = None

    @property
    def locator(self) -> str:
        """Return the best stable locator available in the current index."""

        if self.title:
            return f"{self.title} / chunk {self.chunk_id}"
        return f"chunk {self.chunk_id}"

    def to_prompt_block(self) -> str:
        """Serialize this chunk as explicitly delimited, untrusted evidence."""

        return "\n".join(
            (
                f"<{self.source_id}>",
                f"Title: {self.title or 'Unknown'}",
                f"Chunk ID: {self.chunk_id}",
                f"Parent ID: {self.parent_id or 'Unknown'}",
                "Content:",
                self.content,
                f"</{self.source_id}>",
            )
        )


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Grounded answer plus the evidence that was actually cited."""

    question: str
    answer: str
    sources: tuple[RetrievedChunk, ...]
    gaps: tuple[str, ...] = ()

    @property
    def is_grounded(self) -> bool:
        return bool(self.sources) and not self.gaps
