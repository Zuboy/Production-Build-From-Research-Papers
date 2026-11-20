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
    source_type: str | None = None
    source_uri: str | None = None
    repository_url: str | None = None
    commit_sha: str | None = None
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    page_number: int | None = None
    section: str | None = None

    @property
    def locator(self) -> str:
        """Return the best stable locator available in the current index."""

        if self.file_path and self.line_start is not None and self.line_end is not None:
            commit = f" @ {self.commit_sha}" if self.commit_sha else ""
            return f"{self.file_path}:L{self.line_start}-L{self.line_end}{commit}"
        if self.page_number is not None:
            return f"{self.title or 'paper'} / page {self.page_number}"
        if self.source_uri:
            return self.source_uri
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
                f"Source type: {self.source_type or 'Unknown'}",
                f"Locator: {self.locator}",
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
