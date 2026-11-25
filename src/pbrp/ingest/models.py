"""Provider-independent contracts for safe source ingestion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    """A validated UTF-8 source file read without executing repository code."""

    path: str
    content: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    """An immutable, commit-pinned repository archive and its safe text files."""

    repository_url: str
    commit_sha: str
    archive_name: str
    archive_bytes: bytes = field(repr=False)
    archive_sha256: str
    files: tuple[RepositoryFile, ...]
    skipped_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A paper or repository chunk ready to embed and index."""

    chunk_id: str
    parent_id: str
    source_type: str
    document_id: str
    title: str
    content: str
    source_uri: str
    content_hash: str
    repository_url: str | None = None
    commit_sha: str | None = None
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    page_number: int | None = None
    section: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionManifest:
    """Run-scoped record of the exact sources prepared for indexing."""

    run_id: str
    run_mode: str
    paper_name: str
    paper_sha256: str
    paper_blob_uri: str
    repository_url: str
    commit_sha: str
    archive_sha256: str
    archive_blob_uri: str
    repository_file_count: int
    repository_chunk_count: int
    skipped_files: tuple[str, ...]
    knowledge_index_name: str | None = None
    paper_chunk_count: int = 0
    indexed_chunk_count: int = 0

    def to_json(self) -> str:
        """Serialize deterministically for reproducible artifact comparisons."""

        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
