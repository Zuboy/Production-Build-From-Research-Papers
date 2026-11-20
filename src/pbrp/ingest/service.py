"""Application service for safe paper and repository ingestion."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from pbrp.ingest.chunking import chunk_repository
from pbrp.ingest.models import IngestionManifest, KnowledgeChunk, RepositorySnapshot


class RepositorySource(Protocol):
    def acquire(self, repository_url: str, commit_sha: str) -> RepositorySnapshot: ...


class ArtifactStore(Protocol):
    def upload_bytes(
        self,
        *,
        container_name: str,
        blob_name: str,
        data: bytes,
        content_type: str,
        overwrite: bool = False,
    ) -> str: ...

    def upload_local_file(
        self,
        *,
        container_name: str,
        blob_name: str,
        local_path: Path,
        content_type: str,
        overwrite: bool = False,
    ) -> str: ...

    def upload_text(
        self,
        *,
        container_name: str,
        blob_name: str,
        text: str,
        overwrite: bool = False,
    ) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class KnowledgeIndex(Protocol):
    @property
    def knowledge_index_name(self) -> str: ...

    def run_source_indexer(self, *, timeout_seconds: int = 300) -> None: ...

    def collect_paper_chunks(
        self, *, paper_blob_name: str, paper_blob_uri: str
    ) -> list[KnowledgeChunk]: ...

    def ensure_knowledge_index(self) -> None: ...

    def upload_chunks(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int: ...


class IngestionService:
    """Acquire, fingerprint, store, chunk, embed, and index trusted metadata only."""

    def __init__(
        self,
        *,
        repository_source: RepositorySource,
        artifact_store: ArtifactStore,
        embedder: Embedder,
        knowledge_index: KnowledgeIndex,
        paper_container_name: str,
        artifact_container_name: str,
    ) -> None:
        self._repository_source = repository_source
        self._artifact_store = artifact_store
        self._embedder = embedder
        self._knowledge_index = knowledge_index
        self._paper_container_name = paper_container_name
        self._artifact_container_name = artifact_container_name

    def run(
        self,
        *,
        run_id: str,
        paper_path: Path,
        repository_url: str,
        commit_sha: str,
        run_mode: str = "smoke",
        prepare_only: bool = False,
        refresh_paper_index: bool = True,
    ) -> IngestionManifest:
        """Run ingestion without importing or executing repository content."""

        if run_mode not in {"smoke", "reproduce", "production"}:
            raise ValueError("run_mode must be smoke, reproduce, or production.")
        if not paper_path.is_file():
            raise ValueError(f"Paper file does not exist: {paper_path}")
        if not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("run_id must be a non-empty single path segment.")

        snapshot = self._repository_source.acquire(repository_url, commit_sha)
        repository_chunks = chunk_repository(
            snapshot,
            max_chunks=250 if run_mode == "smoke" else None,
        )
        paper_bytes = paper_path.read_bytes()
        if not paper_bytes.startswith(b"%PDF"):
            raise ValueError("The selected paper is not a valid PDF file.")
        paper_sha256 = hashlib.sha256(paper_bytes).hexdigest()

        paper_blob_uri = self._artifact_store.upload_local_file(
            container_name=self._paper_container_name,
            blob_name=paper_path.name,
            local_path=paper_path,
            content_type="application/pdf",
        )
        run_prefix = f"runs/{run_id}"
        archive_blob_uri = self._artifact_store.upload_bytes(
            container_name=self._artifact_container_name,
            blob_name=f"{run_prefix}/source/{snapshot.archive_name}",
            data=snapshot.archive_bytes,
            content_type="application/gzip",
        )
        indexed_file_paths = {
            chunk.file_path for chunk in repository_chunks if chunk.file_path is not None
        }
        indexed_files = [
            source_file for source_file in snapshot.files if source_file.path in indexed_file_paths
        ]
        for source_file in indexed_files:
            self._artifact_store.upload_text(
                container_name=self._artifact_container_name,
                blob_name=f"{run_prefix}/source/repository/{source_file.path}",
                text=source_file.content,
            )

        manifest = IngestionManifest(
            run_id=run_id,
            run_mode=run_mode,
            paper_name=paper_path.name,
            paper_sha256=paper_sha256,
            paper_blob_uri=paper_blob_uri,
            repository_url=snapshot.repository_url,
            commit_sha=snapshot.commit_sha,
            archive_sha256=snapshot.archive_sha256,
            archive_blob_uri=archive_blob_uri,
            repository_file_count=len(indexed_files),
            repository_chunk_count=len(repository_chunks),
            skipped_files=snapshot.skipped_files,
        )
        manifest_name = f"{run_prefix}/manifest.json"
        self._artifact_store.upload_text(
            container_name=self._artifact_container_name,
            blob_name=manifest_name,
            text=manifest.to_json(),
            overwrite=True,
        )
        if prepare_only:
            return manifest

        if refresh_paper_index:
            self._knowledge_index.run_source_indexer()
        paper_chunks = self._knowledge_index.collect_paper_chunks(
            paper_blob_name=paper_path.name,
            paper_blob_uri=paper_blob_uri,
        )
        all_chunks = [*paper_chunks, *repository_chunks]
        self._knowledge_index.ensure_knowledge_index()
        embeddings = self._embedder.embed([chunk.content for chunk in all_chunks])
        indexed_count = self._knowledge_index.upload_chunks(all_chunks, embeddings)
        completed = replace(
            manifest,
            knowledge_index_name=self._knowledge_index.knowledge_index_name,
            paper_chunk_count=len(paper_chunks),
            indexed_chunk_count=indexed_count,
        )
        self._artifact_store.upload_text(
            container_name=self._artifact_container_name,
            blob_name=manifest_name,
            text=completed.to_json(),
            overwrite=True,
        )
        return completed
