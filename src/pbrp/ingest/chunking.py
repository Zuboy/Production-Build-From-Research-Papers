"""Deterministic static chunking for validated repository text."""

from __future__ import annotations

import hashlib

from pbrp.ingest.models import KnowledgeChunk, RepositorySnapshot


def chunk_repository(
    snapshot: RepositorySnapshot,
    *,
    max_characters: int = 3_500,
    overlap_lines: int = 5,
    max_chunks: int | None = None,
) -> list[KnowledgeChunk]:
    """Chunk source files by line while retaining stable file and line locators."""

    if max_characters < 500:
        raise ValueError("max_characters must be at least 500.")
    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative.")
    if max_chunks is not None and max_chunks < 1:
        raise ValueError("max_chunks must be greater than zero when provided.")

    chunks: list[KnowledgeChunk] = []
    document_id = f"github:{snapshot.commit_sha}"
    source_files = sorted(snapshot.files, key=lambda item: _file_priority(item.path))
    for source_file in source_files:
        lines = source_file.content.splitlines(keepends=True)
        if not lines:
            continue
        start = 0
        while start < len(lines):
            end = _chunk_end(lines, start, max_characters)
            content = "".join(lines[start:end]).strip()
            if content:
                line_start = start + 1
                line_end = end
                locator = f"{source_file.path}:L{line_start}-L{line_end}"
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                chunk_id = hashlib.sha256(
                    f"{snapshot.commit_sha}:{locator}:{content_hash}".encode()
                ).hexdigest()
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        parent_id=document_id,
                        source_type="repository",
                        document_id=document_id,
                        title=f"microsoft/LoRA — {source_file.path}",
                        content=content,
                        source_uri=(
                            f"{snapshot.repository_url}/blob/{snapshot.commit_sha}/"
                            f"{source_file.path}#L{line_start}-L{line_end}"
                        ),
                        content_hash=content_hash,
                        repository_url=snapshot.repository_url,
                        commit_sha=snapshot.commit_sha,
                        file_path=source_file.path,
                        line_start=line_start,
                        line_end=line_end,
                    )
                )
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks
            if end >= len(lines):
                break
            start = max(start + 1, end - overlap_lines)
    return chunks


def _chunk_end(lines: list[str], start: int, max_characters: int) -> int:
    size = 0
    end = start
    while end < len(lines):
        next_size = len(lines[end])
        if end > start and size + next_size > max_characters:
            break
        size += next_size
        end += 1
        if size >= max_characters:
            break
    return max(start + 1, end)


def _file_priority(path: str) -> tuple[int, str]:
    """Prefer project-level evidence and the core LoRA package in smoke mode."""

    normalized = path.casefold()
    depth = normalized.count("/")
    name = normalized.rsplit("/", 1)[-1]
    if depth == 0:
        return 0, normalized
    if normalized.startswith("loralib/"):
        return 1, normalized
    if name.startswith(("readme", "requirements")) or name in {
        "dockerfile",
        "makefile",
        "pipfile",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
    }:
        return 2, normalized
    return 3, normalized
