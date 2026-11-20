"""Safe, static acquisition of commit-pinned GitHub repository archives."""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import urllib.request
from pathlib import PurePosixPath
from typing import cast
from urllib.parse import urlparse

from pbrp.ingest.models import RepositoryFile, RepositorySnapshot

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_ALLOWED_NAMES = {
    "dockerfile",
    "makefile",
    "pipfile",
    "requirements",
}
_IGNORED_PARTS = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


class GitHubArchiveProvider:
    """Download and inspect a GitHub tarball without extracting or executing it."""

    def __init__(
        self,
        *,
        max_archive_bytes: int = 25 * 1024 * 1024,
        max_uncompressed_bytes: int = 100 * 1024 * 1024,
        max_file_bytes: int = 1 * 1024 * 1024,
        max_members: int = 10_000,
    ) -> None:
        self._max_archive_bytes = max_archive_bytes
        self._max_uncompressed_bytes = max_uncompressed_bytes
        self._max_file_bytes = max_file_bytes
        self._max_members = max_members

    def acquire(self, repository_url: str, commit_sha: str) -> RepositorySnapshot:
        """Acquire the exact commit and return only validated UTF-8 allowlisted files."""

        owner, repository, canonical_url = _parse_github_url(repository_url)
        normalized_commit = commit_sha.strip().lower()
        if not _COMMIT_PATTERN.fullmatch(normalized_commit):
            raise ValueError("commit_sha must be a full 40-character lowercase SHA-1.")

        archive_url = f"https://codeload.github.com/{owner}/{repository}/tar.gz/{normalized_commit}"
        archive_bytes = self._download(archive_url)
        files, skipped = self._inspect_archive(archive_bytes)
        if not files:
            raise ValueError("The repository archive contained no supported UTF-8 source files.")
        return RepositorySnapshot(
            repository_url=canonical_url,
            commit_sha=normalized_commit,
            archive_name=f"{repository}-{normalized_commit}.tar.gz",
            archive_bytes=archive_bytes,
            archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
            files=tuple(files),
            skipped_files=tuple(skipped),
        )

    def _download(self, archive_url: str) -> bytes:
        request = urllib.request.Request(
            archive_url,
            headers={"User-Agent": "pbrp-safe-ingestion/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > self._max_archive_bytes:
                raise ValueError("Repository archive exceeds the compressed size limit.")
            payload = cast(bytes, response.read(self._max_archive_bytes + 1))
        if len(payload) > self._max_archive_bytes:
            raise ValueError("Repository archive exceeds the compressed size limit.")
        return payload

    def _inspect_archive(self, archive_bytes: bytes) -> tuple[list[RepositoryFile], list[str]]:
        files: list[RepositoryFile] = []
        skipped: list[str] = []
        uncompressed_bytes = 0
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > self._max_members:
                raise ValueError("Repository archive exceeds the member count limit.")
            root: str | None = None
            for member in members:
                path = _validated_member_path(member.name)
                if root is None and path.parts:
                    root = path.parts[0]
                if member.isdir():
                    continue
                if member.issym() or member.islnk():
                    skipped.append(f"{member.name} (link skipped)")
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsafe archive member type: {member.name}")
                if member.size < 0 or member.size > self._max_file_bytes:
                    skipped.append(member.name)
                    continue
                uncompressed_bytes += member.size
                if uncompressed_bytes > self._max_uncompressed_bytes:
                    raise ValueError("Repository archive exceeds the uncompressed size limit.")
                relative_path = _relative_archive_path(path, root)
                if not _is_supported_path(relative_path):
                    skipped.append(relative_path.as_posix())
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not read archive member: {member.name}")
                data = extracted.read(self._max_file_bytes + 1)
                if len(data) > self._max_file_bytes or b"\x00" in data:
                    skipped.append(relative_path.as_posix())
                    continue
                try:
                    content = data.decode("utf-8")
                except UnicodeDecodeError:
                    skipped.append(relative_path.as_posix())
                    continue
                if any(len(line) > 50_000 for line in content.splitlines()):
                    skipped.append(f"{relative_path.as_posix()} (machine-generated line skipped)")
                    continue
                files.append(
                    RepositoryFile(
                        path=relative_path.as_posix(),
                        content=content,
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )
        return sorted(files, key=lambda item: item.path), sorted(set(skipped))


def _parse_github_url(repository_url: str) -> tuple[str, str, str]:
    parsed = urlparse(repository_url.strip())
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError("repository_url must be an HTTPS github.com repository URL.")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("repository_url must identify exactly one GitHub owner and repository.")
    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository or not all(_safe_slug(value) for value in (owner, repository)):
        raise ValueError("repository_url contains unsupported owner or repository characters.")
    return owner, repository, f"https://github.com/{owner}/{repository}"


def _safe_slug(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))


def _validated_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def _relative_archive_path(path: PurePosixPath, root: str | None) -> PurePosixPath:
    if root is None or not path.parts or path.parts[0] != root or len(path.parts) < 2:
        raise ValueError(f"Archive member is outside the expected root: {path}")
    return PurePosixPath(*path.parts[1:])


def _is_supported_path(path: PurePosixPath) -> bool:
    if any(part.lower() in _IGNORED_PARTS for part in path.parts):
        return False
    name = path.name.lower()
    return path.suffix.lower() in _ALLOWED_SUFFIXES or name in _ALLOWED_NAMES
