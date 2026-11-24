"""Command-line composition root for the golden paper/repository ingestion path."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from azure.identity import DefaultAzureCredential

from pbrp.config import ConfigurationError, IngestionSettings
from pbrp.ingest.models import IngestionManifest
from pbrp.ingest.service import IngestionService
from pbrp.providers.knowledge import AzureKnowledgeIndexProvider
from pbrp.providers.llm import AzureOpenAIEmbeddingProvider
from pbrp.providers.repository import GitHubArchiveProvider
from pbrp.providers.storage import AzureBlobArtifactStore

DEFAULT_REPOSITORY_URL = "https://github.com/microsoft/LoRA"
DEFAULT_COMMIT_SHA = "c4593f060e6a368d7bb5af5273b8e42810cdef90"
DEFAULT_PAPER = Path("ResearchPapers/LoRA_Low_Rank_Adaptation.pdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbrp-ingest-golden",
        description=(
            "Safely store and index the LoRA paper and its pinned Microsoft repository. "
            "Repository code is parsed only as static text and is never executed."
        ),
    )
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--repository-url", default=DEFAULT_REPOSITORY_URL)
    parser.add_argument("--commit", default=DEFAULT_COMMIT_SHA)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--run-mode",
        choices=("smoke", "reproduce", "production"),
        default="smoke",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Store immutable sources and a manifest without changing the Search index.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or _default_run_id(args.commit)

    credential: DefaultAzureCredential | None = None
    artifact_store: AzureBlobArtifactStore | None = None
    embedder: AzureOpenAIEmbeddingProvider | None = None
    knowledge_index: AzureKnowledgeIndexProvider | None = None
    try:
        settings = IngestionSettings.from_env()
        credential = DefaultAzureCredential()
        artifact_store = AzureBlobArtifactStore(
            endpoint=settings.azure_storage_endpoint,
            credential=credential,
        )
        embedder = AzureOpenAIEmbeddingProvider(
            resource_endpoint=settings.azure_openai_embedding_resource_uri,
            deployment_name=settings.azure_ai_embedding_deployment_name,
            credential=credential,
            dimensions=settings.vector_dimensions,
        )
        knowledge_index = AzureKnowledgeIndexProvider(
            endpoint=settings.azure_search_endpoint,
            source_index_name=settings.source_index_name,
            source_indexer_name=settings.source_indexer_name,
            knowledge_index_name=settings.knowledge_index_name,
            semantic_configuration_name=settings.knowledge_semantic_configuration_name,
            embedding_resource_uri=settings.azure_openai_embedding_resource_uri,
            embedding_deployment_name=settings.azure_ai_embedding_deployment_name,
            vector_dimensions=settings.vector_dimensions,
            credential=credential,
        )
        service = IngestionService(
            repository_source=GitHubArchiveProvider(),
            artifact_store=artifact_store,
            embedder=embedder,
            knowledge_index=knowledge_index,
            paper_container_name=settings.paper_container_name,
            artifact_container_name=settings.artifact_container_name,
        )
        result = service.run(
            run_id=run_id,
            paper_path=args.paper,
            repository_url=args.repository_url,
            commit_sha=args.commit,
            run_mode=args.run_mode,
            prepare_only=args.prepare_only,
        )
        _print_result(result, prepare_only=args.prepare_only)
        return 0
    except (ConfigurationError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # Keep the CLI boundary user-readable.
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1
    finally:
        if knowledge_index is not None:
            knowledge_index.close()
        if embedder is not None:
            embedder.close()
        if artifact_store is not None:
            artifact_store.close()
        if credential is not None:
            credential.close()


def _default_run_id(commit_sha: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"lora-{timestamp}-{commit_sha[:8].lower()}"


def _print_result(result: IngestionManifest, *, prepare_only: bool) -> None:
    status = "prepared" if prepare_only else "indexed"
    print(f"Golden source run {status}: {result.run_id}")
    print(f"Paper SHA-256: {result.paper_sha256}")
    print(f"Repository commit: {result.commit_sha}")
    print(f"Safe repository files: {result.repository_file_count}")
    print(f"Repository chunks: {result.repository_chunk_count}")
    print(f"Paper chunks: {result.paper_chunk_count}")
    print(f"Indexed chunks: {result.indexed_chunk_count}")
    print(f"Manifest prefix: outputrp/runs/{result.run_id}/")


if __name__ == "__main__":
    raise SystemExit(main())
