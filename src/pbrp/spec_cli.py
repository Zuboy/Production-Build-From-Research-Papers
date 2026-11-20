"""Command-line composition root for evidence-backed ProductionSpec extraction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from azure.identity import DefaultAzureCredential

from pbrp.config import ConfigurationError, ProductionSpecSettings
from pbrp.providers.llm import FoundryProductionSpecProvider
from pbrp.providers.storage import AzureBlobArtifactStore
from pbrp.providers.vectorstore import AzureSearchVectorStore
from pbrp.spec.models import ProductionSpec, SourceFingerprint
from pbrp.spec.service import ProductionSpecService

DEFAULT_RUN_ID = "lora-smoke-20261118-c4593f06"
DEFAULT_PAPER_SHA256 = "e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a"
DEFAULT_REPOSITORY_URL = "https://github.com/microsoft/LoRA"
DEFAULT_COMMIT_SHA = "c4593f060e6a368d7bb5af5273b8e42810cdef90"
PROMPT_PATH = Path(__file__).parent / "agents" / "prompts" / "production_spec_v1.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbrp-spec",
        description="Extract an evidence-cited ProductionSpec from the unified Azure RAG index.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--paper-sha256", default=DEFAULT_PAPER_SHA256)
    parser.add_argument("--repository-url", default=DEFAULT_REPOSITORY_URL)
    parser.add_argument("--commit", default=DEFAULT_COMMIT_SHA)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print counts and the artifact URI without printing the full JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    credential: DefaultAzureCredential | None = None
    retriever: AzureSearchVectorStore | None = None
    model: FoundryProductionSpecProvider | None = None
    artifacts: AzureBlobArtifactStore | None = None
    try:
        settings = ProductionSpecSettings.from_env()
        instructions = PROMPT_PATH.read_text(encoding="utf-8")
        credential = DefaultAzureCredential()
        retriever = AzureSearchVectorStore(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=credential,
            semantic_configuration_name=settings.semantic_configuration_name,
            vector_candidate_count=settings.vector_candidate_count,
            include_evidence_metadata=True,
        )
        model = FoundryProductionSpecProvider(
            project_endpoint=settings.azure_ai_project_endpoint,
            deployment_name=settings.azure_ai_model_deployment_name,
            credential=credential,
            instructions=instructions,
        )
        artifacts = AzureBlobArtifactStore(
            endpoint=settings.azure_storage_endpoint,
            credential=credential,
        )
        service = ProductionSpecService(
            retriever=retriever,
            model=model,
            retrieval_limit_per_query=settings.retrieval_limit_per_query,
        )
        spec = service.extract(
            run_id=args.run_id,
            sources=SourceFingerprint(
                paper_sha256=args.paper_sha256,
                repository_url=args.repository_url,
                commit_sha=args.commit,
            ),
        )
        blob_uri = _store_spec(
            spec=spec,
            artifacts=artifacts,
            container_name=settings.artifact_container_name,
        )
        _print_result(spec, blob_uri, include_json=not args.summary_only)
        return 0
    except (ConfigurationError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # Keep the CLI boundary user-readable.
        print(f"ProductionSpec extraction failed: {error}", file=sys.stderr)
        return 1
    finally:
        if artifacts is not None:
            artifacts.close()
        if model is not None:
            model.close()
        if retriever is not None:
            retriever.close()
        if credential is not None:
            credential.close()


def _store_spec(
    *,
    spec: ProductionSpec,
    artifacts: AzureBlobArtifactStore,
    container_name: str,
) -> str:
    return artifacts.upload_bytes(
        container_name=container_name,
        blob_name=f"runs/{spec.run_id}/production-spec-v1.json",
        data=spec.to_json().encode("utf-8"),
        content_type="application/json",
        overwrite=True,
    )


def _print_result(spec: ProductionSpec, blob_uri: str, *, include_json: bool) -> None:
    populated = 12 - len(spec.gaps)
    print(f"ProductionSpec created for run: {spec.run_id}")
    print(f"Populated requirement groups: {populated}/12")
    print(f"Explicit gaps: {len(spec.gaps)}")
    print(f"Artifact: {blob_uri}")
    if include_json:
        print("\nPRODUCTION SPEC\n")
        print(spec.to_json())


if __name__ == "__main__":
    raise SystemExit(main())
