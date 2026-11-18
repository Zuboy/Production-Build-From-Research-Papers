"""Command-line composition root for the Azure RAG path."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from azure.identity import DefaultAzureCredential

from pbrp.config import ConfigurationError, Settings
from pbrp.providers.llm import FoundryLLMProvider
from pbrp.providers.vectorstore import AzureSearchVectorStore
from pbrp.rag.models import RAGAnswer
from pbrp.rag.service import RAGService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbrp",
        description="Ask an evidence-grounded question about indexed research papers.",
    )
    parser.add_argument("question", nargs="+", help="Question to answer from the index")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question = " ".join(args.question).strip()

    credential: DefaultAzureCredential | None = None
    vector_store: AzureSearchVectorStore | None = None
    language_model: FoundryLLMProvider | None = None
    try:
        settings = Settings.from_env()
        credential = DefaultAzureCredential()
        vector_store = AzureSearchVectorStore(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=credential,
            semantic_configuration_name=settings.semantic_configuration_name,
            vector_field_name=settings.vector_field_name,
            vector_candidate_count=settings.vector_candidate_count,
        )
        language_model = FoundryLLMProvider(
            project_endpoint=settings.azure_ai_project_endpoint,
            deployment_name=settings.azure_ai_model_deployment_name,
            credential=credential,
        )
        service = RAGService(
            vector_store=vector_store,
            language_model=language_model,
            retrieval_limit=settings.retrieval_limit,
        )
        _print_answer(service.answer(question))
        return 0
    except (ConfigurationError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # Keep the CLI boundary user-readable.
        print(f"RAG request failed: {error}", file=sys.stderr)
        return 1
    finally:
        if language_model is not None:
            language_model.close()
        if vector_store is not None:
            vector_store.close()
        if credential is not None:
            credential.close()


def _print_answer(result: RAGAnswer) -> None:
    print("\nANSWER\n")
    print(result.answer)

    if result.sources:
        print("\nCITED EVIDENCE\n")
        for source in result.sources:
            score = _format_score(source.search_score)
            reranker = _format_score(source.reranker_score)
            print(
                f"[{source.source_id}] {source.locator} "
                f"search_score={score} reranker_score={reranker}"
            )

    if result.gaps:
        print("\nGAPS\n")
        for gap in result.gaps:
            print(f"- {gap}")


def _format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
