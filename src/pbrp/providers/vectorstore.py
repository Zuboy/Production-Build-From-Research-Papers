"""Azure AI Search provider for hybrid evidence retrieval."""

from __future__ import annotations

import hashlib

from azure.core.credentials import TokenCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from pbrp.rag.models import RetrievedChunk
from pbrp.spec.models import RunSource


class AzureSearchVectorStore:
    """Retrieve evidence with keyword, vector, and semantic search."""

    def __init__(
        self,
        *,
        endpoint: str,
        index_name: str,
        credential: TokenCredential,
        semantic_configuration_name: str,
        vector_field_name: str = "text_vector",
        vector_candidate_count: int = 50,
    ) -> None:
        self._semantic_configuration_name = semantic_configuration_name
        self._vector_field_name = vector_field_name
        self._vector_candidate_count = vector_candidate_count
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
            audience="https://search.azure.com",
        )

    def retrieve(self, question: str, limit: int = 5) -> list[RetrievedChunk]:
        """Return the highest-ranked human-readable chunks for a question."""

        return self._retrieve(question, limit)

    def retrieve_source(
        self, query: str, source: RunSource, source_type: str
    ) -> list[RetrievedChunk]:
        """Restrict retrieval to the indexed paper or exact repository commit in a run."""

        if source_type == "paper":
            paper_id = hashlib.sha256(source.paper_blob_uri.encode()).hexdigest()
            filters = f"source_type eq 'paper' and document_id eq 'paper:{paper_id}'"
        elif source_type == "repository":
            repository = source.repository_url.replace("'", "''")
            filters = (
                f"source_type eq 'repository' and commit_sha eq '{source.commit_sha}' "
                f"and repository_url eq '{repository}'"
            )
        else:
            raise ValueError(f"Unsupported evidence source type: {source_type}")
        return self._retrieve(query, 6, filters)

    def _retrieve(
        self, question: str, limit: int, filters: str | None = None
    ) -> list[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("The retrieval question cannot be empty.")
        if limit < 1:
            raise ValueError("The retrieval limit must be greater than zero.")

        vector_query = VectorizableTextQuery(
            text=normalized_question,
            fields=self._vector_field_name,
            k_nearest_neighbors=max(self._vector_candidate_count, limit),
        )
        fields = ["chunk_id", "parent_id", "title", "chunk"]
        if filters is not None:
            fields.extend(("source_type", "source_uri"))
        search_results = self._client.search(
            search_text=normalized_question,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name=self._semantic_configuration_name,
            select=fields,
            filter=filters,
            top=limit,
        )

        chunks: list[RetrievedChunk] = []
        for position, result in enumerate(search_results, start=1):
            content = str(result.get("chunk") or "").strip()
            chunk_id = str(result.get("chunk_id") or "").strip()
            if not content or not chunk_id:
                continue
            chunks.append(
                RetrievedChunk(
                    source_id=f"S{position}",
                    chunk_id=chunk_id,
                    parent_id=_optional_text(result.get("parent_id")),
                    title=_optional_text(result.get("title")),
                    content=content,
                    search_score=_optional_float(result.get("@search.score")),
                    reranker_score=_optional_float(result.get("@search.reranker_score")),
                    source_uri=_optional_text(result.get("source_uri")),
                    source_type=_optional_text(result.get("source_type")),
                )
            )
        return chunks

    def close(self) -> None:
        """Release the underlying Search transport."""

        self._client.close()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return None
    return None
