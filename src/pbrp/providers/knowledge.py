"""Azure AI Search provider for paper indexing and unified knowledge storage."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

from azure.core.credentials import TokenCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from pbrp.ingest.models import KnowledgeChunk


class AzureKnowledgeIndexProvider:
    """Manage the Azure AI Search boundary used by source ingestion."""

    def __init__(
        self,
        *,
        endpoint: str,
        source_index_name: str,
        source_indexer_name: str,
        knowledge_index_name: str,
        semantic_configuration_name: str,
        embedding_resource_uri: str,
        embedding_deployment_name: str,
        vector_dimensions: int,
        credential: TokenCredential,
    ) -> None:
        self._endpoint = endpoint
        self._source_index_name = source_index_name
        self._source_indexer_name = source_indexer_name
        self._knowledge_index_name = knowledge_index_name
        self._semantic_configuration_name = semantic_configuration_name
        self._embedding_resource_uri = embedding_resource_uri
        self._embedding_deployment_name = embedding_deployment_name
        self._vector_dimensions = vector_dimensions
        self._credential = credential
        self._index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=credential,
            audience="https://search.azure.com",
        )
        self._indexer_client = SearchIndexerClient(
            endpoint=endpoint,
            credential=credential,
            audience="https://search.azure.com",
        )

    @property
    def knowledge_index_name(self) -> str:
        return self._knowledge_index_name

    def run_source_indexer(self, *, timeout_seconds: int = 300) -> None:
        """Start the existing integrated-vectorization indexer and await completion."""

        previous = self._indexer_client.get_indexer_status(self._source_indexer_name)
        previous_start = (
            previous.last_result.start_time if previous.last_result is not None else None
        )
        self._indexer_client.run_indexer(self._source_indexer_name)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status = self._indexer_client.get_indexer_status(self._source_indexer_name)
            last_result = status.last_result
            if last_result is not None and last_result.start_time != previous_start:
                normalized_status = str(last_result.status).lower()
                if "success" in normalized_status:
                    return
                if "error" in normalized_status or "fail" in normalized_status:
                    details = "; ".join(
                        str(error.error_message) for error in (last_result.errors or [])
                    )
                    raise RuntimeError(f"Azure AI Search indexer failed: {details}")
            time.sleep(2)
        raise TimeoutError("Azure AI Search indexer did not finish before the timeout.")

    def collect_paper_chunks(
        self,
        *,
        paper_blob_name: str,
        paper_blob_uri: str,
    ) -> list[KnowledgeChunk]:
        """Read the newly indexed paper chunks and attach stable source metadata."""

        client = SearchClient(
            endpoint=self._endpoint,
            index_name=self._source_index_name,
            credential=self._credential,
            audience="https://search.azure.com",
        )
        try:
            results = client.search(
                search_text="*",
                select=["chunk_id", "parent_id", "title", "chunk"],
                top=1000,
            )
            candidates = [dict(result) for result in results]
        finally:
            client.close()

        exact_name = paper_blob_name.casefold()
        stem = paper_blob_name.rsplit(".", 1)[0].casefold()
        matching = [
            item
            for item in candidates
            if exact_name in str(item.get("title") or "").casefold()
            or stem in str(item.get("title") or "").casefold()
        ]
        if not matching:
            raise RuntimeError(
                f"No source chunks for {paper_blob_name!r} were found after indexing."
            )

        document_id = f"paper:{hashlib.sha256(paper_blob_uri.encode()).hexdigest()}"
        chunks: list[KnowledgeChunk] = []
        for item in matching:
            content = str(item.get("chunk") or "").strip()
            source_chunk_id = str(item.get("chunk_id") or "").strip()
            if not content or not source_chunk_id:
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"paper:{paper_blob_uri}:{source_chunk_id}:{content_hash}".encode()
            ).hexdigest()
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    parent_id=document_id,
                    source_type="paper",
                    document_id=document_id,
                    title=str(item.get("title") or paper_blob_name),
                    content=content,
                    source_uri=f"{paper_blob_uri}#chunk={source_chunk_id}",
                    content_hash=content_hash,
                )
            )
        if not chunks:
            raise RuntimeError("The matching paper entries contained no readable chunks.")
        return chunks

    def ensure_knowledge_index(self) -> None:
        """Create or update the unified paper-and-repository index."""

        vectorizer_name = "foundry-text-embedding-3-small"
        algorithm_name = "pbrp-hnsw"
        profile_name = "pbrp-vector-profile"
        index = SearchIndex(
            name=self._knowledge_index_name,
            fields=_knowledge_fields(self._vector_dimensions, profile_name),
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name=algorithm_name,
                        parameters=HnswParameters(
                            metric="cosine",
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                        ),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name=profile_name,
                        algorithm_configuration_name=algorithm_name,
                        vectorizer_name=vectorizer_name,
                    )
                ],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        vectorizer_name=vectorizer_name,
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=self._embedding_resource_uri,
                            deployment_name=self._embedding_deployment_name,
                            model_name="text-embedding-3-small",
                        ),
                    )
                ],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=self._semantic_configuration_name,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[SemanticField(field_name="chunk")],
                            keywords_fields=[
                                SemanticField(field_name="file_path"),
                                SemanticField(field_name="section"),
                            ],
                        ),
                    )
                ]
            ),
        )
        self._index_client.create_or_update_index(index)

    def upload_chunks(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Upload embedded chunks in bounded batches and verify every item."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each knowledge chunk must have exactly one embedding.")
        client = SearchClient(
            endpoint=self._endpoint,
            index_name=self._knowledge_index_name,
            credential=self._credential,
            audience="https://search.azure.com",
        )
        uploaded = 0
        try:
            for start in range(0, len(chunks), 500):
                batch_chunks = chunks[start : start + 500]
                batch_embeddings = embeddings[start : start + 500]
                documents = [
                    _search_document(chunk, embedding)
                    for chunk, embedding in zip(batch_chunks, batch_embeddings, strict=True)
                ]
                results = client.upload_documents(documents=documents)
                failures = [result for result in results if not result.succeeded]
                if failures:
                    messages = "; ".join(f"{item.key}: {item.error_message}" for item in failures)
                    raise RuntimeError(f"Azure AI Search rejected documents: {messages}")
                uploaded += len(results)
        finally:
            client.close()
        return uploaded

    def close(self) -> None:
        self._indexer_client.close()
        self._index_client.close()


def _knowledge_fields(dimensions: int, profile_name: str) -> list[SearchField]:
    return [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="chunk", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="source_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_uri", type=SearchFieldDataType.String),
        SimpleField(name="repository_url", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="commit_sha", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="file_path", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="line_start", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="line_end", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="section", type=SearchFieldDataType.String),
        SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=dimensions,
            vector_search_profile_name=profile_name,
        ),
    ]


def _search_document(
    chunk: KnowledgeChunk, embedding: Sequence[float]
) -> dict[str, str | int | list[float] | None]:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "chunk": chunk.content,
        "title": chunk.title,
        "source_type": chunk.source_type,
        "document_id": chunk.document_id,
        "source_uri": chunk.source_uri,
        "repository_url": chunk.repository_url,
        "commit_sha": chunk.commit_sha,
        "file_path": chunk.file_path,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "content_hash": chunk.content_hash,
        "text_vector": list(embedding),
    }
