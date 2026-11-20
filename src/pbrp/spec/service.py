"""Provider-independent extraction and evidence validation for ProductionSpec."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from pbrp.rag.models import RetrievedChunk
from pbrp.spec.models import (
    DraftEvidence,
    DraftListRequirement,
    DraftTextRequirement,
    Evidence,
    ListRequirement,
    ProductionSpec,
    ProductionSpecDraft,
    SourceFingerprint,
    TextRequirement,
)

SPEC_QUERIES = (
    "What task, model architecture, programming language, and ML framework does this project use?",
    "What dependencies, package versions, Python version, and installation "
    "commands are documented?",
    "What executable entry points, training commands, evaluation commands, "
    "or smoke commands exist?",
    "What datasets, input formats, preprocessing steps, and reported metrics are required?",
    "What hardware requirements and serving or inference interface are evidenced?",
)


class EvidenceRetriever(Protocol):
    def retrieve(self, question: str, limit: int = 5) -> list[RetrievedChunk]: ...


class SpecModel(Protocol):
    def extract(self, chunks: list[RetrievedChunk]) -> ProductionSpecDraft: ...


class ProductionSpecService:
    """Retrieve diverse evidence and accept only code-validated model facts."""

    def __init__(
        self,
        *,
        retriever: EvidenceRetriever,
        model: SpecModel,
        retrieval_limit_per_query: int = 6,
    ) -> None:
        if retrieval_limit_per_query < 1:
            raise ValueError("retrieval_limit_per_query must be positive.")
        self._retriever = retriever
        self._model = model
        self._retrieval_limit = retrieval_limit_per_query

    def extract(
        self,
        *,
        run_id: str,
        sources: SourceFingerprint,
    ) -> ProductionSpec:
        chunks = self._retrieve_evidence()
        if not chunks:
            raise RuntimeError("No evidence was retrieved for ProductionSpec extraction.")
        draft = self._model.extract(chunks)
        chunks_by_id: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            chunks_by_id[chunk.source_id] = chunk
            chunks_by_id[chunk.chunk_id] = chunk

        fields: dict[str, TextRequirement | ListRequirement] = {
            "task": _validate_text("task", draft.task, chunks_by_id),
            "language": _validate_text("language", draft.language, chunks_by_id),
            "framework": _validate_text("framework", draft.framework, chunks_by_id),
            "python_version": _validate_text("python_version", draft.python_version, chunks_by_id),
            "dependencies": _validate_list("dependencies", draft.dependencies, chunks_by_id),
            "install_commands": _validate_list(
                "install_commands", draft.install_commands, chunks_by_id
            ),
            "smoke_commands": _validate_list("smoke_commands", draft.smoke_commands, chunks_by_id),
            "entry_points": _validate_list("entry_points", draft.entry_points, chunks_by_id),
            "datasets": _validate_list("datasets", draft.datasets, chunks_by_id),
            "metrics": _validate_list("metrics", draft.metrics, chunks_by_id),
            "hardware": _validate_list("hardware", draft.hardware, chunks_by_id),
            "serving_interface": _validate_text(
                "serving_interface", draft.serving_interface, chunks_by_id
            ),
        }
        gaps = tuple(
            f"{name}: {requirement.gap}"
            for name, requirement in fields.items()
            if requirement.gap is not None
        )
        return ProductionSpec(
            run_id=run_id,
            sources=sources,
            task=_as_text(fields["task"]),
            language=_as_text(fields["language"]),
            framework=_as_text(fields["framework"]),
            python_version=_as_text(fields["python_version"]),
            dependencies=_as_list(fields["dependencies"]),
            install_commands=_as_list(fields["install_commands"]),
            smoke_commands=_as_list(fields["smoke_commands"]),
            entry_points=_as_list(fields["entry_points"]),
            datasets=_as_list(fields["datasets"]),
            metrics=_as_list(fields["metrics"]),
            hardware=_as_list(fields["hardware"]),
            serving_interface=_as_text(fields["serving_interface"]),
            gaps=gaps,
        )

    def _retrieve_evidence(self) -> list[RetrievedChunk]:
        deduplicated: dict[str, RetrievedChunk] = {}
        for query in SPEC_QUERIES:
            for chunk in self._retriever.retrieve(query, limit=self._retrieval_limit):
                deduplicated.setdefault(chunk.chunk_id, chunk)
        return [
            replace(chunk, source_id=f"S{position}")
            for position, chunk in enumerate(deduplicated.values(), start=1)
        ]


def _validate_text(
    name: str,
    draft: DraftTextRequirement,
    chunks: dict[str, RetrievedChunk],
) -> TextRequirement:
    value = draft.value.strip() if draft.value is not None else None
    if not value:
        return _missing_text(draft.gap or f"No supported {name} was found.")
    evidence = _validated_evidence(draft.evidence, chunks)
    if not evidence or draft.confidence <= 0:
        return _missing_text(f"The proposed {name} had no verifiable evidence.")
    if not _text_is_supported(value, evidence):
        return _missing_text(f"The proposed {name} extended beyond its cited evidence.")
    if name == "serving_interface" and not _has_serving_context(evidence):
        return _missing_text("A serving package was named without evidence of an interface.")
    return TextRequirement(value=value, evidence=evidence, confidence=draft.confidence)


def _validate_list(
    name: str,
    draft: DraftListRequirement,
    chunks: dict[str, RetrievedChunk],
) -> ListRequirement:
    value = tuple(item.strip() for item in (draft.value or ()) if item.strip())
    if name.endswith("_commands"):
        value = tuple(item for item in value if _command_is_complete(item))
    if not value:
        return _missing_list(draft.gap or f"No supported {name} were found.")
    evidence = _validated_evidence(draft.evidence, chunks)
    if not evidence or draft.confidence <= 0:
        return _missing_list(f"The proposed {name} had no verifiable evidence.")
    supported_value, supporting_evidence = _supported_list_items(value, evidence)
    if not supported_value:
        return _missing_list(f"The proposed {name} extended beyond its cited evidence.")
    return ListRequirement(
        value=supported_value,
        evidence=supporting_evidence,
        confidence=draft.confidence,
    )


def _validated_evidence(
    proposed: tuple[DraftEvidence, ...],
    chunks: dict[str, RetrievedChunk],
) -> tuple[Evidence, ...]:
    validated: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for item in proposed:
        chunk = chunks.get(item.source_id)
        quote = item.quote.strip()
        key = (chunk.chunk_id if chunk is not None else item.source_id, quote)
        if chunk is None or not quote or key in seen:
            continue
        if _normalize(quote) not in _normalize(chunk.content):
            continue
        seen.add(key)
        validated.append(
            Evidence(
                source_id=chunk.source_id,
                source_type=chunk.source_type or "unknown",
                chunk_id=chunk.chunk_id,
                locator=chunk.locator,
                quote=quote,
                search_score=chunk.search_score,
                reranker_score=chunk.reranker_score,
            )
        )
    return tuple(validated)


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _text_is_supported(value: str, evidence: tuple[Evidence, ...]) -> bool:
    value_tokens = _meaningful_tokens(value)
    if not value_tokens:
        return False
    evidence_tokens = _meaningful_tokens(" ".join(item.quote for item in evidence))
    return len(value_tokens & evidence_tokens) / len(value_tokens) >= 0.75


def _supported_list_items(
    values: tuple[str, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[tuple[str, ...], tuple[Evidence, ...]]:
    supported_values: list[str] = []
    used_evidence: list[Evidence] = []
    for value in values:
        value_tokens = _meaningful_tokens(value)
        matches = [
            item
            for item in evidence
            if value_tokens
            and len(value_tokens & _meaningful_tokens(item.quote)) / len(value_tokens) >= 0.75
        ]
        if matches:
            supported_values.append(value)
            for item in matches:
                if item not in used_evidence:
                    used_evidence.append(item)
    return tuple(supported_values), tuple(used_evidence)


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "when",
        "while",
        "with",
    }
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in stopwords
    }


def _stem(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _command_is_complete(command: str) -> bool:
    normalized = command.rstrip()
    lowered = normalized.casefold()
    return (
        bool(normalized)
        and "\\" not in normalized
        and not lowered.startswith("run ")
        and " -it" not in lowered
        and " --interactive" not in lowered
        and not normalized.endswith(("&&", "|"))
    )


def _has_serving_context(evidence: tuple[Evidence, ...]) -> bool:
    serving_terms = {"api", "endpoint", "grpc", "http", "predict", "rest", "serve", "server"}
    for item in evidence:
        tokens = _meaningful_tokens(item.quote)
        if len(tokens) >= 3 and tokens & serving_terms:
            return True
    return False


def _missing_text(gap: str) -> TextRequirement:
    return TextRequirement(value=None, evidence=(), confidence=0.0, gap=gap.strip())


def _missing_list(gap: str) -> ListRequirement:
    return ListRequirement(value=None, evidence=(), confidence=0.0, gap=gap.strip())


def _as_text(value: TextRequirement | ListRequirement) -> TextRequirement:
    if not isinstance(value, TextRequirement):
        raise TypeError("Expected a text requirement.")
    return value


def _as_list(value: TextRequirement | ListRequirement) -> ListRequirement:
    if not isinstance(value, ListRequirement):
        raise TypeError("Expected a list requirement.")
    return value
