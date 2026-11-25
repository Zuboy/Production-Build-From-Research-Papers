"""Extract a specification for one completed Azure ingestion run."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from pydantic import TypeAdapter

from pbrp.config import IngestionSettings
from pbrp.providers.llm import FoundryLLMProvider
from pbrp.providers.storage import AzureBlobArtifactStore
from pbrp.providers.vectorstore import AzureSearchVectorStore
from pbrp.smoke.contract import PLANS_CONTAINER, RESULTS_CONTAINER
from pbrp.spec.models import RunId, RunSource
from pbrp.spec.service import build_spec, retrieve_spec_evidence


@dataclass
class PlanningClients:
    storage: AzureBlobArtifactStore
    search: AzureSearchVectorStore
    llm: FoundryLLMProvider
    artifact_container: str
    knowledge_index: str

    def load_source(self, run_id: str) -> RunSource:
        run_id = TypeAdapter(RunId).validate_python(run_id)
        manifest = self.storage.download(self.artifact_container, f"runs/{run_id}/manifest.json")
        source = RunSource.model_validate_json(manifest)
        if source.run_id != run_id or source.knowledge_index_name != self.knowledge_index:
            raise ValueError("The ingestion manifest does not match the requested run and index.")
        return source

    def publish_plan(self, plan_id: str, payload: str) -> str:
        return self.storage.create_json(PLANS_CONTAINER, f"plans/{plan_id}.json", payload)

    def read_result(self, plan_id: str) -> str | None:
        try:
            return self.storage.download(RESULTS_CONTAINER, f"results/{plan_id}.json").decode()
        except ResourceNotFoundError:
            return None


@contextmanager
def planning_clients() -> Iterator[PlanningClients]:
    settings = IngestionSettings.from_env()
    endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT") or os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    credential = (
        ManagedIdentityCredential() if os.getenv("IDENTITY_ENDPOINT") else DefaultAzureCredential()
    )
    with credential:
        storage = AzureBlobArtifactStore(
            endpoint=settings.azure_storage_endpoint, credential=credential
        )
        try:
            search = AzureSearchVectorStore(
                endpoint=settings.azure_search_endpoint,
                index_name=settings.knowledge_index_name,
                semantic_configuration_name=settings.knowledge_semantic_configuration_name,
                credential=credential,
            )
            try:
                llm = FoundryLLMProvider(
                    project_endpoint=endpoint,
                    deployment_name=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
                    credential=credential,
                )
                try:
                    yield PlanningClients(
                        storage,
                        search,
                        llm,
                        settings.artifact_container_name,
                        settings.knowledge_index_name,
                    )
                finally:
                    llm.close()
            finally:
                search.close()
        finally:
            storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    with planning_clients() as clients:
        source = clients.load_source(args.run_id)
        chunks = retrieve_spec_evidence(clients.search, source)
        spec = build_spec(source, chunks, clients.llm)
        print(spec.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
