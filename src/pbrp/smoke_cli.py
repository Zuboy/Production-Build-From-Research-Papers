"""Approve a reviewed LoRA smoke plan or read its result using Azure sign-in."""

from __future__ import annotations

import argparse
import os

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from pydantic import TypeAdapter

from pbrp.graph.state import PlanId
from pbrp.providers.runner import Approval, AzureSmokeRunner
from pbrp.providers.storage import AzureBlobArtifactStore
from pbrp.smoke.contract import PLANS_CONTAINER, RESULTS_CONTAINER, verify_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    approve = commands.add_parser("approve")
    approve.add_argument("--plan-id", required=True)
    approve.add_argument("--sha256", required=True)
    status = commands.add_parser("status")
    status.add_argument("--plan-id", required=True)
    args = parser.parse_args()
    plan_id = TypeAdapter(PlanId).validate_python(args.plan_id)

    with DefaultAzureCredential() as credential:
        blob_endpoint = os.environ["AZURE_STORAGE_ENDPOINT"]
        storage = AzureBlobArtifactStore(endpoint=blob_endpoint, credential=credential)
        try:
            if args.command == "status":
                try:
                    print(storage.download(RESULTS_CONTAINER, f"results/{plan_id}.json").decode())
                except ResourceNotFoundError:
                    print("No completed experiment result exists for this plan yet.")
                return

            approval = Approval(plan_id=plan_id, plan_sha256=args.sha256)
            raw_plan = storage.download(PLANS_CONTAINER, f"plans/{plan_id}.json")
            verify_plan(plan_id, approval.plan_sha256, raw_plan)
            runner = AzureSmokeRunner(
                queue_endpoint=os.environ["AZURE_STORAGE_QUEUE_ENDPOINT"], credential=credential
            )
            try:
                print(
                    f"Queued approved smoke experiment {plan_id}. Message {runner.submit(approval)}"
                )
            finally:
                runner.close()
        finally:
            storage.close()


if __name__ == "__main__":
    main()
