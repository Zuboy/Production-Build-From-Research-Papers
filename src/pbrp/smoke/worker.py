"""Consume one Azure approval and write the bounded smoke experiment report."""

from __future__ import annotations

import json
import os
from time import monotonic

from azure.core.exceptions import ResourceExistsError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient

from pbrp.smoke.contract import (
    PLANS_CONTAINER,
    PRESET,
    QUEUE_NAME,
    RESULTS_CONTAINER,
    validate_approval,
    verify_plan,
)
from pbrp.smoke.experiment import run_experiment


def main() -> None:
    with (
        ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"]) as credential,
        QueueClient(os.environ["AZURE_STORAGE_QUEUE_ENDPOINT"], QUEUE_NAME, credential) as queue,
        BlobServiceClient(os.environ["AZURE_STORAGE_ENDPOINT"], credential=credential) as blobs,
    ):
        message = queue.receive_message(visibility_timeout=360)
        if message is None:
            return
        try:
            plan_id, digest = _approval_fields(message.content)
        except ValueError:
            queue.delete_message(message)
            print("Discarded an invalid smoke approval.", flush=True)
            return

        report = blobs.get_blob_client(RESULTS_CONTAINER, f"results/{plan_id}.json")
        if report.exists():
            queue.delete_message(message)
            return

        plan = blobs.get_blob_client(PLANS_CONTAINER, f"plans/{plan_id}.json")
        raw_plan = plan.download_blob(offset=0, length=1_048_577).readall()
        try:
            if len(raw_plan) > 1_048_576:
                raise ValueError("Plan exceeds its size limit.")
            run_id = verify_plan(plan_id, digest, raw_plan)
        except ValueError:
            queue.delete_message(message)
            print("Discarded an approval whose plan failed verification.", flush=True)
            return

        start = monotonic()
        metrics = run_experiment()
        result = {
            "run_id": run_id,
            "plan_id": plan_id,
            "plan_sha256": digest,
            "preset": PRESET,
            "dataset": "synthetic",
            "status": "succeeded" if metrics["passed"] else "failed",
            "metrics": metrics,
            "elapsed_seconds": monotonic() - start,
            "image": os.environ.get("PBRP_SMOKE_IMAGE"),
            "azure_execution": os.getenv("CONTAINER_APP_JOB_EXECUTION_NAME", os.getenv("HOSTNAME")),
        }
        try:
            report.upload_blob(
                json.dumps(result, indent=2),
                overwrite=False,
                content_settings=ContentSettings(content_type="application/json"),
            )
        except ResourceExistsError:
            pass  # A previous delivery already published the immutable result.
        queue.delete_message(message)
        print(json.dumps(result), flush=True)
        if not metrics["passed"]:
            raise SystemExit(1)


def _approval_fields(content: str | None) -> tuple[str, str]:
    value = json.loads(content or "null")
    if not isinstance(value, dict) or set(value) != {"plan_id", "plan_sha256"}:
        raise ValueError("Invalid approval fields.")
    plan_id, digest = value["plan_id"], value["plan_sha256"]
    if not isinstance(plan_id, str) or not isinstance(digest, str):
        raise ValueError("Approval fields must be strings.")
    validate_approval(plan_id, digest)
    return plan_id, digest


if __name__ == "__main__":
    main()
