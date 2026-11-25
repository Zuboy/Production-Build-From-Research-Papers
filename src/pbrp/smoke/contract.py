"""Validate the fixed preset and bind an approval to exact stored plan bytes."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Literal

LORA_REPOSITORY = "https://github.com/microsoft/LoRA"
LORA_COMMIT = "c4593f060e6a368d7bb5af5273b8e42810cdef90"
LORA_PAPER = "e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a"
PRESET: Literal["lora-cpu-v1"] = "lora-cpu-v1"
QUEUE_NAME = "pbrp-smoke-approvals"
PLANS_CONTAINER = "pbrp-plans"
RESULTS_CONTAINER = "pbrp-smoke-results"


def validate_approval(plan_id: str, digest: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{32}", plan_id) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Provide the plan ID and SHA-256 exactly as shown by the agent.")


def verify_plan(plan_id: str, digest: str, raw_plan: bytes) -> str:
    validate_approval(plan_id, digest)
    if sha256(raw_plan).hexdigest() != digest:
        raise ValueError("The stored plan does not match the approved SHA-256.")
    data = json.loads(raw_plan)
    if not isinstance(data, dict) or not all(
        data.get(key) == expected
        for key, expected in {
            "schema_version": 1,
            "plan_id": plan_id,
            "state": "awaiting_approval",
            "preset": PRESET,
            "dataset": "synthetic",
            "timeout_seconds": 300,
            "replicas": 1,
            "retries": 0,
        }.items()
    ):
        raise ValueError("The plan does not describe the approved smoke preset.")
    spec = data.get("spec")
    source = spec.get("source") if isinstance(spec, dict) else None
    if not isinstance(source, dict) or not all(
        source.get(key) == expected
        for key, expected in {
            "repository_url": LORA_REPOSITORY,
            "commit_sha": LORA_COMMIT,
            "paper_sha256": LORA_PAPER,
            "run_mode": "smoke",
        }.items()
    ):
        raise ValueError("The plan source does not match the pinned LoRA smoke inputs.")
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", run_id):
        raise ValueError("The plan has an invalid ingestion run ID.")
    return run_id
