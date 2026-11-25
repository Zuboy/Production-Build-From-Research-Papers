"""Azure worker handoff through an operator-only approval queue."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from azure.core.credentials import TokenCredential
from azure.storage.queue import QueueClient

from pbrp.smoke.contract import QUEUE_NAME, validate_approval


@dataclass(frozen=True)
class Approval:
    plan_id: str
    plan_sha256: str

    def __post_init__(self) -> None:
        validate_approval(self.plan_id, self.plan_sha256)


class AzureSmokeRunner:
    def __init__(self, *, queue_endpoint: str, credential: TokenCredential) -> None:
        self._queue = QueueClient(queue_endpoint, QUEUE_NAME, credential=credential)

    def submit(self, approval: Approval) -> str:
        """Submit the reviewed ID and digest, with a ten-minute delivery window."""

        return self._queue.send_message(json.dumps(asdict(approval)), time_to_live=600).id

    def close(self) -> None:
        self._queue.close()
