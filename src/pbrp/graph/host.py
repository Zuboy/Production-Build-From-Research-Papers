"""Connect the compiled graph to the Azure providers for a hosted process."""

from __future__ import annotations

import atexit
import os
from contextlib import ExitStack

from langgraph.graph.state import CompiledStateGraph

from pbrp.graph.state import PBRPState
from pbrp.graph.workflow import make_graph
from pbrp.spec_cli import planning_clients


def graph() -> CompiledStateGraph[PBRPState, None, PBRPState, PBRPState]:
    clients = ExitStack()
    try:
        workspace = clients.enter_context(planning_clients())
        compiled = make_graph(workspace, os.environ["PBRP_DEFAULT_RUN_ID"])
    except Exception:
        clients.close()
        raise
    atexit.register(clients.close)
    return compiled
