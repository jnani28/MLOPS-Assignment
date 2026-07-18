"""Run model training with optional OpenLineage event emission."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone

from openlineage.client import OpenLineageClient
from openlineage.client.run import InputDataset, Job, OutputDataset, Run, RunEvent, RunState


def emit_event(client: OpenLineageClient, state: RunState, run_id: str) -> None:
    event = RunEvent(
        eventType=state,
        eventTime=datetime.now(timezone.utc).isoformat(),
        run=Run(runId=run_id),
        job=Job(namespace="mlops-assignment", name="model-training"),
        inputs=[
            InputDataset(namespace="file", name="data_raw/bank-additional.csv"),
        ],
        outputs=[
            OutputDataset(namespace="file", name="model.pkl"),
            OutputDataset(namespace="file", name="model_metadata.json"),
            OutputDataset(namespace="file", name="columns_info.json"),
        ],
        producer="https://github.com/jnani28/MLOPS-Assignment",
    )
    client.emit(event)


def main() -> int:
    url = os.getenv("OPENLINEAGE_URL")
    if not url:
        print("OPENLINEAGE_URL is not set. Running training without lineage emission.")
        completed = subprocess.run([sys.executable, "model_training.py"], check=False)
        return completed.returncode

    client = OpenLineageClient(url=url)
    run_id = str(uuid.uuid4())

    emit_event(client, RunState.START, run_id)
    completed = subprocess.run([sys.executable, "model_training.py"], check=False)

    final_state = RunState.COMPLETE if completed.returncode == 0 else RunState.FAIL
    emit_event(client, final_state, run_id)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
