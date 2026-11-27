# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host contracts for native request lifetime in the eager MLU Beam adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRRetireMetadata,
)
from vllm_gr.v1.worker.mlu import beam_stage_runner as mlu

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def make_worker():
    live = {"unrelated-request"}

    def finish(scheduled):
        live.difference_update(scheduled.finished_req_ids)

    worker = SimpleNamespace(
        live=live,
        finish_requests=Mock(side_effect=finish),
        free_states=Mock(),
        update_requests=Mock(),
        block_tables=SimpleNamespace(apply_staged_writes=Mock()),
        execute_model_state=None,
    )
    return worker


def make_owner(worker, monkeypatch, request_id, *, terminal):
    owner = mlu.MLUEagerBeamStageRunner(worker)
    owner.logits = torch.zeros((1, 4))
    owner.pending = SimpleNamespace(
        session_id="session",
        request_ids=(request_id,),
        beam_params=SimpleNamespace(temperature=0.0),
        is_last_stage=terminal,
        dispatch_id=0,
        stage_index=0,
        owner_generation=1,
        output_device_state=None,
    )
    monkeypatch.setattr(
        mlu.BeamBatchExecutionView,
        "from_stage_metadata",
        lambda _: SimpleNamespace(num_execution_rows=1, candidate_top_k=2, beam_width=2),
    )
    monkeypatch.setattr(owner, "_terminal_result", lambda: object())

    def drain():
        owner.closed = True

    monkeypatch.setattr(owner, "close", drain)
    return owner


def test_terminal_results_release_native_capacity_before_acknowledging(monkeypatch):
    worker = make_worker()
    # Keep one unrelated resident request while reusing the other slot more
    # times than the worker's total capacity. A missing cleanup fails here.
    for number in range(5):
        assert len(worker.live) < 2
        request_id = f"native-{number}"
        worker.live.add(request_id)
        owner = make_owner(worker, monkeypatch, request_id, terminal=True)
        output = owner.sample()
        receipt = getattr(output, GR_WORKER_RESULT_ATTR)["session"]
        assert receipt.resources_released
        assert owner.closed
        assert worker.live == {"unrelated-request"}
    assert worker.finish_requests.call_count == 5


def test_nonterminal_result_keeps_native_request(monkeypatch):
    worker = make_worker()
    worker.live.add("native")
    owner = make_owner(worker, monkeypatch, "native", terminal=False)
    output = owner.sample()
    assert not getattr(output, GR_WORKER_RESULT_ATTR)["session"].resources_released
    assert worker.live == {"unrelated-request", "native"}
    worker.finish_requests.assert_not_called()


def test_explicit_retirement_uses_native_id_without_mutating_scheduler_frame():
    worker = make_worker()
    worker.live.add("native")
    owner = SimpleNamespace(
        closed=False,
        pending=None,
        previous=SimpleNamespace(session_id="session", owner_generation=1, request_ids=("native",)),
        close=Mock(),
    )
    worker._gr_mlu_stage_runner = owner
    scheduled = SchedulerOutput.make_empty()
    scheduled.finished_req_ids.add("session")
    setattr(scheduled, GR_RETIRE_OUTPUT_ATTR, GRRetireMetadata("session", 1, 2))
    first = mlu.execute_retirement(worker, scheduled)
    assert worker.live == {"unrelated-request"}
    assert scheduled.finished_req_ids == {"session"}
    assert mlu.execute_retirement(worker, scheduled) is first
    worker.finish_requests.assert_called_once()
    owner.close.assert_called_once()
