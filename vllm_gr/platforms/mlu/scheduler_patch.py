# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Apply shared Beam scheduling to the class selected by the MLU platform."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from vllm_gr.config import GRConfig


class _DeferredAdmissionSet:
    """Count only the owner while writing native deferred-state changes through."""

    def __init__(self, requests: set[Any], owner_ids: tuple[str, ...]):
        self.requests = requests
        self.owner_ids = frozenset(owner_ids)

    def __contains__(self, request: Any) -> bool:
        return request in self.requests and request.request_id in self.owner_ids

    def __iter__(self) -> Iterator[Any]:
        return (r for r in self.requests if r.request_id in self.owner_ids)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def add(self, request: Any) -> None:
        self.requests.add(request)

    def discard(self, request: Any) -> None:
        self.requests.discard(request)

    def difference_update(self, requests: Iterable[Any]) -> None:
        self.requests.difference_update(requests)


@contextmanager
def mlu_admission_queues(scheduler: Any, owner_ids: tuple[str, ...]) -> Iterator[None]:
    """Filter both v0.33 selection queues and matching deferred bookkeeping."""
    from vllm_gr.v1.core.sched.beam_async_scheduler import _AdmissionQueue

    waiting = scheduler.waiting
    kv_holding = scheduler.kv_holding_waiting
    deferred = scheduler.deferred_waiting
    try:
        scheduler.waiting = _AdmissionQueue(waiting, owner_ids)
        scheduler.kv_holding_waiting = _AdmissionQueue(kv_holding, owner_ids)
        scheduler.deferred_waiting = _DeferredAdmissionSet(deferred, owner_ids)
        yield
    finally:
        scheduler.waiting = waiting
        scheduler.kv_holding_waiting = kv_holding
        scheduler.deferred_waiting = deferred


def validate_mlu_scheduler_patch(config: Any) -> str | None:
    if GRConfig.from_vllm_config(config).beam_max_width is None:
        return None

    from vllm_mlu.v1.core.sched.async_scheduler import AsyncScheduler

    from vllm_gr.v1.core.sched.scheduler_patch import validate_scheduler_patch

    scheduler_cls = config.scheduler_config.get_scheduler_cls()
    if scheduler_cls is not AsyncScheduler:
        return "MLU Beam scheduling currently requires the vendor AsyncScheduler"
    if getattr(scheduler_cls, "_gr_admission_queue_context", None) is not mlu_admission_queues:
        return "MLU Beam scheduling requires native v0.33 admission queues"
    return validate_scheduler_patch(scheduler_cls=scheduler_cls)


def install_mlu_scheduler_patch(config: Any) -> int:
    """Wrap vendor scheduling after plugin loading without replacing its algorithm."""
    from vllm.platforms import current_platform

    if (
        current_platform.device_type != "mlu"
        or GRConfig.from_vllm_config(config).beam_max_width is None
    ):
        return 0

    from vllm_mlu.v1.core.sched.async_scheduler import AsyncScheduler

    from vllm_gr.v1.core.sched.scheduler_patch import apply_scheduler_patch

    scheduler_cls = config.scheduler_config.get_scheduler_cls()
    if scheduler_cls is not AsyncScheduler:
        raise NotImplementedError(
            "MLU Beam scheduling currently requires the vendor AsyncScheduler"
        )

    was_installed = validate_mlu_scheduler_patch(config) is None
    scheduler_cls._gr_admission_queue_context = mlu_admission_queues
    apply_scheduler_patch(scheduler_cls=scheduler_cls)
    error = validate_mlu_scheduler_patch(config)
    if error is not None:
        raise RuntimeError(error)
    return int(not was_installed)
