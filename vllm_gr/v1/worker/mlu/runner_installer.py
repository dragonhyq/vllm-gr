# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Defer MLU V2 runner integration until native Worker device initialization."""

from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import Any

from vllm_gr.config.backend_plan import read_backend_plan

V2_RUNNER = "vllm_gr.v1.worker.mlu.model_runner_patch"
_WORKER_MARKER = "_gr_mlu_runner_selection"
_INSTALL_LOCK = RLock()


def discover_worker_runners() -> tuple[tuple[str, type[Any]], ...]:
    # As with CUDA, initial verification covers the deferred Worker hook.
    # Import the native runner only when init_device is about to construct it.
    from vllm_mlu.v1.worker.gpu_worker import MLUWorker

    return (("MLUWorker.init_device", MLUWorker),)


def is_patched(worker_class: type[Any]) -> bool:
    method = worker_class.__dict__.get("init_device")
    return callable(method) and callable(getattr(method, _WORKER_MARKER, None))


def install() -> None:
    """Install an idempotent hook on the vendor Worker, not its CUDA base."""
    with _INSTALL_LOCK:
        ((_, worker_class),) = discover_worker_runners()
        if is_patched(worker_class):
            return
        original = worker_class.init_device

        @wraps(original)
        def init_device(worker: Any, *args: Any, **kwargs: Any) -> Any:
            if not worker.use_v2_model_runner:
                raise ValueError("MLU GR runner integration requires ModelRunner V2")
            plan = read_backend_plan(worker.vllm_config)
            if plan is not None and (
                plan.backend_id != "mlu" or plan.choices.to_dict().get("runner") != V2_RUNNER
            ):
                raise RuntimeError("MLU Worker runner differs from the admitted backend plan")

            from vllm_mlu._mlu_utils import VLLM_MLU_DP_AR_OVERLAP

            if worker.vllm_config.is_mm_encoder_only or VLLM_MLU_DP_AR_OVERLAP:
                raise NotImplementedError(
                    "MLU GR runner integration currently targets the standard MLUModelRunner; "
                    "encoder-only and overlap runners are not supported"
                )

            from vllm_gr.v1.worker.mlu import model_runner_patch

            with _INSTALL_LOCK:
                model_runner_patch.install()
                ((_, runner_class),) = model_runner_patch.discover_worker_runners()
                if not model_runner_patch.is_patched(runner_class):
                    raise RuntimeError("MLU V2 runner lifecycle hooks were not installed")
            result = original(worker, *args, **kwargs)
            if type(worker.model_runner) is not runner_class:
                raise RuntimeError("MLU Worker constructed an unexpected native runner")
            return result

        setattr(init_device, _WORKER_MARKER, original)
        worker_class.init_device = init_device
