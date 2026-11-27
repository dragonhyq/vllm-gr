# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""MLU V2 lifecycle wiring and eager Beam stage dispatch."""

from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import Any

from vllm_gr.v1.beam.dispatch import GR_RETIRE_OUTPUT_ATTR, GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.worker.mlu.numpy_staging import (
    install_numpy_staging,
    is_numpy_staging_installed,
)
from vllm_gr.v1.worker.mlu.runner_installer import V2_RUNNER
from vllm_gr.v1.worker.runner_binding import (
    bind_after_model_load,
    close_backend_binding,
)
from vllm_gr.v1.worker.runner_lifecycle import (
    has_runner_lifecycle,
    install_runner_lifecycle,
)

_METHODS = ("load_model", "prepare_inputs", "execute_model", "sample_tokens", "shutdown")
_INSTALLED_METHODS = "_gr_mlu_lifecycle_methods"
_INSTALL_LOCK = RLock()


def discover_worker_runners() -> tuple[tuple[str, type[Any]], ...]:
    from vllm_mlu.v1.worker.gpu.model_runner import MLUModelRunner

    return (("MLUModelRunnerV2", MLUModelRunner),)


def is_patched(runner_class: type[Any]) -> bool:
    installed = runner_class.__dict__.get(_INSTALLED_METHODS)
    return (
        isinstance(installed, dict)
        and set(installed) == set(_METHODS)
        and all(runner_class.__dict__.get(name) is installed[name] for name in _METHODS)
        and has_runner_lifecycle(runner_class, V2_RUNNER)
        and is_numpy_staging_installed()
    )


def _guard_execute(original: Any) -> Any:
    @wraps(original)
    def execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        from .beam_stage_runner import MLUEagerBeamStageRunner, execute_retirement

        try:
            if getattr(scheduler_output, "beam_data", None):
                raise NotImplementedError("MLU V2 does not support legacy beam_data frames")
            if getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR, None) is not None:
                output = execute_retirement(self, scheduler_output)
                self._gr_mlu_execution_error = None
                return output
            if getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None):
                owner = getattr(self, "_gr_mlu_stage_runner", None)
                if owner is None or owner.closed:
                    owner = MLUEagerBeamStageRunner(self)
                    self._gr_mlu_stage_runner = owner
                return owner.execute(scheduler_output, original, *args, **kwargs)
            return original(self, scheduler_output, *args, **kwargs)
        except Exception as error:
            # Async workers may sample after an execute failure. Never let that
            # consume stale native logits from an earlier request.
            self._gr_mlu_execution_error = error
            raise error

    return execute


def _guard_sample(original: Any) -> Any:
    @wraps(original)
    def sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        error = getattr(self, "_gr_mlu_execution_error", None)
        if error is not None:
            raise error
        owner = getattr(self, "_gr_mlu_stage_runner", None)
        if owner is not None and (owner.pending is not None or owner.execution_error is not None):
            return owner.sample(*args, **kwargs)
        return original(self, *args, **kwargs)

    return sample


def _close_on_shutdown(original: Any) -> Any:
    @wraps(original)
    def shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        from .beam_stage_runner import close_stage_runner

        try:
            close_stage_runner(self)
            return original(self, *args, **kwargs)
        finally:
            close_backend_binding(self)

    return shutdown


def install() -> None:
    """Bind the vendor subclass without modifying the upstream CUDA runner."""
    with _INSTALL_LOCK:
        ((_, runner_class),) = discover_worker_runners()
        install_numpy_staging()
        if is_patched(runner_class):
            return
        if _INSTALLED_METHODS in runner_class.__dict__:
            raise RuntimeError("MLU runner lifecycle hooks were replaced after installation")
        if runner_class.__dict__.get("_gr_runner_implementation_id") is not None:
            raise RuntimeError("MLU runner already has a GR lifecycle owner")
        for name in _METHODS:
            if not callable(getattr(runner_class, name, None)):
                raise RuntimeError(f"Native MLU runner does not implement {name}()")

        from .beam_stage_runner import close_stage_runner

        runner_class.load_model = bind_after_model_load(
            runner_class.load_model, close_runtime=close_stage_runner
        )
        runner_class.execute_model = _guard_execute(runner_class.execute_model)
        runner_class.sample_tokens = _guard_sample(runner_class.sample_tokens)
        runner_class.shutdown = _close_on_shutdown(runner_class.shutdown)
        install_runner_lifecycle(runner_class, V2_RUNNER, prepare="prepare_inputs")
        setattr(
            runner_class,
            _INSTALLED_METHODS,
            {name: runner_class.__dict__[name] for name in _METHODS},
        )
