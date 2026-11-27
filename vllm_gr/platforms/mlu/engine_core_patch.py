# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compose vendor and GR EngineCore initialization in spawned MLU processes."""

from importlib import import_module
from inspect import unwrap
from typing import Any


def run_mlu_engine_core(
    *args: Any, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs: Any
) -> None:
    """Load vendor replacements before GR captures the engine initializer."""
    from .config_patch import install_mlu_config_filter

    install_mlu_config_filter()
    # General plugins may run GR before mlu_hijack. Import vendor replacements
    # first so GR wraps the MLU initializer, never an in-flight upstream init.
    import_module("vllm_mlu.mlu_hijack")

    from vllm.plugins import load_general_plugins
    from vllm.v1.engine.core import EngineCore
    from vllm_mlu.v1.engine.core import EngineCore_MluHijack, EngineCoreProc_MluHijack

    load_general_plugins()

    from vllm_gr.patch import initialize_engine_core_beam_patches

    # Plugin loading can replace previously installed methods. Revalidate the
    # complete child plan before the vendor entrypoint constructs any engine.
    initialize_engine_core_beam_patches()

    from .scheduler_patch import install_mlu_scheduler_patch

    # The shared plan verifies the upstream base. MLU selects a separate class
    # whose schedule override must be wrapped before any requests can run.
    install_mlu_scheduler_patch(kwargs["vllm_config"])
    if unwrap(EngineCore.__init__) is not EngineCore_MluHijack.__init__:
        raise RuntimeError("MLU EngineCore GR wrapper does not preserve vendor initialization")

    # Call the vendor function directly: the class attribute may now be a GR
    # wrapper. General plugins are already loaded, so native entry is stable.
    EngineCoreProc_MluHijack.run_engine_core(
        *args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs
    )


def is_mlu_engine_core_entry_installed() -> bool:
    from vllm.v1.engine.core import EngineCoreProc

    return EngineCoreProc.run_engine_core is run_mlu_engine_core


def install_mlu_engine_core_entry() -> int:
    """Reassert the picklable MLU entrypoint after native plugins/configuration."""
    from vllm.platforms import current_platform
    from vllm.v1.engine.core import EngineCoreProc

    if current_platform.device_type != "mlu" or is_mlu_engine_core_entry_installed():
        return 0
    EngineCoreProc.run_engine_core = staticmethod(run_mlu_engine_core)
    return 1
