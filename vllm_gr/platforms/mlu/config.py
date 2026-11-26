# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""MLU process startup integration after native configuration is complete."""

from typing import Any

from vllm_gr.config import GRConfig

from ..config import BackendConfigService


class MLUConfigService(BackendConfigService):
    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        from .engine_core_patch import install_mlu_engine_core_entry
        from .scheduler_patch import install_mlu_scheduler_patch

        install_mlu_engine_core_entry()
        install_mlu_scheduler_patch(config)
