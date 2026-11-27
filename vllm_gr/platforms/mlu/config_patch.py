# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Keep GR transport keys out of native MLU configuration validation."""

from functools import wraps
from typing import Any


def is_mlu_config_filter_installed() -> bool:
    from vllm_mlu import mlu_config

    return getattr(mlu_config.init_mlu_config, "_vllm_gr_sanitized", False) is True


def install_mlu_config_filter() -> int:
    """Filter only during vendor init and restore the original mapping on exit."""
    from vllm_mlu import mlu_config

    from vllm_gr.config.parser import BACKEND_TRANSPORT_KEYS, GR_CONFIG_KEY

    if is_mlu_config_filter_installed():
        return 0
    original = mlu_config.init_mlu_config

    @wraps(original)
    def init_mlu_config(vllm_config: Any) -> Any:
        additional = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional, dict):
            return original(vllm_config)
        filtered = {
            key: value
            for key, value in additional.items()
            if key != GR_CONFIG_KEY and key not in BACKEND_TRANSPORT_KEYS
        }
        if len(filtered) == len(additional):
            return original(vllm_config)
        # Match Ascend's scoped filter: retain the engine config identity and
        # restore the exact mapping used for GR serialization and plan binding.
        vllm_config.additional_config = filtered
        try:
            return original(vllm_config)
        finally:
            vllm_config.additional_config = additional

    setattr(init_mlu_config, "_vllm_gr_sanitized", True)
    # Native MLU platform, EngineCore and Worker import this function locally
    # at each call. No module-global aliases need rebinding on this runtime.
    mlu_config.init_mlu_config = init_mlu_config
    return 1
