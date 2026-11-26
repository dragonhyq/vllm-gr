# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reuse bulk NumPy staging with the vendor vLLM 0.33 write contract."""

from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import Any

import numpy as np

_INSTALL_LOCK = RLock()
_TRANSFER_MARKER = "_gr_mlu_numpy_transfer"


def is_numpy_staging_installed() -> bool:
    from vllm.v1.worker.gpu import buffer_utils

    transfer = getattr(buffer_utils.StagedWriteTensor.stage_write, _TRANSFER_MARKER, None)
    return callable(transfer) and transfer is buffer_utils.async_tensor_h2d


def install_numpy_staging() -> int:
    """Install after vendor replacements, preserving native device-write methods."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "mlu":
        return 0

    from vllm.v1.worker.gpu import buffer_utils

    from vllm_gr.v1.worker.cuda.numpy_staging import (
        install_numpy_staging as install_bulk_numpy_staging,
    )

    with _INSTALL_LOCK:
        if is_numpy_staging_installed():
            return 0
        current = buffer_utils.StagedWriteTensor.stage_write
        if getattr(current, "_gr_numpy_tokens", False):
            raise RuntimeError("MLU NumPy staging has an unexpected or replaced patch binding")

        # These helpers collect host arrays and wrap native transfer. They do
        # not install CUDA kernels or replace the vendor apply_write methods.
        install_bulk_numpy_staging()
        bulk_stage = buffer_utils.StagedWriteTensor.stage_write

        @wraps(bulk_stage)
        def stage_write(self: Any, index: int, start: int, values: Any) -> None:
            bulk_stage(self, index, start, values)
            if isinstance(values, np.ndarray) and values.size:
                # vLLM 0.33 uses this length to launch every 1024-element chunk.
                # Native list/scalar writes already maintain it themselves.
                self._staged_write_max_len = max(self._staged_write_max_len, values.size)

        setattr(stage_write, _TRANSFER_MARKER, buffer_utils.async_tensor_h2d)
        buffer_utils.StagedWriteTensor.stage_write = stage_write  # type: ignore[method-assign]
        return 1
