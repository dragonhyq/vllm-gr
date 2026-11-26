# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""MLU discovery, native V2 lifecycle, and provisional eager Beam execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NoReturn, cast

from vllm_gr.config.backend_plan import BackendPlan, BackendRequest, PlanData

from ..compat import SUPPORTED_VLLM_MLU_VERSION
from ..identity import PlatformIdentity, current_platform_identity
from ..interface import (
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    ModelRunnerInstaller,
)
from ..services import BuiltinPlanSupport, require_provider_plan

if TYPE_CHECKING:
    import torch

    from ..config import BackendConfigService


def _execution_unavailable() -> NoReturn:
    raise NotImplementedError("MLU GR constrained top-k execution is not implemented")


@dataclass(frozen=True)
class MLUBackendProvider(BuiltinPlanSupport):
    backend_id: str = "mlu"
    device_type: str = "mlu"
    # Native lifecycle, config, child startup, token staging, and Beam scheduling.
    patch_ids: ClassVar[tuple[str, ...]] = (
        "GR-PATCH-023",
        "GR-PATCH-024",
        "GR-PATCH-025",
        "GR-PATCH-026",
        "GR-PATCH-027",
    )
    api_version: ClassVar[int] = PROVIDER_API_VERSION
    gr_version_range: ClassVar[str | None] = ">=0.30.0,<0.31.0"
    vllm_version_range: ClassVar[str | None] = f"=={SUPPORTED_VLLM_MLU_VERSION}"
    # The vendor plugin version is internal; compatibility targets vLLM.
    vendor_version_range: ClassVar[None] = None
    vendor_distribution: ClassVar[str | None] = "vllm-mlu"

    def matches(self, identity: PlatformIdentity) -> bool:
        # Other out-of-tree platforms also use OOT: require the vendor lineage.
        return identity.device_type == "mlu" and identity.has_lineage(
            "vllm_mlu.platforms.mlu.MLUPlatform"
        )

    def check_environment(self) -> dict[str, str]:
        from ..compat import installed_version, require_runtime_version

        if not self.matches(current_platform_identity()):
            raise RuntimeError("MLU provider requires the native MLU platform")
        report = {
            "platform": "mlu",
            "vllm": require_runtime_version("vllm", SUPPORTED_VLLM_MLU_VERSION),
        }
        vendor_version = installed_version("vllm-mlu")
        if vendor_version is not None:
            report["vllm-mlu"] = vendor_version
        return report

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        from .config_patch import install_mlu_config_filter

        del force
        return install_mlu_config_filter()

    def create_config_service(self) -> BackendConfigService:
        from .config import MLUConfigService

        return MLUConfigService()

    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Admit V2 startup, including Beam plans, while execution is being ported."""
        from vllm_gr.v1.worker.mlu.runner_installer import V2_RUNNER

        if request.options.to_dict():
            raise ValueError("MLU currently accepts no additional backend options")
        gr = request.config.to_dict()
        native = request.native.to_dict()
        if not native["use_v2_model_runner"]:
            raise ValueError("MLU GR runner integration requires ModelRunner V2")
        if request.has_constraint_table:
            raise NotImplementedError("MLU GR constraint-table execution is not implemented")
        if gr["npu_decode_execution"] == "xllm":
            raise ValueError("MLU does not support npu_decode_execution='xllm'")

        # Beam admission is provisional: preserve its configuration in request
        # while selecting the implemented runner lifecycle. CUSTOM attention is
        # still required when requested; only eager attention is implemented.
        # The eager stage adapter validates its narrower execution scope before
        # running. Graphs and constraints remain unsupported. See the tracking
        # record for real-request validation and unresolved runtime boundaries.
        return BackendPlan(
            self.backend_id,
            self.api_version,
            request,
            PlanData(
                {
                    "runner": V2_RUNNER,
                    "attention": "vllm_gr.v1.attention.backends.beam_attn_mlu",
                    "requires_attention": request.uses_custom_attention,
                    "constraint": "none",
                }
            ),
        )

    def create_attention_executor(
        self, *, plan: BackendPlan, device: object
    ) -> BeamAttentionExecution:
        require_provider_plan(self, plan)
        if str(device).split(":", 1)[0] != "mlu":
            raise ValueError("MLU attention requires an MLU device")
        execution = self.get_attention_execution()
        if not execution.is_available():
            raise RuntimeError("MLU TMO attention operations are unavailable")
        return execution

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        from vllm_gr.v1.worker.mlu import runner_installer

        return runner_installer

    def get_attention_execution(self) -> BeamAttentionExecution:
        from vllm_gr.v1.attention.backends import beam_attn_mlu

        return cast(BeamAttentionExecution, beam_attn_mlu)

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
        plan: BackendPlan | None = None,
    ) -> NoReturn:
        _execution_unavailable()


MLU_BACKEND = MLUBackendProvider()
