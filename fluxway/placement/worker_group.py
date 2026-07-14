from dataclasses import dataclass, field
from decimal import Decimal
from collections.abc import Mapping
from typing import Any

from .planner import PlacementAttestationError


def ray_actor_options(resources: Mapping[str, Decimal]) -> dict[str, Any]:
    values = {resource: float(quantity) for resource, quantity in resources.items()}
    options: dict[str, Any] = {
        "num_cpus": values.pop("CPU", 0),
        "num_gpus": values.pop("GPU", 0),
    }
    memory = values.pop("memory", 0)
    if memory > 0:
        options["memory"] = memory
    custom = {
        resource: quantity for resource, quantity in values.items() if quantity > 0
    }
    if custom:
        options["resources"] = custom
    return options


def attest_node_ids(
    spmd_id: str,
    expected: tuple[str, ...],
    actual: tuple[str, ...],
) -> None:
    if actual != expected:
        raise PlacementAttestationError(
            f"placement attestation failed spmd_id={spmd_id} "
            f"expected={expected} actual={actual}"
        )


@dataclass(frozen=True)
class SPMDWorkerGroup:
    spmd_id: str
    placement_group: Any
    rank_to_bundle: tuple[int, ...]
    expected_node_ids: tuple[str, ...]
    worker_resources: dict[str, Decimal]
    driver_bundle_index: int = 0
    driver_resources: dict[str, Decimal] = field(default_factory=dict)

    def attest(self, actual_node_ids: tuple[str, ...]) -> None:
        attest_node_ids(self.spmd_id, self.expected_node_ids, actual_node_ids)
