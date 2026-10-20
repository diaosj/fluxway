from collections import deque
from collections.abc import Mapping
from typing import Any

import pytest

from fluxway.common.types import CheckpointCut, ControllerState, DriverState
from fluxway.control_plane.components import (
    ControlPlaneComponent,
    ControlPlaneComponentRegistry,
    DriverComponentRegistry,
)


class MockLinearizedSourceCoordinator:
    def __init__(self, component_id: str = "spmd:read", stream_id: str = "stream1"):
        self.component_id = component_id
        self.stream_id = stream_id
        self.unpublished_records: deque[dict[str, Any]] = deque()
        self.builder_next_position = 0
        self.last_saved_cut: CheckpointCut | None = None
        self.last_restored_cut: CheckpointCut | None = None

    def generate(self, count: int) -> None:
        for _ in range(count):
            self.unpublished_records.append(
                {
                    "stream_id": self.stream_id,
                    "position": self.builder_next_position,
                    "payload": f"payload_{self.builder_next_position}",
                }
            )
            self.builder_next_position += 1

    async def on_state(self, state: ControllerState | DriverState) -> None:
        pass

    def _rebase_unpublished(self, stream_cut: int) -> None:
        if stream_cut > self.builder_next_position:
            raise RuntimeError(
                f"Causality violation: stream_cut {stream_cut} exceeds builder next_position {self.builder_next_position}"
            )
        while (
            self.unpublished_records
            and self.unpublished_records[0]["position"] < stream_cut
        ):
            self.unpublished_records.popleft()

        expected = stream_cut
        for record in self.unpublished_records:
            if record["position"] != expected:
                raise RuntimeError(
                    f"Unpublished queue gap or disorder: expected={expected} actual={record['position']}"
                )
            expected += 1
        if expected != self.builder_next_position:
            raise RuntimeError(
                f"Unpublished queue tail mismatch: expected={self.builder_next_position} actual={expected}"
            )

    def checkpoint_state(self, cut: CheckpointCut) -> dict[str, Any]:
        if not isinstance(cut, CheckpointCut):
            raise TypeError("checkpoint requires CheckpointCut")
        self.last_saved_cut = cut
        try:
            stream_cut = cut.stream_cuts[self.stream_id]
        except KeyError as error:
            raise ValueError(
                f"missing authoritative stream cut: {self.stream_id!r}"
            ) from error
        self._rebase_unpublished(stream_cut)
        return {
            "unpublished_records": list(self.unpublished_records),
            "builder_next_position": self.builder_next_position,
        }

    def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
        if not isinstance(cut, CheckpointCut):
            raise TypeError("restore requires CheckpointCut")
        self.last_restored_cut = cut
        self.builder_next_position = state["builder_next_position"]
        self.unpublished_records = deque(state["unpublished_records"])
        try:
            stream_cut = cut.stream_cuts[self.stream_id]
        except KeyError as error:
            raise ValueError(
                f"missing authoritative stream cut: {self.stream_id!r}"
            ) from error
        self._rebase_unpublished(stream_cut)


def test_component_protocol_compliance():
    coord = MockLinearizedSourceCoordinator()
    assert isinstance(coord, ControlPlaneComponent)


def test_driver_component_registry_propagates_cut():
    registry = DriverComponentRegistry()
    coord = MockLinearizedSourceCoordinator(component_id="reader:read")
    registry.register(coord)

    coord.generate(5)  # positions 0, 1, 2, 3, 4. builder_next_position = 5
    # Simulate unacknowledged append of 0, 1, 2, 3 (durable cut = 4)
    cut = CheckpointCut(
        cut_id="cut-4",
        position=4,
        stream_cuts={"stream1": 4},
    )

    state = registry.checkpoint_state(cut)
    assert coord.last_saved_cut == cut

    # Position 0..3 must be discarded, position 4 must be retained
    saved_records = state["reader:read"]["unpublished_records"]
    assert len(saved_records) == 1
    assert saved_records[0]["position"] == 4

    # Restore in a fresh registry
    new_registry = DriverComponentRegistry()
    restored_coord = MockLinearizedSourceCoordinator(component_id="reader:read")
    new_registry.register(restored_coord)

    new_registry.restore_state(state, cut)
    assert restored_coord.last_restored_cut == cut
    assert len(restored_coord.unpublished_records) == 1
    assert restored_coord.unpublished_records[0]["position"] == 4
    assert restored_coord.builder_next_position == 5


def test_rebase_causality_violation_raises():
    registry = DriverComponentRegistry()
    coord = MockLinearizedSourceCoordinator(component_id="reader:read")
    registry.register(coord)

    coord.generate(3)  # builder_next_position = 3
    cut = CheckpointCut(
        cut_id="cut-invalid",
        position=5,
        stream_cuts={"stream1": 5},
    )

    with pytest.raises(RuntimeError, match="Causality violation"):
        registry.checkpoint_state(cut)


def test_rebase_queue_gap_raises():
    registry = DriverComponentRegistry()
    coord = MockLinearizedSourceCoordinator(component_id="reader:read")
    registry.register(coord)

    coord.generate(4)  # 0, 1, 2, 3
    # Inject corruption: remove position 2
    del coord.unpublished_records[2]  # now contains 0, 1, 3
    cut = CheckpointCut(
        cut_id="cut-2",
        position=1,
        stream_cuts={"stream1": 1},
    )

    with pytest.raises(RuntimeError, match="Unpublished queue gap"):
        registry.checkpoint_state(cut)


def test_controller_component_registry_requires_checkpoint_cut():
    registry = ControlPlaneComponentRegistry()
    coord = MockLinearizedSourceCoordinator(component_id="global:coord")
    registry.register(coord)

    with pytest.raises(TypeError, match="requires CheckpointCut"):
        registry.checkpoint_state(None)  # type: ignore

    with pytest.raises(TypeError, match="requires CheckpointCut"):
        registry.restore_state({"global:coord": {}}, "invalid-cut")  # type: ignore
