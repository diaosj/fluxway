from __future__ import annotations

from dataclasses import dataclass

from fluxway.profiler.record import stable_u64


TRACE_ROOT_TRACK_UUID = stable_u64("fluxway-profiler", "root-track")
TRACE_ROOT_TRACK_NAME = "Fluxway MPMD"


_ROLE_ORDER = {
    "Control Plane": 0,
    "SPMDs": 10,
    "controller": 0,
    "driver": 0,
    "workers": 10,
}

_SUBSYSTEM_ORDER = {
    "lifecycle": 0,
    "topology": 10,
    "spmd_manager": 20,
    "fsm": 30,
    "drain": 40,
    "checkpoint": 50,
    "source": 60,
    "completion": 70,
    "backpressure": 80,
    "profiler": 90,
    "compute": 100,
    "tensor": 110,
    "transport": 120,
    "mailbox": 130,
    "worker": 140,
    "heartbeat": 900,
}


@dataclass(frozen=True, slots=True)
class TraceTrackNode:
    uuid: int
    parent_uuid: int
    name: str
    sibling_order_rank: int


@dataclass(frozen=True, slots=True)
class TraceTrackLayout:
    nodes: tuple[TraceTrackNode, ...]
    leaf_uuid: int


@dataclass(frozen=True, slots=True)
class TraceTrackHierarchy:
    groups: tuple[str, ...]
    leaf_name: str

    @classmethod
    def from_track_name(cls, track_name: str) -> TraceTrackHierarchy:
        parts = tuple(part for part in track_name.split("/") if part)
        if not parts:
            raise ValueError("trace track name must be non-empty.")
        if parts[0] == "controller":
            return cls._from_controller(parts, track_name)
        if parts[0] == "driver":
            return cls._from_driver(parts, track_name)
        if parts[0] == "worker":
            return cls._from_worker(parts, track_name)
        raise ValueError(f"unknown Fluxway profiler track root: {track_name!r}")

    @classmethod
    def _from_controller(
        cls, parts: tuple[str, ...], track_name: str
    ) -> TraceTrackHierarchy:
        if len(parts) < 3 or parts[1] != "controller":
            raise ValueError(f"invalid controller profiler track: {track_name!r}")
        return cls(
            groups=("Control Plane", "controller"),
            leaf_name="/".join(parts[2:]),
        )

    @classmethod
    def _from_driver(
        cls, parts: tuple[str, ...], track_name: str
    ) -> TraceTrackHierarchy:
        if len(parts) < 4 or parts[2] != "driver":
            raise ValueError(f"invalid driver profiler track: {track_name!r}")
        return cls(
            groups=("SPMDs", parts[1], "driver"),
            leaf_name="/".join(parts[3:]),
        )

    @classmethod
    def _from_worker(
        cls, parts: tuple[str, ...], track_name: str
    ) -> TraceTrackHierarchy:
        if len(parts) < 5 or not parts[2].startswith("rank:"):
            raise ValueError(f"invalid worker profiler track: {track_name!r}")
        return cls(
            groups=("SPMDs", parts[1], "workers", parts[2], parts[3]),
            leaf_name="/".join(parts[4:]),
        )

    def to_layout(self, *, leaf_uuid: int) -> TraceTrackLayout:
        nodes: list[TraceTrackNode] = [
            TraceTrackNode(
                uuid=TRACE_ROOT_TRACK_UUID,
                parent_uuid=0,
                name=TRACE_ROOT_TRACK_NAME,
                sibling_order_rank=0,
            )
        ]
        parent_uuid = TRACE_ROOT_TRACK_UUID
        for depth in range(1, len(self.groups) + 1):
            group_path = self.groups[:depth]
            group_name = group_path[-1]
            group_uuid = _group_uuid(group_path)
            nodes.append(
                TraceTrackNode(
                    uuid=group_uuid,
                    parent_uuid=parent_uuid,
                    name=group_name,
                    sibling_order_rank=_order_rank(group_path),
                )
            )
            parent_uuid = group_uuid
        nodes.append(
            TraceTrackNode(
                uuid=leaf_uuid,
                parent_uuid=parent_uuid,
                name=self.leaf_name,
                sibling_order_rank=_leaf_order_rank(self.leaf_name),
            )
        )
        return TraceTrackLayout(nodes=tuple(nodes), leaf_uuid=leaf_uuid)


def _group_uuid(group_path: tuple[str, ...]) -> int:
    return stable_u64("fluxway-profiler", "track-group", *group_path)


def _order_rank(group_path: tuple[str, ...]) -> int:
    name = group_path[-1]
    if name.startswith("rank:"):
        return 1000 + int(name.removeprefix("rank:"))
    return _ROLE_ORDER.get(name, _SUBSYSTEM_ORDER.get(name, 500))


def _leaf_order_rank(leaf_name: str) -> int:
    first = leaf_name.split("/", 1)[0]
    return _SUBSYSTEM_ORDER.get(first, 500)
