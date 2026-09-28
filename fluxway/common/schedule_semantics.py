from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.operator_policy import SchedulePolicy


type PhasePredecessor = tuple[str, int]
type PhaseRelation = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class PhaseGate:
    predecessors: tuple[PhasePredecessor, ...]

    def is_blocked(self, completed_steps: Mapping[str, int], *, step: int) -> bool:
        for predecessor, step_delta in self.predecessors:
            required_step = step - step_delta
            if required_step < 0:
                continue
            if completed_steps[predecessor] <= required_step:
                return True
        return False


@dataclass(frozen=True, slots=True)
class CompiledPhaseSchedule:
    gates: Mapping[str, PhaseGate]

    def __post_init__(self) -> None:
        if not isinstance(self.gates, Mapping):
            raise TypeError("compiled phase schedule gates must be a mapping")
        if self.gates.__class__.__name__ != "FrozenMap":
            raise TypeError("compiled phase schedule gates must be frozen")

    @classmethod
    def from_mapping(cls, gates: Mapping[str, PhaseGate]) -> CompiledPhaseSchedule:
        return cls(gates=freeze_mapping(gates))

    @property
    def relations(self) -> tuple[PhaseRelation, ...]:
        return tuple(
            sorted(
                (predecessor, operator_id, step_delta)
                for operator_id, gate in self.gates.items()
                for predecessor, step_delta in gate.predecessors
            )
        )


def compile_phase_schedule(
    operator_policies: Mapping[str, SchedulePolicy],
) -> CompiledPhaseSchedule:
    groups: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}
    for operator_id in sorted(operator_policies):
        policy = operator_policies[operator_id]
        if policy.kind not in ("phase_order", "phase_graph"):
            continue
        group_key = (policy.kind, tuple(policy.phase_ordering))
        phase = policy.phase
        if phase is None:
            raise ValueError(f"phase schedule operator {operator_id!r} has no phase.")
        phase_to_operator = groups.setdefault(group_key, {})
        existing = phase_to_operator.get(phase)
        if existing is not None:
            raise ValueError(
                f"{policy.kind} phase {phase!r} is assigned to multiple operators: "
                f"{existing!r} and {operator_id!r}."
            )
        phase_to_operator[phase] = operator_id

    gates = {}
    for (kind, phase_order), phase_to_operator in sorted(groups.items()):
        missing = [phase for phase in phase_order if phase not in phase_to_operator]
        if missing:
            raise ValueError(
                f"{kind} schedule missing operators for phases {missing!r}."
            )
        for phase_index, phase in enumerate(phase_order):
            operator_id = phase_to_operator[phase]
            policy = operator_policies[operator_id]
            phase_relations = (
                _phase_order_relations(phase_order, phase_index)
                if kind == "phase_order"
                else tuple(
                    (before, step_delta)
                    for before, after, step_delta in policy.phase_dependencies
                    if after == phase
                )
            )
            predecessors = tuple(
                sorted(
                    (phase_to_operator[before], step_delta)
                    for before, step_delta in phase_relations
                )
            )
            gates[operator_id] = PhaseGate(
                predecessors=predecessors,
            )
    return CompiledPhaseSchedule.from_mapping(gates)


def _phase_order_relations(
    phase_order: tuple[str, ...],
    phase_index: int,
) -> tuple[tuple[str, int], ...]:
    predecessors = [(phase, 0) for phase in phase_order[:phase_index]]
    if phase_index > 0:
        predecessors.extend((phase, 1) for phase in phase_order[phase_index:])
    return tuple(predecessors)
