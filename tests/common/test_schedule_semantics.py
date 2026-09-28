import pytest

from fluxway.common.operator_policy import SchedulePolicy
from fluxway.common.schedule_semantics import (
    CompiledPhaseSchedule,
    PhaseGate,
    compile_phase_schedule,
)


PHASES = ("prepare", "train", "update")


def _phase_order_policies():
    return {
        phase: SchedulePolicy.phase_order(phase=phase, phase_order=PHASES)
        for phase in PHASES
    }


def test_compile_phase_order_schedule_exposes_runtime_gates():
    schedule = compile_phase_schedule(_phase_order_policies())

    assert schedule.gates["prepare"].predecessors == ()
    assert schedule.gates["train"].predecessors == (
        ("prepare", 0),
        ("train", 1),
        ("update", 1),
    )
    assert schedule.gates["update"].predecessors == (
        ("prepare", 0),
        ("train", 0),
        ("update", 1),
    )
    assert schedule.relations == (
        ("prepare", "train", 0),
        ("prepare", "update", 0),
        ("train", "train", 1),
        ("train", "update", 0),
        ("update", "train", 1),
        ("update", "update", 1),
    )


def test_compile_phase_graph_schedule_preserves_explicit_dependencies():
    dependencies = (
        ("prepare", "train", 0),
        ("train", "update", 0),
        ("update", "train", 1),
    )
    policies = {
        phase: SchedulePolicy.phase_graph(
            phase=phase,
            phase_order=PHASES,
            dependencies=dependencies,
        )
        for phase in PHASES
    }

    schedule = compile_phase_schedule(policies)

    assert schedule.gates["prepare"].predecessors == ()
    assert schedule.gates["train"].predecessors == (
        ("prepare", 0),
        ("update", 1),
    )
    assert schedule.gates["update"].predecessors == (("train", 0),)
    assert schedule.relations == dependencies


def test_phase_gate_matches_completed_step_counter_semantics():
    gate = compile_phase_schedule(_phase_order_policies()).gates["train"]

    assert gate.is_blocked(
        {"prepare": 2, "train": 1, "update": 0},
        step=1,
    )
    assert not gate.is_blocked(
        {"prepare": 2, "train": 1, "update": 1},
        step=1,
    )


def test_compile_phase_schedule_ignores_non_phase_policies():
    schedule = compile_phase_schedule({"read": SchedulePolicy.fifo()})

    assert schedule.gates == {}
    assert schedule.relations == ()


def test_compiled_phase_relations_are_derived_from_runtime_gates():
    schedule = CompiledPhaseSchedule.from_mapping(
        {"after": PhaseGate(predecessors=(("before", 0),))}
    )

    assert schedule.relations == (("before", "after", 0),)


def test_compile_phase_schedule_rejects_missing_phase_operator():
    policies = _phase_order_policies()
    del policies["update"]

    with pytest.raises(ValueError, match="missing operators for phases.*update"):
        compile_phase_schedule(policies)


def test_compile_phase_schedule_rejects_duplicate_phase_operator():
    policies = _phase_order_policies()
    policies["duplicate"] = SchedulePolicy.phase_order(
        phase="train",
        phase_order=PHASES,
    )

    with pytest.raises(ValueError, match="assigned to multiple operators"):
        compile_phase_schedule(policies)
