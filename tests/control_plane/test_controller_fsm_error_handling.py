"""
Unit tests for ControllerStateMachine error handling and fail-fast logic.

Covers:
- Bug fix: restart_spmd failure triggers RECOVERY_FAILED → FAILING → FAILED
- fail_fast=True: skip recovery, go straight to FAILING
- fail_fast=False: normal recovery path
- BLOCKED state timeout: auto-trigger RECOVERY_FAILED
- BLOCKED timeout cancellation on state exit
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from fluxway.common.types import ControllerState, ControllerEvent
from fluxway.common.types.configs import (
    SPMDConfig,
    Graph,
    Node,
    HeartbeatConfig,
    ErrorPolicy,
)
from fluxway.control_plane.state_machine.controller_fsm import ControllerStateMachine


def _make_controller(
    *,
    fail_fast: bool = True,
    restart_raises: bool = False,
    restart_returns: bool = True,
) -> MagicMock:
    node = Node(
        name="dense_spmd",
        spmd_config=SPMDConfig(spmd_id="dense_spmd", fail_fast=fail_fast),
    )
    graph = Graph(nodes={"dense_spmd": node})

    ctrl = MagicMock()
    ctrl._graph = graph
    ctrl.block_all_spmds = AsyncMock()
    ctrl._cleanup_all_spmds = AsyncMock()
    ctrl._release_stopped_spmd_handles = MagicMock()

    if restart_raises:
        ctrl.restart_spmd = AsyncMock(side_effect=RuntimeError("restart failed"))
    else:
        ctrl.restart_spmd = AsyncMock(return_value=restart_returns)

    return ctrl


def _make_fsm(
    ctrl: MagicMock,
    blocked_timeout: float = 999.0,
    error_policy: ErrorPolicy | None = None,
) -> ControllerStateMachine:
    return ControllerStateMachine(
        ctrl, blocked_timeout=blocked_timeout, error_policy=error_policy
    )


async def _advance_to_running(fsm: ControllerStateMachine) -> None:
    fsm._state = ControllerState.RUNNING


class TestRecoveryFailedBugFix:
    """restart_spmd failure must trigger RECOVERY_FAILED, not silently stay BLOCKED."""

    @pytest.mark.asyncio
    async def test_restart_exception_triggers_recovery_failed(self):
        ctrl = _make_controller(fail_fast=False, restart_raises=True)
        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="OOM"
        )

        assert fsm.state == ControllerState.FAILED
        ctrl.block_all_spmds.assert_called_once()
        ctrl.restart_spmd.assert_called_once_with("dense_spmd")
        ctrl._cleanup_all_spmds.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_returns_false_triggers_recovery_failed(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=False)
        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="node down"
        )

        assert fsm.state == ControllerState.FAILED
        ctrl.restart_spmd.assert_called_once()
        ctrl._cleanup_all_spmds.assert_called_once()


class TestFailFast:
    """fail_fast=True on error node must skip recovery entirely."""

    @pytest.mark.asyncio
    async def test_fail_fast_skips_recovery(self):
        ctrl = _make_controller(fail_fast=True)
        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="GPU fault"
        )

        assert fsm.state == ControllerState.FAILED
        ctrl.restart_spmd.assert_not_called()
        ctrl.block_all_spmds.assert_called_once()
        ctrl._cleanup_all_spmds.assert_called_once()

    @pytest.mark.asyncio
    async def test_fail_fast_false_attempts_recovery(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=True)
        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="temp error"
        )
        # restart_spmd succeeded -> stays BLOCKED waiting for DRIVER_RECOVERED
        assert fsm.state == ControllerState.BLOCKED
        ctrl.restart_spmd.assert_called_once_with("dense_spmd")

    @pytest.mark.asyncio
    async def test_fail_fast_false_recovery_success_returns_to_running(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=True)
        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="temp"
        )
        assert fsm.state == ControllerState.BLOCKED

        await fsm.trigger_event(ControllerEvent.DRIVER_RECOVERED, spmd_id="dense_spmd")
        assert fsm.state == ControllerState.RUNNING


class TestFailFastMultiNode:
    """Multi-node scenarios: one fail_fast node and one non-fail-fast node."""

    @pytest.mark.asyncio
    async def test_mixed_nodes_fail_fast_wins(self):
        reader_node = Node(
            name="reader",
            spmd_config=SPMDConfig(spmd_id="reader", fail_fast=False),
        )
        dense_node = Node(
            name="dense_spmd",
            spmd_config=SPMDConfig(spmd_id="dense_spmd", fail_fast=True),
        )
        graph = Graph(nodes={"reader": reader_node, "dense_spmd": dense_node})

        ctrl = MagicMock()
        ctrl._graph = graph
        ctrl.block_all_spmds = AsyncMock()
        ctrl._cleanup_all_spmds = AsyncMock()
        ctrl._release_stopped_spmd_handles = MagicMock()
        ctrl.restart_spmd = AsyncMock(return_value=True)

        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR,
            spmd_id="dense_spmd",
            error="GPU fault",
        )

        assert fsm.state == ControllerState.FAILED
        ctrl.restart_spmd.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_fail_fast_node_error_attempts_recovery(self):
        reader_node = Node(
            name="reader",
            spmd_config=SPMDConfig(spmd_id="reader", fail_fast=False),
        )
        graph = Graph(nodes={"reader": reader_node})

        ctrl = MagicMock()
        ctrl._graph = graph
        ctrl.block_all_spmds = AsyncMock()
        ctrl._cleanup_all_spmds = AsyncMock()
        ctrl._release_stopped_spmd_handles = MagicMock()
        ctrl.restart_spmd = AsyncMock(return_value=True)

        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="reader", error="I/O timeout"
        )

        assert fsm.state == ControllerState.BLOCKED
        ctrl.restart_spmd.assert_called_once_with("reader")


class TestBlockedTimeout:
    """BLOCKED state timeout must auto-trigger RECOVERY_FAILED."""

    @pytest.mark.asyncio
    async def test_blocked_timeout_triggers_recovery_failed(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=True)
        fsm = _make_fsm(ctrl, blocked_timeout=0.1)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="temp error"
        )
        assert fsm.state == ControllerState.BLOCKED

        await asyncio.sleep(0.25)

        assert fsm.state == ControllerState.FAILED
        ctrl._cleanup_all_spmds.assert_called_once()

    @pytest.mark.asyncio
    async def test_blocked_timeout_cancelled_on_recovery(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=True)
        fsm = _make_fsm(ctrl, blocked_timeout=0.1)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="temp"
        )
        assert fsm.state == ControllerState.BLOCKED

        await fsm.trigger_event(ControllerEvent.DRIVER_RECOVERED, spmd_id="dense_spmd")
        assert fsm.state == ControllerState.RUNNING

        # Wait past original timeout — should NOT trigger RECOVERY_FAILED
        await asyncio.sleep(0.2)
        assert fsm.state == ControllerState.RUNNING

    @pytest.mark.asyncio
    async def test_blocked_timeout_cancelled_on_reset(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=True)
        fsm = _make_fsm(ctrl, blocked_timeout=0.1)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="temp"
        )
        assert fsm.state == ControllerState.BLOCKED

        await fsm.reset()
        assert fsm.state == ControllerState.INITIALIZING

        await asyncio.sleep(0.2)
        assert fsm.state == ControllerState.INITIALIZING


class TestBlockedTimeoutConfig:
    """blocked_timeout is correctly wired from HeartbeatConfig."""

    def test_default_blocked_timeout(self):
        config = HeartbeatConfig()
        assert config.blocked_timeout == 300.0

    def test_custom_blocked_timeout(self):
        config = HeartbeatConfig(blocked_timeout=60.0)
        assert config.blocked_timeout == 60.0

    def test_fsm_receives_blocked_timeout(self):
        ctrl = _make_controller()
        fsm = ControllerStateMachine(ctrl, blocked_timeout=42.0)
        assert fsm._blocked_timeout_secs == 42.0


class TestFailFastConfig:
    """fail_fast config defaults and overrides."""

    def test_default_fail_fast_is_true(self):
        config = SPMDConfig()
        assert config.fail_fast is True

    def test_fail_fast_override_false(self):
        config = SPMDConfig(fail_fast=False)
        assert config.fail_fast is False


class TestErrorPolicyConfig:
    """ErrorPolicy config defaults and overrides."""

    def test_default_empty_patterns(self):
        policy = ErrorPolicy()
        assert policy.recoverable_patterns == []

    def test_custom_patterns(self):
        policy = ErrorPolicy(recoverable_patterns=["timeout", "connection reset"])
        assert len(policy.recoverable_patterns) == 2

    def test_fsm_receives_error_policy(self):
        ctrl = _make_controller()
        policy = ErrorPolicy(recoverable_patterns=["test"])
        fsm = ControllerStateMachine(ctrl, error_policy=policy)
        assert fsm._error_policy is policy

    def test_fsm_default_error_policy(self):
        ctrl = _make_controller()
        fsm = ControllerStateMachine(ctrl)
        assert fsm._error_policy.recoverable_patterns == []


class TestShouldFailFast:
    """Unit tests for _should_fail_fast decision logic."""

    def test_no_patterns_falls_back_to_fail_fast_true(self):
        ctrl = _make_controller(fail_fast=True)
        fsm = _make_fsm(ctrl)
        assert fsm._should_fail_fast("dense_spmd", "OOM") is True

    def test_no_patterns_falls_back_to_fail_fast_false(self):
        ctrl = _make_controller(fail_fast=False)
        fsm = _make_fsm(ctrl)
        assert fsm._should_fail_fast("dense_spmd", "OOM") is False

    def test_matching_pattern_overrides_fail_fast_true(self):
        ctrl = _make_controller(fail_fast=True)
        policy = ErrorPolicy(recoverable_patterns=["timeout"])
        fsm = _make_fsm(ctrl, error_policy=policy)
        assert fsm._should_fail_fast("dense_spmd", "I/O timeout") is False

    def test_non_matching_pattern_uses_fail_fast(self):
        ctrl = _make_controller(fail_fast=True)
        policy = ErrorPolicy(recoverable_patterns=["timeout"])
        fsm = _make_fsm(ctrl, error_policy=policy)
        assert fsm._should_fail_fast("dense_spmd", "OOM") is True

    def test_regex_pattern_matching(self):
        ctrl = _make_controller(fail_fast=True)
        policy = ErrorPolicy(recoverable_patterns=[r"timeout|retry"])
        fsm = _make_fsm(ctrl, error_policy=policy)
        assert fsm._should_fail_fast("dense_spmd", "connection timeout") is False
        assert fsm._should_fail_fast("dense_spmd", "please retry") is False
        assert fsm._should_fail_fast("dense_spmd", "OOM") is True

    def test_unknown_node_defaults_to_fail_fast(self):
        ctrl = _make_controller(fail_fast=True)
        fsm = _make_fsm(ctrl)
        assert fsm._should_fail_fast("nonexistent_spmd", "error") is True

    def test_multiple_patterns_any_match_recoverable(self):
        ctrl = _make_controller(fail_fast=True)
        policy = ErrorPolicy(
            recoverable_patterns=["connection reset", r"I/O\s+timeout"]
        )
        fsm = _make_fsm(ctrl, error_policy=policy)
        assert fsm._should_fail_fast("dense_spmd", "connection reset by peer") is False
        assert fsm._should_fail_fast("dense_spmd", "I/O timeout occurred") is False
        assert fsm._should_fail_fast("dense_spmd", "GPU fault") is True


class TestErrorPolicyIntegration:
    """Integration tests for ErrorPolicy with full FSM transitions."""

    @pytest.mark.asyncio
    async def test_recoverable_pattern_overrides_fail_fast_allows_recovery(self):
        ctrl = _make_controller(fail_fast=True, restart_returns=True)
        policy = ErrorPolicy(recoverable_patterns=["timeout"])
        fsm = _make_fsm(ctrl, error_policy=policy)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR,
            spmd_id="dense_spmd",
            error="connection timeout",
        )

        # Despite fail_fast=True, the error matches a recoverable pattern
        assert fsm.state == ControllerState.BLOCKED
        ctrl.restart_spmd.assert_called_once_with("dense_spmd")


class TestRepeatedDriverError:
    """BLOCKED + DRIVER_ERROR → stays BLOCKED (accumulates errors)."""

    @pytest.mark.asyncio
    async def test_second_driver_error_stays_blocked(self):
        reader_node = Node(
            name="reader",
            spmd_config=SPMDConfig(spmd_id="reader", fail_fast=False),
        )
        dense_node = Node(
            name="dense",
            spmd_config=SPMDConfig(spmd_id="dense", fail_fast=False),
        )
        graph = Graph(nodes={"reader": reader_node, "dense": dense_node})

        ctrl = MagicMock()
        ctrl._graph = graph
        ctrl.block_all_spmds = AsyncMock()
        ctrl._cleanup_all_spmds = AsyncMock()
        ctrl._release_stopped_spmd_handles = MagicMock()
        ctrl.restart_spmd = AsyncMock(return_value=True)

        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="reader", error="err1"
        )
        assert fsm.state == ControllerState.BLOCKED

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense", error="err2"
        )
        assert fsm.state == ControllerState.BLOCKED
        assert "reader" in fsm._error_spmds
        assert "dense" in fsm._error_spmds

    @pytest.mark.asyncio
    async def test_partial_recovery_stays_blocked(self):
        reader_node = Node(
            name="reader",
            spmd_config=SPMDConfig(spmd_id="reader", fail_fast=False),
        )
        dense_node = Node(
            name="dense",
            spmd_config=SPMDConfig(spmd_id="dense", fail_fast=False),
        )
        graph = Graph(nodes={"reader": reader_node, "dense": dense_node})

        ctrl = MagicMock()
        ctrl._graph = graph
        ctrl.block_all_spmds = AsyncMock()
        ctrl._cleanup_all_spmds = AsyncMock()
        ctrl._release_stopped_spmd_handles = MagicMock()
        ctrl.restart_spmd = AsyncMock(return_value=True)

        fsm = _make_fsm(ctrl)
        await _advance_to_running(fsm)

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="reader", error="err1"
        )
        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense", error="err2"
        )

        # Only one recovers — should stay BLOCKED
        await fsm.trigger_event(ControllerEvent.DRIVER_RECOVERED, spmd_id="reader")
        assert fsm.state == ControllerState.BLOCKED
        assert "reader" not in fsm._error_spmds
        assert "dense" in fsm._error_spmds


class TestInitPhaseErrors:
    """DRIVER_ERROR during PENDING or TOPOLOGY_REGISTERING phases."""

    @pytest.mark.asyncio
    async def test_pending_driver_error_goes_to_blocked(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=True)
        fsm = _make_fsm(ctrl)
        fsm._state = ControllerState.PENDING

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="init fail"
        )
        assert fsm.state == ControllerState.BLOCKED

    @pytest.mark.asyncio
    async def test_topology_registering_error_goes_to_blocked(self):
        ctrl = _make_controller(fail_fast=False, restart_returns=True)
        fsm = _make_fsm(ctrl)
        fsm._state = ControllerState.TOPOLOGY_REGISTERING

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="topo fail"
        )
        assert fsm.state == ControllerState.BLOCKED

    @pytest.mark.asyncio
    async def test_pending_error_with_fail_fast_goes_to_failed(self):
        ctrl = _make_controller(fail_fast=True)
        fsm = _make_fsm(ctrl)
        fsm._state = ControllerState.PENDING

        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="dense_spmd", error="fatal"
        )
        assert fsm.state == ControllerState.FAILED
