from dataclasses import dataclass
import pytest

from fluxway.common.decorators import operator, data_type
from fluxway.common.types import AsyncPolicy, SchedulePolicy, StatePolicy, BarrierType
from fluxway.common.types.tensor import TransportTensor


@data_type()
@dataclass
class RaggedInput:
    feature1: TransportTensor
    feature2: TransportTensor


@data_type()
@dataclass
class OtherInput:
    other_tensor: TransportTensor


class MockSPMD:
    @operator(
        name="test_uniform", input_sharding={"data": {"type": "static", "axis": ["dp"]}}
    )
    def test_uniform(self, data: RaggedInput):
        pass

    @operator(
        name="test_nested",
        input_sharding={
            "data": {
                "feature1": {"type": "static", "axis": ["dp"]},
                "feature2": {"type": "dynamic", "axis": ["tp"]},
            }
        },
    )
    def test_nested(self, data: RaggedInput):
        pass

    @operator(
        name="test_mixed",
        input_sharding={
            "data": {"type": "static", "axis": ["dp"]},
            "other": {"type": "static", "axis": ["mp"]},
        },
    )
    def test_mixed(self, data: RaggedInput, other: OtherInput):
        pass

    @operator(
        name="test_output",
        output_sharding={"return": {"type": "static", "axis": ["dp"]}},
    )
    def test_output(self) -> RaggedInput:  # type: ignore
        pass

    @operator(
        name="test_relaxed",
        local_pipeline_depth=8,
        commit_policy="relaxed",
        state_access="thread_safe",
    )
    def test_relaxed(self):
        pass


class AsyncPolicySPMD:
    @operator(name="sync_default", async_policy=AsyncPolicy.sync())
    def sync_default(self):
        pass

    @operator(
        name="exact_pipe",
        async_policy=AsyncPolicy.pipelined(
            depth=4,
            commit="ordered",
            schedule=SchedulePolicy.fifo(base_priority=3),
            state=StatePolicy.exclusive("dense"),
        ),
    )
    def exact_pipe(self):
        pass

    @operator(
        name="stale_window",
        data_argument=["ids"],
        async_policy=AsyncPolicy.bounded_stale(
            depth=4,
            max_staleness=3,
            schedule=SchedulePolicy.deadline_boost(boost_priority=150),
            state=StatePolicy.thread_safe(),
        ),
    )
    def stale_window(self, ids: TransportTensor):
        pass


def test_sharding_resolution():
    meta_uniform = getattr(MockSPMD.test_uniform, "_fluxway_operator_meta", None)
    assert meta_uniform is not None
    # Single parameter RaggedInput: fields should NOT be prefixed (use_prefix=False)
    assert meta_uniform.input_sharding["feature1"]["axis"] == ["dp"]
    assert meta_uniform.input_sharding["feature2"]["axis"] == ["dp"]
    assert "data" in meta_uniform.input_param_map
    assert "feature1" in meta_uniform.input_param_map["data"]

    meta_nested = getattr(MockSPMD.test_nested, "_fluxway_operator_meta", None)
    assert meta_nested is not None
    assert meta_nested.input_sharding["feature1"]["axis"] == ["dp"]
    assert meta_nested.input_sharding["feature2"]["axis"] == ["tp"]

    meta_mixed = getattr(MockSPMD.test_mixed, "_fluxway_operator_meta", None)
    assert meta_mixed is not None
    # Multiple parameters: fields ARE prefixed with parameter names
    assert meta_mixed.input_sharding["data.feature1"]["axis"] == ["dp"]
    assert meta_mixed.input_sharding["other.other_tensor"]["axis"] == ["mp"]

    # Test Output Sharding & Return-Alias
    meta_output = getattr(MockSPMD.test_output, "_fluxway_operator_meta", None)
    assert meta_output is not None
    # "return" alias should have been expanded to individual fields
    assert "feature1" in meta_output.output_sharding
    assert "feature2" in meta_output.output_sharding
    assert meta_output.output_sharding["feature1"]["axis"] == ["dp"]
    assert "return" in meta_output.output_param_map
    assert "feature1" in meta_output.output_param_map["return"]


def test_operator_local_pipeline_metadata():
    meta_relaxed = getattr(MockSPMD.test_relaxed, "_fluxway_operator_meta", None)
    assert meta_relaxed is not None
    assert meta_relaxed.local_pipeline_depth == 8
    assert meta_relaxed.commit_policy == "relaxed"
    assert meta_relaxed.state_access == "thread_safe"
    assert meta_relaxed.state_resource is None


def test_operator_async_policy_lowering():
    meta_sync = getattr(AsyncPolicySPMD.sync_default, "_fluxway_operator_meta", None)
    assert meta_sync is not None
    assert meta_sync.async_policy.mode == "sync"
    assert meta_sync.input_policy == "exact"
    assert meta_sync.local_pipeline_depth == 1
    assert meta_sync.commit_policy == "ordered"
    assert meta_sync.state_access == "exclusive"
    assert meta_sync.schedule_policy.kind == "fifo"

    meta_pipe = getattr(AsyncPolicySPMD.exact_pipe, "_fluxway_operator_meta", None)
    assert meta_pipe is not None
    assert meta_pipe.async_policy.mode == "pipelined"
    assert meta_pipe.local_pipeline_depth == 4
    assert meta_pipe.priority == 3
    assert meta_pipe.state_access == "exclusive"
    assert meta_pipe.state_resource == "dense"
    assert meta_pipe.max_async_lag == 3
    assert meta_pipe.max_inflight_steps == 3

    meta_window = getattr(AsyncPolicySPMD.stale_window, "_fluxway_operator_meta", None)
    assert meta_window is not None
    assert meta_window.async_policy.mode == "bounded_stale"
    assert meta_window.input_policy == "exact"
    assert meta_window.local_pipeline_depth == 4
    assert meta_window.max_inflight_steps == 3
    assert meta_window.max_async_lag == 3
    assert meta_window.commit_policy == "relaxed"
    assert meta_window.state_access == "thread_safe"
    assert meta_window.schedule_policy.kind == "deadline_boost"
    assert meta_window.schedule_policy.boost_priority == 150
    assert meta_window.barrier_config.type is BarrierType.STANDARD


def test_operator_rejects_invalid_local_pipeline_options():
    with pytest.raises(ValueError, match="local_pipeline_depth"):

        @operator(name="bad_depth", local_pipeline_depth=0)
        def bad_depth(self):
            pass

    with pytest.raises(ValueError, match="commit_policy"):

        @operator(name="bad_policy", commit_policy="free")
        def bad_policy(self):
            pass

    with pytest.raises(ValueError, match="state_access"):

        @operator(name="bad_state", state_access="shared")
        def bad_state(self):
            pass

    with pytest.raises(ValueError, match="import-stable class methods"):

        @operator(name="bad_free_thread_safe", state_access="thread_safe")
        def bad_free_thread_safe():
            pass

    with pytest.raises(ValueError, match="cannot be combined"):

        class BadResource:
            @operator(
                name="bad_resource",
                state_access="thread_safe",
                state_resource="table",
            )
            def bad_resource(self):
                pass

    with pytest.raises(ValueError, match="legacy async arguments"):

        class BadMixedPolicy:
            @operator(
                name="bad_mixed_policy",
                local_pipeline_depth=2,
                async_policy=AsyncPolicy.pipelined(depth=2),
            )
            def bad_mixed_policy(self):
                pass

    with pytest.raises(ValueError, match="depth <= max_staleness"):
        AsyncPolicy.bounded_stale(depth=4, max_staleness=2)

    with pytest.raises(ValueError, match="max_staleness"):
        AsyncPolicy.bounded_stale(depth=1, max_staleness={"ids": 0})  # type: ignore[arg-type]


if __name__ == "__main__":
    test_sharding_resolution()
    print("All operator interface tests passed!")
