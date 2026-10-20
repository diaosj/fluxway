from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy
from fluxway import OperatorSpec
from fluxway.spmd.worker.worker import SPMDWorker


PHASES = (
    "prepare",
    "train",
    "scatter",
)

DEPS = (
    ("prepare", "train", 0),
    ("train", "scatter", 0),
)


class DummyManager:
    def __init__(self, metas):
        self._metas = metas

    def _get_operator_spec(self, op_name):
        return self._metas[op_name]


def worker_with(metas):
    worker_cls = (
        SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    )  # pyrefly: ignore [missing-attribute]
    worker = object.__new__(worker_cls)
    worker._op_manager = DummyManager(metas)
    worker._input_prepare_runtime = None
    return worker


def policy_for(phase, **kwargs):
    return SchedulePolicy.phase_graph(
        phase=phase,
        phase_order=PHASES,
        dependencies=DEPS,
        **kwargs,
    )


def meta_for(phase, policy):
    return OperatorSpec(
        name=phase,
        async_policy=AsyncPolicy.pipelined(depth=2, schedule=policy),
    )


def test_default_phase_graph_retentions_step_major_priority():
    metas = {phase: meta_for(phase, policy_for(phase)) for phase in PHASES}
    worker = worker_with(metas)

    assert worker._effective_schedule_key(
        "scatter", 0
    ) < worker._effective_schedule_key("prepare", 1)


def test_lane_aware_phase_graph_prioritizes_compute_and_next_prepare():
    priorities = {"prepare": 300, "train": 500, "scatter": 20}
    metas = {
        "prepare": meta_for(
            "prepare",
            policy_for(
                "prepare",
                lane="data",
                lane_order=("compute", "data"),
                lane_priorities={"compute": 1000, "data": 100},
                phase_priorities=priorities,
            ),
        ),
        "train": meta_for(
            "train",
            policy_for(
                "train",
                lane="compute",
                lane_order=("compute", "data"),
                lane_priorities={"compute": 1000, "data": 100},
                phase_priorities=priorities,
            ),
        ),
        "scatter": meta_for(
            "scatter",
            policy_for(
                "scatter",
                lane="data",
                lane_order=("compute", "data"),
                lane_priorities={"compute": 1000, "data": 100},
                phase_priorities=priorities,
            ),
        ),
    }
    worker = worker_with(metas)

    train_current = worker._effective_schedule_key("train", 0)
    prepare_next = worker._effective_schedule_key("prepare", 1)
    scatter_current = worker._effective_schedule_key("scatter", 0)

    assert train_current < prepare_next < scatter_current
