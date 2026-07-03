import inspect

from fluxway.spmd.worker.user_callable_executor import (
    UserCallableExecutor,
    UserCallableLane,
    UserCallableSpec,
    _RayOperatorExecutionActor,
)


def test_ray_operator_execution_actor_accepts_rank0_address_context():
    assert "rank0_address" in UserCallableSpec.__dataclass_fields__

    actor_cls = _RayOperatorExecutionActor.__ray_metadata__.modified_class  # type: ignore
    params = inspect.signature(actor_cls.__init__).parameters
    assert "rank0_address" in params


def test_process_affine_stateful_operator_uses_worker_process_lane():
    assert "process_affine" in UserCallableSpec.__dataclass_fields__

    def fn():
        return None

    spec = UserCallableSpec(
        callable_id="step",
        fn=fn,
        is_stateful=True,
        resource_id=0,
        operator_name="step",
        process_affine=True,
    )

    executor = UserCallableExecutor()
    assert executor._lane_for(spec) is UserCallableLane.WORKER_PROCESS


def test_stateful_ops_with_same_owner_share_ray_actor_key():
    class Table:
        def lookup(self):
            return None

        def update(self):
            return None

    executor = UserCallableExecutor()
    owner = (Table.__module__, Table.__qualname__, Table)
    lookup = UserCallableSpec(
        callable_id="lookup",
        fn=lambda: None,
        is_stateful=True,
        resource_id=0,
        operator_name="lookup",
        operator_spmd_id="Embedding",
        operator_owner_classes=(owner,),
    )
    update = UserCallableSpec(
        callable_id="update",
        fn=lambda: None,
        is_stateful=True,
        resource_id=0,
        operator_name="update",
        operator_spmd_id="Embedding",
        operator_owner_classes=(owner,),
    )

    assert executor._ray_actor_key(lookup) == executor._ray_actor_key(update)
