"""User callable execution lanes for worker-local DAG nodes."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import importlib
import inspect
import statistics
import sys
import sysconfig
import time
from typing import Any

import ray

from fluxway.common.types.manifest import ManifestOperatorFacts
from fluxway.profiler.runtime import ProfilerRuntimeFactory
from fluxway.profiler.user_api import activate_user_profiler
from fluxway.spmd.worker.runtime import (
    WORKER_PROCESS_DEFAULT_EXECUTION_LANE,
    WorkerProcessCallRuntime,
)


USER_CALLABLE_DEFAULT_OPERATOR_LANE = WORKER_PROCESS_DEFAULT_EXECUTION_LANE


class UserCallableLane(str, Enum):
    INLINE_ASYNC = "inline_async"
    INLINE_SYNC = "inline_sync"
    WORKER_PROCESS = "worker_process"
    LOCAL_FREE_THREAD = "local_free_thread"
    RAY_EXECUTION_ACTOR = "ray_execution_actor"
    NATIVE_BACKEND = "native_backend"


@dataclass(frozen=True, slots=True)
class UserCallableSpec:
    callable_id: str
    fn: Callable[..., Any]
    is_stateful: bool
    resource_id: int
    operator_lane: str
    save_state: Callable[[str], Any] | None = None
    restore_state: Callable[[Any, str], None] | None = None
    operator_config: Any | None = None
    operator_name: str | None = None
    operator_step: int | None = None
    operator_args: tuple[Any, ...] = ()
    operator_kwargs: dict[str, Any] | None = None
    operator_rank: int | None = None
    operator_spmd_id: str = ""
    local_rank: int | None = None
    local_world_size: int | None = None
    node_rank: int = 0
    num_nodes: int = 1
    rank0_address: str | None = None
    operator_owner_classes: tuple[tuple[str, str, type], ...] = ()
    process_affine: bool = False
    operator_facts: ManifestOperatorFacts | None = None


@dataclass(slots=True)
class _CallableProfile:
    samples: list[float]
    lane: UserCallableLane | None = None


@dataclass(frozen=True, slots=True)
class _ActorRecord:
    actor: Any
    callable_id: str
    operator_name: str | None
    operator_names: set[str | None] = field(default_factory=set)


def python_free_threading_enabled() -> bool:
    if not bool(sysconfig.get_config_var("Py_GIL_DISABLED")):
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None:
        return False
    return not bool(is_gil_enabled())


def _call_state_hook(instance: Any, hook_name: str, *args: Any) -> Any:
    hook = getattr(instance, hook_name, None)
    if not callable(hook):
        return None
    signature = inspect.signature(hook)
    if hook_name == "save_state":
        if "ckpt_dir" in signature.parameters:
            return hook(ckpt_dir=args[0])
        return hook()
    if hook_name == "restore_state":
        if "ckpt_dir" in signature.parameters:
            hook(args[0], ckpt_dir=args[1])
        else:
            hook(args[0])
        return None
    return hook(*args)


def _register_owner_class(module_name: str, class_qualname: str, cls: type) -> None:
    if "<locals>" in class_qualname:
        raise ValueError(
            f"operator owner class {module_name}.{class_qualname} is not import-stable."
        )
    module = importlib.import_module(module_name)
    parts = class_qualname.split(".")
    parent: Any = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], cls)


@ray.remote(max_concurrency=1)  # type: ignore
class _RayFunctionExecutionActor:
    def __init__(
        self,
        fn: Callable[..., Any],
        save_state: Callable[[str], Any] | None,
        restore_state: Callable[[Any, str], None] | None,
        profiler_config: Any | None,
        profiler_process_name: str,
    ) -> None:
        self._fn = fn
        self._save_state = save_state
        self._restore_state = restore_state
        self._profiler_runtime = ProfilerRuntimeFactory.from_config(
            profiler_config,
            process_name=profiler_process_name,
        )

    async def execute(self, op_name: str, step: int, rank: int) -> Any:
        if self._profiler_runtime.enabled:
            with activate_user_profiler(
                recorder=self._profiler_runtime.recorder,
                op_name=op_name,
                step=step,
                rank=rank,
                execution_mode=UserCallableLane.RAY_EXECUTION_ACTOR.value,
            ):
                return await self._execute_inner()
        return await self._execute_inner()

    async def _execute_inner(self) -> Any:
        result = self._fn()
        if inspect.isawaitable(result):
            result = await result
        return result

    async def export_profiler_snapshot(self) -> Any:
        return self._profiler_runtime.snapshot()

    async def shutdown(self) -> None:
        instance = self._bound_instance()
        if instance is None:
            return
        result = _call_state_hook(instance, "on_shutdown")
        if inspect.isawaitable(result):
            await result

    async def on_snapshot_begin(self, spmd_dir: str) -> None:
        instance = self._bound_instance()
        if instance is not None:
            _call_state_hook(instance, "on_snapshot_begin", spmd_dir)

    async def on_snapshot_commit(self, spmd_dir: str, op_states: Any) -> None:
        instance = self._bound_instance()
        if instance is not None:
            _call_state_hook(instance, "on_snapshot_commit", spmd_dir, op_states)

    async def save_state(self, ckpt_dir: str) -> Any:
        if self._save_state is not None:
            return self._save_state(ckpt_dir)
        instance = self._bound_instance()
        if instance is None:
            return None
        return _call_state_hook(instance, "save_state", ckpt_dir)

    async def restore_state(self, state: Any, ckpt_dir: str) -> None:
        if self._restore_state is not None:
            self._restore_state(state, ckpt_dir)
            return
        instance = self._bound_instance()
        if instance is not None:
            _call_state_hook(instance, "restore_state", state, ckpt_dir)

    def _bound_instance(self) -> Any | None:
        return getattr(self._fn, "__self__", None)


@ray.remote(max_concurrency=1)  # type: ignore
class _RayOperatorExecutionActor:
    def __init__(
        self,
        config: Any,
        rank: int,
        spmd_id: str,
        local_rank: int | None,
        local_world_size: int | None,
        node_rank: int,
        num_nodes: int,
        rank0_address: str | None,
        owner_classes: tuple[tuple[str, str, type], ...],
        profiler_config: Any | None,
        profiler_process_name: str,
    ) -> None:
        from fluxway.spmd.worker.operator_manager import OperatorManager
        from fluxway.spmd.worker.worker import SPMDWorker

        for module_name, class_qualname, cls in owner_classes:
            _register_owner_class(module_name, class_qualname, cls)
        SPMDWorker._pre_instantiate_operators(
            config,
            rank,
            operator_location="execution_actor",
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=node_rank,
            num_nodes=num_nodes,
            rank0_address=rank0_address,
        )
        self._config = config
        self._rank = rank
        self._profiler_runtime = ProfilerRuntimeFactory.from_config(
            profiler_config,
            process_name=profiler_process_name,
        )
        self._manager = OperatorManager(
            config,
            rank=rank,
            spmd_id=spmd_id,
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=node_rank,
            num_nodes=num_nodes,
        )

    async def execute(
        self,
        op_name: str,
        step: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if self._profiler_runtime.enabled:
            with activate_user_profiler(
                recorder=self._profiler_runtime.recorder,
                op_name=op_name,
                step=step,
                rank=self._rank,
                execution_mode=UserCallableLane.RAY_EXECUTION_ACTOR.value,
            ):
                return await self._manager.execute_async(op_name, step, *args, **kwargs)
        return await self._manager.execute_async(op_name, step, *args, **kwargs)

    async def export_profiler_snapshot(self) -> Any:
        return self._profiler_runtime.snapshot()

    async def shutdown(self, operator_names: tuple[str, ...]) -> None:
        instances: dict[int, Any] = {}
        for op_name in operator_names:
            instance = self._operator_instance(op_name)
            instances.setdefault(id(instance), instance)
        for instance in instances.values():
            result = _call_state_hook(instance, "on_shutdown")
            if inspect.isawaitable(result):
                await result

    async def save_state(self, op_name: str, ckpt_dir: str) -> Any:
        instance = self._operator_instance(op_name)
        return _call_state_hook(instance, "save_state", ckpt_dir)

    async def restore_state(self, op_name: str, state: Any, ckpt_dir: str) -> None:
        instance = self._operator_instance(op_name)
        _call_state_hook(instance, "restore_state", state, ckpt_dir)

    async def on_snapshot_begin(self, op_name: str, spmd_dir: str) -> None:
        instance = self._operator_instance(op_name)
        _call_state_hook(instance, "on_snapshot_begin", spmd_dir)

    async def on_snapshot_commit(
        self, op_name: str, spmd_dir: str, op_states: Any
    ) -> None:
        instance = self._operator_instance(op_name)
        _call_state_hook(instance, "on_snapshot_commit", spmd_dir, op_states)

    def _operator_instance(self, op_name: str) -> Any:
        from fluxway.common.decorators import _resolve_owner_class, get_instance_storage

        op_meta = (self._config.operator_metas or {}).get(op_name)
        if op_meta is None or not op_meta.owner_class_qualname:
            raise RuntimeError(f"operator {op_name!r} has no stateful owner metadata.")
        cls = _resolve_owner_class(
            op_meta.owner_class_module, op_meta.owner_class_qualname
        )
        storage, lock = get_instance_storage()
        with lock:
            instance = storage.get(id(cls))
        if instance is None:
            raise RuntimeError(f"operator {op_name!r} instance is not initialized.")
        return instance


class UserCallableExecutor:
    def __init__(
        self,
        *,
        worker_process_runtime: WorkerProcessCallRuntime,
        sample_count: int = 3,
        sync_latency_threshold_sec: float = 0.010,
        profiler_config: Any | None = None,
        profiler_process_name: str | None = None,
    ) -> None:
        if profiler_config is not None and not profiler_process_name:
            raise ValueError(
                "profiler_process_name is required when profiler_config is set."
            )
        self._sample_count = sample_count
        self._sync_latency_threshold_sec = sync_latency_threshold_sec
        self._profiler_config = profiler_config
        self._profiler_process_name = profiler_process_name or ""
        self._profiles: dict[str, _CallableProfile] = defaultdict(
            lambda: _CallableProfile(samples=[])
        )
        self._ray_actors: dict[tuple[Any, ...], _ActorRecord] = {}
        self._local_free_thread_locks: dict[int, asyncio.Lock] = {}
        self._free_threading_enabled = python_free_threading_enabled()
        self._worker_process_runtime = worker_process_runtime

    async def execute(self, spec: UserCallableSpec) -> Any:
        lane = self._lane_for(spec)
        if lane is UserCallableLane.INLINE_ASYNC:
            return await spec.fn()
        if lane is UserCallableLane.INLINE_SYNC:
            return self._execute_sync_sampled(spec)
        if lane is UserCallableLane.WORKER_PROCESS:
            return await self._execute_worker_process(spec)
        if lane is UserCallableLane.LOCAL_FREE_THREAD:
            return await self._execute_local_free_thread(spec)
        if lane is UserCallableLane.RAY_EXECUTION_ACTOR:
            return await self._execute_ray_actor(spec)
        if lane is UserCallableLane.NATIVE_BACKEND:
            raise NotImplementedError(
                "native_backend user callable lane is not installed."
            )
        raise ValueError(f"unsupported user callable lane {lane!r}.")

    async def drain(self) -> None:
        await self._worker_process_runtime.drain()

    async def shutdown(self) -> None:
        shutdown_refs = []
        shutdown_errors: list[Exception] = []
        for record in self._ray_actors.values():
            try:
                if record.operator_name is None:
                    shutdown_refs.append(record.actor.shutdown.remote())
                else:
                    operator_names = tuple(
                        sorted(
                            name for name in record.operator_names if name is not None
                        )
                    )
                    shutdown_refs.append(record.actor.shutdown.remote(operator_names))
            except Exception as error:
                shutdown_errors.append(error)
        results = await asyncio.gather(*shutdown_refs, return_exceptions=True)
        shutdown_errors.extend(
            result for result in results if isinstance(result, Exception)
        )
        self._ray_actors.clear()
        if shutdown_errors:
            raise ExceptionGroup(
                "execution actor shutdown failed",
                shutdown_errors,
            )

    async def export_profiler_snapshots(self) -> tuple[Any, ...]:
        refs = [
            record.actor.export_profiler_snapshot.remote()
            for record in self._ray_actors.values()
        ]
        if not refs:
            return ()
        results = await asyncio.gather(*refs)
        return tuple(snapshot for snapshot in results if snapshot is not None)

    def actor_owned_operator_names(self) -> set[str | None]:
        return {
            op_name
            for record in self._ray_actors.values()
            for op_name in record.operator_names
        }

    async def trigger_snapshot_begin(self, spmd_dir: str) -> None:
        refs = []
        for record in self._ray_actors.values():
            if record.operator_name is None:
                refs.append(record.actor.on_snapshot_begin.remote(spmd_dir))
            else:
                refs.append(
                    record.actor.on_snapshot_begin.remote(
                        record.operator_name, spmd_dir
                    )
                )
        if refs:
            await asyncio.gather(*refs)

    async def trigger_snapshot_commit(
        self, spmd_dir: str, operator_states: dict[str, Any]
    ) -> None:
        refs = []
        for record in self._ray_actors.values():
            if record.operator_name is None:
                refs.append(
                    record.actor.on_snapshot_commit.remote(
                        spmd_dir, operator_states.get(record.callable_id, {})
                    )
                )
            else:
                refs.append(
                    record.actor.on_snapshot_commit.remote(
                        record.operator_name,
                        spmd_dir,
                        operator_states.get(record.operator_name, {}),
                    )
                )
        if refs:
            await asyncio.gather(*refs)

    async def save_state(self, ckpt_dir: str) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for record in self._ray_actors.values():
            if record.operator_name is None:
                state = await record.actor.save_state.remote(ckpt_dir)
                state_key = record.callable_id
            else:
                state = await record.actor.save_state.remote(
                    record.operator_name, ckpt_dir
                )
                state_key = record.operator_name
            if state is not None:
                states[state_key] = state
        return states

    async def restore_state(self, states: dict[str, Any], ckpt_dir: str) -> None:
        refs = []
        for record in self._ray_actors.values():
            if record.operator_name is None:
                state_key = record.callable_id
                if state_key in states:
                    refs.append(
                        record.actor.restore_state.remote(states[state_key], ckpt_dir)
                    )
            else:
                state_key = record.operator_name
                if state_key in states:
                    refs.append(
                        record.actor.restore_state.remote(
                            record.operator_name, states[state_key], ckpt_dir
                        )
                    )
        if refs:
            await asyncio.gather(*refs)

    def _lane_for(self, spec: UserCallableSpec) -> UserCallableLane:
        if spec.process_affine:
            return UserCallableLane.WORKER_PROCESS
        if inspect.iscoroutinefunction(spec.fn) and not spec.is_stateful:
            return UserCallableLane.INLINE_ASYNC
        profile = self._profiles[spec.callable_id]
        if spec.is_stateful:
            profile.lane = UserCallableLane.RAY_EXECUTION_ACTOR
            return profile.lane
        if self._free_threading_enabled:
            return UserCallableLane.LOCAL_FREE_THREAD
        if profile.lane is not None:
            return profile.lane
        if len(profile.samples) < self._sample_count:
            return UserCallableLane.INLINE_SYNC
        p50 = statistics.median(profile.samples)
        profile.lane = (
            UserCallableLane.RAY_EXECUTION_ACTOR
            if p50 > self._sync_latency_threshold_sec
            else UserCallableLane.INLINE_SYNC
        )
        return profile.lane

    def _execute_sync_sampled(self, spec: UserCallableSpec) -> Any:
        start = time.perf_counter()
        result = spec.fn()
        elapsed = time.perf_counter() - start
        profile = self._profiles[spec.callable_id]
        if len(profile.samples) < self._sample_count:
            profile.samples.append(elapsed)
            if len(profile.samples) == self._sample_count:
                p50 = statistics.median(profile.samples)
                if p50 > self._sync_latency_threshold_sec:
                    profile.lane = UserCallableLane.RAY_EXECUTION_ACTOR
                else:
                    profile.lane = UserCallableLane.INLINE_SYNC
        return result

    async def _execute_local_free_thread(self, spec: UserCallableSpec) -> Any:
        lock = self._local_free_thread_locks.setdefault(
            spec.resource_id, asyncio.Lock()
        )
        async with lock:
            return await asyncio.to_thread(spec.fn)

    async def _execute_worker_process(self, spec: UserCallableSpec) -> Any:
        if not spec.operator_lane:
            raise ValueError(
                f"process-affine callable {spec.callable_id!r} requires operator_lane."
            )
        return await self._worker_process_runtime.submit(
            spec.fn,
            callable_id=spec.callable_id,
            resource_id=spec.resource_id,
            execution_lane=spec.operator_lane,
        )

    async def _execute_ray_actor(self, spec: UserCallableSpec) -> Any:
        key = self._ray_actor_key(spec)
        record = self._ray_actors.get(key)
        if record is None:
            record = self._create_ray_actor(spec)
            self._ray_actors[key] = record
        elif spec.is_stateful and spec.operator_name is not None:
            record.operator_names.add(spec.operator_name)
        if record.operator_name is None:
            return await record.actor.execute.remote(
                self._profiler_op_name(spec),
                self._profiler_step(spec),
                self._profiler_rank(spec),
            )
        step = spec.operator_step
        if step is None:
            raise ValueError("stateful operator execution requires operator_step.")
        if spec.operator_name is None:
            raise ValueError("stateful operator execution requires operator_name.")
        return await record.actor.execute.remote(
            spec.operator_name,
            step,
            spec.operator_args,
            {} if spec.operator_kwargs is None else spec.operator_kwargs,
        )

    def _ray_actor_key(self, spec: UserCallableSpec) -> tuple[Any, ...]:
        if spec.is_stateful and spec.operator_name is not None:
            if not spec.operator_owner_classes:
                raise ValueError(
                    "stateful operator execution requires operator_owner_classes."
                )
            owner_module, owner_qualname, _ = spec.operator_owner_classes[0]
            return (
                "operator",
                spec.operator_spmd_id,
                owner_module,
                owner_qualname,
                spec.resource_id,
            )
        return ("callable", spec.callable_id, spec.resource_id)

    def _create_ray_actor(self, spec: UserCallableSpec) -> _ActorRecord:
        if spec.is_stateful and spec.operator_name is not None:
            return self._create_operator_actor(spec)
        actor = _RayFunctionExecutionActor.remote(  # type: ignore
            spec.fn,
            spec.save_state,
            spec.restore_state,
            self._profiler_config,
            self._profiler_process_name,
        )
        return _ActorRecord(
            actor=actor,
            callable_id=spec.callable_id,
            operator_name=None,
            operator_names=set(),
        )

    def _create_operator_actor(self, spec: UserCallableSpec) -> _ActorRecord:
        if spec.operator_config is None:
            raise ValueError("stateful operator execution requires operator_config.")
        if spec.operator_rank is None:
            raise ValueError("stateful operator execution requires operator_rank.")
        actor = _RayOperatorExecutionActor.remote(  # type: ignore
            spec.operator_config,
            spec.operator_rank,
            spec.operator_spmd_id,
            spec.local_rank,
            spec.local_world_size,
            spec.node_rank,
            spec.num_nodes,
            spec.rank0_address,
            spec.operator_owner_classes,
            self._profiler_config,
            self._profiler_process_name,
        )
        return _ActorRecord(
            actor=actor,
            callable_id=spec.callable_id,
            operator_name=spec.operator_name,
            operator_names={spec.operator_name},
        )

    @staticmethod
    def _profiler_op_name(spec: UserCallableSpec) -> str:
        return spec.operator_name or spec.callable_id

    @staticmethod
    def _profiler_step(spec: UserCallableSpec) -> int:
        if spec.operator_step is None:
            raise ValueError("ray execution actor profiling requires operator_step.")
        return spec.operator_step

    @staticmethod
    def _profiler_rank(spec: UserCallableSpec) -> int:
        if spec.operator_rank is None:
            raise ValueError("ray execution actor profiling requires operator_rank.")
        return spec.operator_rank
