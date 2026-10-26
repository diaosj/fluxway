from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import SPMDConfig
from fluxway import OperatorSpec
from fluxway.common.compiled_graph import compile_graph
from fluxway.spmd.driver.driver import SPMDDriver
from tests.compiled_graph_fixtures import compiled_spmd_spec
import pytest


def _driver_cls():
    cls = getattr(SPMDDriver, "__ray_metadata__", None)
    return cls.modified_class if cls else SPMDDriver


def _make_driver():
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(4,))
    operator = OperatorSpec(name="compute")
    compiled_graph = compile_graph(
        name="driver-local-rank-test",
        spmds=(
            compiled_spmd_spec(
                spmd_id="spmd-a",
                mesh=mesh,
                operators={"compute": operator},
            ),
        ),
        dependencies=(),
    ).spmd("spmd-a")
    config = SPMDConfig(
        spmd_id="spmd-a",
        graph_view=compiled_graph,
        mesh=mesh,
        operator_specs={"compute": operator},
    )
    driver_cls = _driver_cls()
    return driver_cls(
        spmd_id="spmd-a",
        config=config,
        placement_group=object(),
        worker_bundle_indices=[],
        controller_handle=None,
        graph_topology={"predecessors": [], "successors": []},
        worker_resource={"CPU": 0},
        expected_node_ids=[],
    )


def test_compute_rank_info_single_node():
    driver = _make_driver()
    node_ids = {0: "node-a", 1: "node-a", 2: "node-a"}

    assert driver.executor._compute_rank_info(node_ids) == {
        0: {"local_rank": 0, "local_world_size": 3, "node_rank": 0, "num_nodes": 1},
        1: {"local_rank": 1, "local_world_size": 3, "node_rank": 0, "num_nodes": 1},
        2: {"local_rank": 2, "local_world_size": 3, "node_rank": 0, "num_nodes": 1},
    }


def test_compute_rank_info_partial_fill_orders_node_rank_by_smallest_rank():
    driver = _make_driver()
    node_ids = {0: "node-b", 1: "node-b", 2: "node-a", 3: "node-a", 4: "node-a"}

    assert driver.executor._compute_rank_info(node_ids) == {
        0: {"local_rank": 0, "local_world_size": 2, "node_rank": 0, "num_nodes": 2},
        1: {"local_rank": 1, "local_world_size": 2, "node_rank": 0, "num_nodes": 2},
        2: {"local_rank": 0, "local_world_size": 3, "node_rank": 1, "num_nodes": 2},
        3: {"local_rank": 1, "local_world_size": 3, "node_rank": 1, "num_nodes": 2},
        4: {"local_rank": 2, "local_world_size": 3, "node_rank": 1, "num_nodes": 2},
    }


class _FakeAsyncRemoteCall:
    def __init__(self, func):
        async def remote(*args, **kwargs):
            return func(*args, **kwargs)

        self.remote = remote


class _DummyFuture:
    def __await__(self):
        yield
        return None


class _FakeSyncRemoteCall:
    def __init__(self, func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            return _DummyFuture()

        self.remote = wrapper


class FakeWorker:
    def __init__(self, node_id):
        self.node_id = node_id
        self.setup_calls = []
        self.find_free_port_calls = []
        self.get_node_id = _FakeAsyncRemoteCall(lambda: node_id)
        self.get_node_ip = _FakeAsyncRemoteCall(lambda: "127.0.0.1")
        self.find_free_port = _FakeAsyncRemoteCall(
            lambda host=None: self.find_free_port_calls.append(host) or 29500
        )
        self.setup_and_start = _FakeSyncRemoteCall(
            lambda **kwargs: self.setup_calls.append(kwargs)
        )


@pytest.mark.asyncio
async def test_start_workers_queries_node_ids_before_setup(monkeypatch):
    driver = _make_driver()
    driver.executor.worker_bundle_indices = [0, 1, 2]
    driver.executor.expected_node_ids = ["node-a", "node-a", "node-b"]
    fake_workers = [FakeWorker("node-a"), FakeWorker("node-a"), FakeWorker("node-b")]
    worker_iter = iter(fake_workers)

    class _FakeActorClass:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            return next(worker_iter)

    monkeypatch.setattr(
        "fluxway.spmd.driver.executor.ray_executor.SPMDWorker",
        _FakeActorClass(),
    )
    await driver.executor.start_workers(driver.spmd_id, None, driver.controller_handle)

    assert fake_workers[0].setup_calls == [
        {
            "local_rank": 0,
            "local_world_size": 2,
            "node_rank": 0,
            "num_nodes": 2,
            "rank0_address": "127.0.0.1:29500",
        }
    ]
    assert fake_workers[1].setup_calls == [
        {
            "local_rank": 1,
            "local_world_size": 2,
            "node_rank": 0,
            "num_nodes": 2,
            "rank0_address": "127.0.0.1:29500",
        }
    ]
    assert fake_workers[2].setup_calls == [
        {
            "local_rank": 0,
            "local_world_size": 1,
            "node_rank": 1,
            "num_nodes": 2,
            "rank0_address": "127.0.0.1:29500",
        }
    ]
    assert fake_workers[0].find_free_port_calls == ["127.0.0.1"]


def test_worker_find_free_port_real():
    from fluxway.spmd.worker.worker import SPMDWorker

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    worker_cls = cls.modified_class if cls else SPMDWorker
    worker = worker_cls.__new__(worker_cls)
    port = worker.find_free_port(host="127.0.0.1")
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_worker_find_free_port_candidate_iteration(monkeypatch):
    import socket
    from fluxway.spmd.worker.worker import SPMDWorker

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    worker_cls = cls.modified_class if cls else SPMDWorker
    worker = worker_cls.__new__(worker_cls)

    candidates = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 1)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: candidates)
    port = worker.find_free_port(host="127.0.0.1")
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_worker_find_free_port_all_fail_raises(monkeypatch):
    import socket
    from fluxway.spmd.worker.worker import SPMDWorker

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    worker_cls = cls.modified_class if cls else SPMDWorker
    worker = worker_cls.__new__(worker_cls)

    candidates = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 1)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: candidates)
    with pytest.raises(OSError):
        worker.find_free_port(host="127.0.0.1")
