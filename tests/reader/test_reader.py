"""
End-to-end test for the Reader → Processor graph pipeline.

Verifies the full lifecycle: graph compilation → controller startup →
topology negotiation → actual data flow between a GrainReader SPMD and a
downstream Processor SPMD.
"""

import asyncio
import os

import pytest
import ray

from fluxway.control_plane.controller import create_controller
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.placement import AFFINITY_FIRST_V1, PlacementRecipe, ResourceProfile

# Import helper nodes from the separate module so Ray can deserialize them.
from tests.reader.helper_nodes import MockDataSource, SimpleProcessor, TypedGrainReader

os.environ["FLUXWAY_BACKEND"] = "jax"

_DATA_SIZE = 10
_POLL_INTERVAL_S = 1.0
_MAX_POLL_ROUNDS = 30


@pytest.fixture(scope="module", autouse=True)
def ray_session():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=True)
    yield
    # Ray is intentionally kept alive across tests in this module to avoid
    # repeated cluster startup cost.


def _build_graph() -> FunctionalGraph:
    graph = FunctionalGraph()
    reader = graph.reader(
        "Reader",
        reader_class=TypedGrainReader,
        mesh_axis_names=("data",),
        mesh_shape=(1,),
        data_source=MockDataSource(_DATA_SIZE),
        shuffle=False,
        num_epochs=1,
    )
    processor = graph.node("Processor", SimpleProcessor)
    processor.process(batch=reader.read())
    return graph


@pytest.mark.asyncio
async def test_reader_processor_data_flows():
    """Both Reader and Processor must make progress within the timeout."""
    task_graph = _build_graph().compile("ReaderE2E")

    reader_id = next(n for n in task_graph.nodes if "Reader" in n)
    proc_id = next(n for n in task_graph.nodes if "Processor" in n)

    recipe = PlacementRecipe(
        recipe_id="reader-e2e",
        version="1.0.0",
        profiles={
            TypedGrainReader: ResourceProfile(base={"CPU": 1}),
            SimpleProcessor: ResourceProfile(base={"CPU": 1}),
        },
        constraints=(),
        objective=AFFINITY_FIRST_V1,
    )
    controller = await create_controller(task_graph, placement_recipe=recipe)
    try:
        for _ in range(_MAX_POLL_ROUNDS):
            await asyncio.sleep(_POLL_INTERVAL_S)
            progress = await controller.get_async_step_progress.remote()
            reader_steps = progress.get(reader_id, {})
            proc_steps = progress.get(proc_id, {})
            if reader_steps.get("read", 0) > 0 and proc_steps.get("process", 0) > 0:
                break

        progress = await controller.get_async_step_progress.remote()
        reader_steps = progress.get(reader_id, {})
        proc_steps = progress.get(proc_id, {})

        assert reader_steps.get("read", 0) > 0, (
            f"Reader {reader_id} never produced data. Async progress: {reader_steps}"
        )
        assert proc_steps.get("process", 0) > 0, (
            f"Processor {proc_id} never consumed data. Async progress: {proc_steps}"
        )
    finally:
        await controller.stop.remote()


if __name__ == "__main__":
    asyncio.run(test_reader_processor_data_flows())
