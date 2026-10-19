"""
End-to-end test for the Reader → Processor graph pipeline.

Verifies the full lifecycle: graph compilation → controller startup →
topology negotiation → actual data flow between a GrainReader SPMD and a
downstream Processor SPMD.
"""

import asyncio
import os

import pytest

import fluxway
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.placement import AFFINITY_FIRST, PlacementRecipe, ResourceProfile

# Import helper nodes from the separate module so Ray can deserialize them.
from tests.reader.helper_nodes import MockDataSource, SimpleProcessor, TypedGrainReader
from tests.ray_cluster_process import local_ray_cluster

os.environ["FLUXWAY_BACKEND"] = "jax"

_DATA_SIZE = 10
_POLL_INTERVAL_S = 1.0
_MAX_POLL_ROUNDS = 30


@pytest.fixture(scope="module", autouse=True)
def ray_session():
    with local_ray_cluster(num_cpus=8):
        yield


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
        resource_profile=ResourceProfile(base={"CPU": 1}),
    )
    processor = graph.node(
        "Processor",
        SimpleProcessor,
        resource_profile=ResourceProfile(base={"CPU": 1}),
    )
    processor.process(batch=reader.read())
    return graph


@pytest.mark.asyncio
async def test_reader_processor_data_flows():
    """Both Reader and Processor must make progress within the timeout."""
    recipe = PlacementRecipe(
        recipe_id="reader-e2e",
        constraints=(),
        objective=AFFINITY_FIRST,
    )
    graph = _build_graph().build("ReaderE2E")
    reader_id = next(n for n in graph.nodes if "Reader" in n)
    proc_id = next(n for n in graph.nodes if "Processor" in n)
    execution = await fluxway.start(graph, placement_recipe=recipe)
    try:
        for _ in range(_MAX_POLL_ROUNDS):
            await asyncio.sleep(_POLL_INTERVAL_S)
            progress = await execution.progress()
            reader_steps = progress.get(reader_id, {})
            proc_steps = progress.get(proc_id, {})
            if reader_steps.get("read", 0) > 0 and proc_steps.get("process", 0) > 0:
                break

        progress = await execution.progress()
        reader_steps = progress.get(reader_id, {})
        proc_steps = progress.get(proc_id, {})

        assert reader_steps.get("read", 0) > 0, (
            f"Reader {reader_id} never produced data. Async progress: {reader_steps}"
        )
        assert proc_steps.get("process", 0) > 0, (
            f"Processor {proc_id} never consumed data. Async progress: {proc_steps}"
        )
    finally:
        await execution.cancel()


if __name__ == "__main__":
    asyncio.run(test_reader_processor_data_flows())
