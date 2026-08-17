import pytest
import ray

import fluxway
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types import AsyncStepConfig
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.placement import AFFINITY_FIRST_V1, PlacementRecipe
from tests.spmd.interleaved_source_nodes import (
    ImmediatelyExhaustedInterleavedSource,
    InterleavedSourceSink,
)


@pytest.fixture(scope="module", autouse=True)
def ray_session():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=True)
    yield


def _build_exhausted_source_graph():
    with FunctionalGraph("interleaved-source-exhaustion") as graph:
        source = ImmediatelyExhaustedInterleavedSource(
            name="Reader",  # type: ignore[call-arg]
            mesh=DeviceMesh(("worker",), (2,)),  # type: ignore[call-arg]
        )
        sink = InterleavedSourceSink(
            name="Sink",  # type: ignore[call-arg]
            mesh=DeviceMesh(("worker",), (1,)),  # type: ignore[call-arg]
        )
        sink.consume(source.read())
        return graph.build()


@pytest.mark.asyncio
async def test_interleaved_source_local_exhaustion_completes_execution():
    execution = await fluxway.start(
        _build_exhausted_source_graph(),
        placement_recipe=PlacementRecipe(
            recipe_id="interleaved-source-exhaustion",
            version="1.0.0",
            constraints=(),
            objective=AFFINITY_FIRST_V1,
        ),
        async_step_config=AsyncStepConfig(max_steps=2),
    )
    try:
        await execution.wait(timeout=10)
        progress = await execution.progress()
        assert progress["Sink"].get("consume", 0) == 0
    finally:
        await execution.cancel()
