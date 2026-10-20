import asyncio

import pytest_asyncio

from fluxway.common.io_lane import IOLane
from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer


@pytest_asyncio.fixture
async def io_lane():
    lane = IOLane("test-checkpoint-repository", capacity=1)
    try:
        yield lane
    finally:
        await lane.close()


@pytest_asyncio.fixture
async def checkpointer_factory():
    instances = []

    async def create(*args, **kwargs):
        instance = await FluxwayCheckpointer.create(*args, **kwargs)
        instances.append(instance)
        return instance

    try:
        yield create
    finally:
        await asyncio.gather(*(item.close() for item in instances if not item._closed))
