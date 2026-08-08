from fluxway.common.types.checkpoint import RestoreTopologyMode
from fluxway.control_plane.checkpointer.orbax import OrbaxCheckpointComponent


def test_orbax_declares_global_reshard_and_requires_restore_layout():
    async def save(module, context):
        return {}

    async def restore(module, state, context):
        return None

    async def layout(module, context):
        return {}

    backend = OrbaxCheckpointComponent("dense", save, restore, layout)
    assert backend.descriptor.restore_topology is RestoreTopologyMode.GLOBAL_RESHARD
