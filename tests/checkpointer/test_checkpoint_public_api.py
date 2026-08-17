from fluxway.checkpoint import (
    CommitSaveSession,
    CheckpointComponentSpec,
    SaveSession,
    register_checkpoint_component,
    state_checkpoint_component,
)
from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointCut,
    TopologySnapshot,
)


def test_public_component_api_defaults_to_one_state_object() -> None:
    spec = CheckpointComponentSpec("reader")
    assert spec.checkpoint_objects == ()


def test_save_session_builds_opaque_checkpoint_objects() -> None:
    session = SaveSession(
        instance=object(),
        attempt_id="attempt",
        checkpoint_id="checkpoint",
        commit_permit=CheckpointCommitPermit("attempt", 2, b"permit"),
        cut=CheckpointCut.for_step("cut", 3),
        spmd_id="train",
        rank=0,
        source_topology=TopologySnapshot(1, (), (), "v1"),
        component_id="embedding",
    )
    checkpoint_object = session.checkpoint_object_ref(
        "embedding_store", "version-1", b"opaque"
    )
    checkpoint = session.component_checkpoint((checkpoint_object,))
    commit = CommitSaveSession(
        instance=session.instance,
        attempt_id=session.attempt_id,
        checkpoint_id=session.checkpoint_id,
        commit_permit=session.commit_permit,
        cut=session.cut,
        spmd_id=session.spmd_id,
        rank=session.rank,
        source_topology=session.source_topology,
        component_id=session.component_id,
        worker_results={0: session.worker_save_result(b"fragment")},
    )

    assert checkpoint.shared_records[0].checkpoint_objects == (checkpoint_object,)
    assert commit.worker_results[0].commit_metadata == b"fragment"


def test_state_component_factory_builds_a_manifest_only_component() -> None:
    component = state_checkpoint_component(
        "reader",
        lambda instance, session: {"cursor": 4},
        lambda instance, state, session: None,
    )

    assert component.descriptor.component_id == "reader"
    assert component.descriptor.checkpoint_objects == ()


def test_public_registration_uses_the_component_descriptor() -> None:
    component = state_checkpoint_component(
        "reader",
        lambda instance, session: {"cursor": 4},
        lambda instance, state, session: None,
    )

    registration = register_checkpoint_component(
        component,
        component_lifecycle=component,
        save=component,
        restore=component,
    )

    assert registration.component is component
    assert registration.descriptor is component.descriptor
