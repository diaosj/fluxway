import pytest

from fluxway.common.coherence import CoherenceInstance, CoherenceSubject
from fluxway.spmd.worker.data_plane import InstancePublication, PublicationBatch


def test_instance_publication_requires_unique_children() -> None:
    instance = CoherenceInstance(CoherenceSubject(1, 0, 1), 0)
    with pytest.raises(ValueError, match="unique"):
        InstancePublication(instance, (2, 2))


def test_publication_batch_keeps_immutable_target_and_instance_sets() -> None:
    instance = CoherenceInstance(CoherenceSubject(1, 0, 1), 0)
    batch = object()
    publication = PublicationBatch(
        (batch,),
        (InstancePublication(instance, (0, 1)),),
    )
    assert publication.batches == (batch,)
    assert publication.publications[0].child_slots == (0, 1)
