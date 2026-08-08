from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "fluxway" / "example" / "recommendation"


def test_recommendation_examples_register_backends_at_spmd_boundary():
    expected = {
        "reader.py": "StateCheckpointComponent",
        "dense_spmd.py": "OrbaxCheckpointComponent",
        "sparse_spmd.py": "SparseEmbeddingCheckpointComponent",
    }
    for filename, backend in expected.items():
        source = (EXAMPLE / filename).read_text()
        assert "checkpoint_components=" in source
        assert backend in source


def test_recommendation_examples_do_not_define_operator_checkpoint_hooks():
    obsolete_hooks = tuple(
        "_".join(parts)
        for parts in (
            ("start_save", "checkpoint", "checkpoint_objects"),
            ("checkpoint", "metadata"),
            ("commit_save", "checkpoint", "checkpoint_objects"),
            ("rollback_save", "checkpoint", "checkpoint_objects"),
        )
    )
    source = "\n".join(path.read_text() for path in EXAMPLE.glob("*.py"))
    for hook in obsolete_hooks:
        assert hook not in source
