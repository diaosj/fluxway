"""Fluxway integration for the optional ``sparse_embedding`` package."""

from .checkpoint import (
    GravitinoLanceStorageFactory,
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
    compact_sparse_checkpoints,
    modulo_partition,
)

__all__ = [
    "GravitinoLanceStorageFactory",
    "LocalLanceStorageFactory",
    "SparseEmbeddingCheckpointComponent",
    "compact_sparse_checkpoints",
    "modulo_partition",
]
