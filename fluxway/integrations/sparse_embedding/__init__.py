"""Fluxway integration for the optional ``sparse_embedding`` package."""

from .checkpoint import (
    GravitinoLanceStorageFactory,
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
    modulo_partition,
)

__all__ = [
    "GravitinoLanceStorageFactory",
    "LocalLanceStorageFactory",
    "SparseEmbeddingCheckpointComponent",
    "modulo_partition",
]
