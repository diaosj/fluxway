"""Compatibility import for the recommendation example.

The third-party integration lives under :mod:`fluxway.integrations` so Fluxway's
core remains independent from sparse_embedding.
"""

from fluxway.integrations.sparse_embedding.checkpoint import (
    GravitinoLanceStorageFactory,
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
)

__all__ = [
    "GravitinoLanceStorageFactory",
    "LocalLanceStorageFactory",
    "SparseEmbeddingCheckpointComponent",
]
