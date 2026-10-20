"""
Fluxway Reader module for data reading.

This module provides data reading capabilities for Fluxway framework,
including various datasource implementations.
"""

# Ensure INFO logs are visible even when importing this subpackage directly.
from fluxway.common.utils.logging import ensure_logging_configured


from .base_reader import BaseReader
from .grain_reader import GrainReader, StridedIndexSampler
from .ingestion_layout import (
    IngestionReadMetrics,
    IngestionLayoutSession,
    IngestionReadStream,
    IngestionReadSession,
    ScheduledSegmentRead,
    SegmentReadChunk,
    read_ingestion_layout,
)

__all__ = [
    "BaseReader",
    "GrainReader",
    "IngestionReadMetrics",
    "IngestionLayoutSession",
    "IngestionReadStream",
    "IngestionReadSession",
    "ScheduledSegmentRead",
    "SegmentReadChunk",
    "StridedIndexSampler",
    "read_ingestion_layout",
]

ensure_logging_configured()
