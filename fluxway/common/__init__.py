"""Common utilities and types for Fluxway framework."""

# Ensure INFO logs are visible even when importing subpackages directly.
from .utils.logging import ensure_logging_configured

from .logical_partition_binding import LogicalPartitionBinding

__all__ = ["ensure_logging_configured", "LogicalPartitionBinding"]

ensure_logging_configured()
