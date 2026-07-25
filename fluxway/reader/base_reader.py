"""
Base class for all Reader SPMDs.
"""

from typing import Any
from fluxway.common.decorators import spmd, operator


@spmd()
class BaseReader:
    """
    Abstract base class for high-performance data readers.

    A Reader is a specialized SPMD source node that initiates data flow
    in the computational graph.
    """

    rank: int
    world_size: int
    static_args: dict[str, Any]

    def __init__(
        self,
        name: str = "",
        mesh: Any = None,
        runtime_env: Any = None,
        resource: Any = None,
        **kwargs: Any,
    ):
        pass

    @operator("read")
    def read(self) -> Any:
        """
        Produce a batch of data.
        Must return a @data_type decorated object or a Pytree of tensors.
        """
        raise NotImplementedError("Subclasses must implement read()")
