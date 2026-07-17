from __future__ import annotations

from dataclasses import dataclass
import traceback as traceback_module


@dataclass(frozen=True, slots=True)
class ExecutionFailure:
    spmd_id: str
    rank: int | None
    op_name: str | None
    step: int | None
    phase: str
    exception_type: str
    message: str
    traceback: str | None

    def __post_init__(self) -> None:
        if not self.spmd_id:
            raise ValueError("ExecutionFailure requires a non-empty spmd_id.")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise ValueError("ExecutionFailure rank must be a non-negative integer.")
        if self.op_name is not None and not self.op_name:
            raise ValueError("ExecutionFailure op_name cannot be empty.")
        if self.step is not None and (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValueError("ExecutionFailure step must be a non-negative integer.")
        if not self.phase:
            raise ValueError("ExecutionFailure requires a non-empty phase.")
        if not self.exception_type:
            raise ValueError("ExecutionFailure requires a non-empty exception_type.")

    @classmethod
    def from_exception(
        cls,
        *,
        spmd_id: str,
        rank: int | None,
        phase: str,
        exception: BaseException,
        op_name: str | None = None,
        step: int | None = None,
    ) -> ExecutionFailure:
        return cls(
            spmd_id=spmd_id,
            rank=rank,
            op_name=op_name,
            step=step,
            phase=phase,
            exception_type=type(exception).__name__,
            message=str(exception),
            traceback="".join(
                traceback_module.format_exception(
                    type(exception),
                    exception,
                    exception.__traceback__,
                )
            ),
        )

    def __str__(self) -> str:
        context = [self.spmd_id]
        if self.rank is not None:
            context.append(f"rank={self.rank}")
        if self.op_name is not None:
            context.append(f"op={self.op_name}")
        if self.step is not None:
            context.append(f"step={self.step}")
        context.append(f"phase={self.phase}")
        return f"{' '.join(context)} {self.exception_type}: {self.message}"
