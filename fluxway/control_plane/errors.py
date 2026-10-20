from collections.abc import Mapping

from fluxway.common.types import ExecutionFailure


class ExecutionFailedError(RuntimeError):
    def __init__(self, failures: Mapping[str, ExecutionFailure]) -> None:
        self.failures = dict(failures)
        if not self.failures:
            raise ValueError("ExecutionFailedError requires at least one failure.")
        mismatched = {
            spmd_id: failure.spmd_id
            for spmd_id, failure in self.failures.items()
            if spmd_id != failure.spmd_id
        }
        if mismatched:
            raise ValueError(
                "ExecutionFailedError failure keys must match failure.spmd_id: "
                f"{mismatched}"
            )
        super().__init__(self.failures)

    def __str__(self) -> str:
        failures = "; ".join(
            str(self.failures[spmd_id]) for spmd_id in sorted(self.failures)
        )
        return f"Fluxway execution failed: {failures}"
