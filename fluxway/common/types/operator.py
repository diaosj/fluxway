from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class StatefulOperatorProtocol(Protocol):
    """
    Protocol for stateful operators in Fluxway.
    Operators that need to persist state across snapshots should implement these methods.
    """

    def on_snapshot_begin(self, ckpt_dir: str) -> None: ...
    def on_snapshot_commit(
        self, ckpt_dir: str, gathered_states: dict[int, dict]
    ) -> None: ...
    def save_state(self, ckpt_dir: str = "") -> dict[str, Any]: ...
    def restore_state(self, state: dict[str, Any], ckpt_dir: str = "") -> None: ...
