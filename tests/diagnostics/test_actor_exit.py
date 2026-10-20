from __future__ import annotations

from ray.core.generated.common_pb2 import ActorDiedErrorContext
from ray.exceptions import ActorDiedError

from fluxway.diagnostics.actor_exit import report_actor_exit


def test_actor_exit_reporter_preserves_ray_fact_and_bounds_message() -> None:
    error = ActorDiedError(
        ActorDiedErrorContext(
            class_name="SPMDWorker",
            actor_id=b"1234567890123456",
            pid=173116,
            error_message="x" * 5000,
        )
    )
    events: list[tuple[str, dict[str, str]]] = []

    report_actor_exit(
        error,
        spmd_id="dense",
        emit=lambda message, **fields: events.append((message, fields)),
    )

    assert len(events) == 1
    message, fields = events[0]
    assert message == "Fluxway Ray actor exited unexpectedly"
    assert fields["diagnostic_kind"] == "actor_exit"
    assert fields["exception_type"] == "ActorDiedError"
    assert fields["spmd_id"] == "dense"
    assert fields["actor_id"] == "31323334353637383930313233343536"
    assert len(fields["error_message"]) == 4096


def test_actor_exit_dashboard_adapter_emits_core_worker_event(monkeypatch) -> None:
    from fluxway.diagnostics import actor_exit

    error = ActorDiedError(
        ActorDiedErrorContext(
            class_name="SPMDWorker",
            actor_id=b"1234567890123456",
            error_message="signal",
        )
    )
    events: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        actor_exit,
        "global_worker",
        type(
            "Worker",
            (),
            {"node": type("Node", (), {"get_logs_dir_path": lambda _self: "/tmp"})()},
        )(),
    )
    monkeypatch.setattr(
        actor_exit,
        "get_event_logger",
        lambda _source, _logs_dir: type(
            "Logger",
            (),
            {
                "error": lambda _self, message, **fields: events.append(
                    (message, fields)
                )
            },
        )(),
    )

    actor_exit.report_actor_exit_to_dashboard(error, spmd_id="dense")

    assert events[0][0] == "Fluxway Ray actor exited unexpectedly"
    assert events[0][1]["diagnostic_kind"] == "actor_exit"
