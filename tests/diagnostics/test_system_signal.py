from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from ray._private.ray_constants import WORKER_PROCESS_SETUP_HOOK_ENV_VAR

import fluxway.diagnostics.system_signal as system_signal


def test_worker_process_setup_hook_requires_importable_user_hook() -> None:
    assert (
        system_signal.compose_worker_process_setup_hook(None)
        == "fluxway.diagnostics.system_signal.worker_process_setup_hook"
    )
    assert (
        system_signal.compose_worker_process_setup_hook("package.module.hook")
        == "fluxway.diagnostics.system_signal.worker_process_setup_hook"
    )
    with pytest.raises(TypeError, match="module path string"):
        system_signal.compose_worker_process_setup_hook(lambda: None)


def test_prepare_worker_runtime_env_has_one_signal_bootstrap_owner() -> None:
    runtime_env = {
        "env_vars": {"UNRELATED": "value"},
        "worker_process_setup_hook": "package.module.hook",
        "pip": ["example"],
    }

    prepared = system_signal.prepare_worker_runtime_env(runtime_env)

    assert prepared == {
        "env_vars": {
            "UNRELATED": "value",
            WORKER_PROCESS_SETUP_HOOK_ENV_VAR: (
                "fluxway.diagnostics.system_signal.worker_process_setup_hook"
            ),
            system_signal._USER_SETUP_HOOK_ENV: "package.module.hook",
        },
        "pip": ["example"],
    }
    assert runtime_env == {
        "env_vars": {"UNRELATED": "value"},
        "worker_process_setup_hook": "package.module.hook",
        "pip": ["example"],
    }


def test_prepare_worker_runtime_env_rejects_conflicting_ray_hook() -> None:
    with pytest.raises(ValueError, match="must not define"):
        system_signal.prepare_worker_runtime_env(
            {"env_vars": {WORKER_PROCESS_SETUP_HOOK_ENV_VAR: "other.hook"}}
        )


def test_system_signal_reporter_emits_and_delegates(monkeypatch) -> None:
    previous_calls: list[tuple[int, object]] = []
    installed: dict[int, object] = {}
    events: list[tuple[str, dict[str, str]]] = []
    delivered = threading.Event()

    def previous(signum: int, frame: object) -> None:
        previous_calls.append((signum, frame))

    monkeypatch.setattr(
        system_signal,
        "global_worker",
        SimpleNamespace(
            node=SimpleNamespace(get_logs_dir_path=lambda: "/tmp/ray-logs"),
            _fluxway_system_signal_reporter_installed=False,
        ),
    )
    monkeypatch.setattr(
        system_signal.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "node",
            get_job_id=lambda: "job",
            get_actor_id=lambda: "actor",
            get_worker_id=lambda: "worker",
        ),
    )
    monkeypatch.setattr(
        system_signal,
        "get_event_logger",
        lambda _source, _logs_dir: SimpleNamespace(
            error=lambda message, **fields: (
                events.append((message, fields)),
                delivered.set(),
            )
        ),
    )
    monkeypatch.setattr(system_signal.signal, "getsignal", lambda _signum: previous)
    monkeypatch.setattr(
        system_signal.signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )

    system_signal.install_system_signal_reporter(spmd_id="Train", rank=2)
    frame = object()
    installed[system_signal.signal.SIGTERM](system_signal.signal.SIGTERM, frame)
    assert delivered.wait(timeout=1)

    assert events == [
        (
            "Fluxway worker received system signal",
            {
                "diagnostic_kind": "system_signal",
                "job_id": "job",
                "actor_id": "actor",
                "worker_id": "worker",
                "spmd_id": "Train",
                "rank": "2",
                "pid": str(system_signal.os.getpid()),
                "node_id": "node",
                "signal_number": "15",
                "signal_name": "SIGTERM",
            },
        )
    ]
    assert previous_calls == [(system_signal.signal.SIGTERM, frame)]


def test_system_signal_reporter_preserves_ignored_signal(monkeypatch) -> None:
    installed: dict[int, object] = {}
    monkeypatch.setattr(
        system_signal,
        "global_worker",
        SimpleNamespace(
            node=SimpleNamespace(get_logs_dir_path=lambda: "/tmp/ray-logs"),
            _fluxway_system_signal_reporter_installed=False,
        ),
    )
    monkeypatch.setattr(
        system_signal.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "node",
            get_job_id=lambda: "job",
            get_actor_id=lambda: "actor",
            get_worker_id=lambda: "worker",
        ),
    )
    monkeypatch.setattr(
        system_signal,
        "get_event_logger",
        lambda _source, _logs_dir: SimpleNamespace(
            error=lambda *_args, **_kwargs: None
        ),
    )
    monkeypatch.setattr(
        system_signal.signal,
        "getsignal",
        lambda _signum: system_signal.signal.SIG_IGN,
    )
    monkeypatch.setattr(
        system_signal.signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )

    system_signal.install_system_signal_reporter(spmd_id="Train", rank=2)
    installed[system_signal.signal.SIGTERM](system_signal.signal.SIGTERM, None)


def test_system_signal_reporter_does_not_install_from_non_main_thread(
    monkeypatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        system_signal.signal,
        "signal",
        lambda signum, _handler: calls.append(signum),
    )

    thread = threading.Thread(
        target=system_signal.install_system_signal_reporter,
        kwargs={"spmd_id": "Train", "rank": 2},
    )
    thread.start()
    thread.join()

    assert calls == []


def test_signal_reporter_delegates_when_event_queue_is_full(monkeypatch) -> None:
    previous_calls: list[tuple[int, object]] = []
    installed: dict[int, object] = {}
    emitter = system_signal._SignalEventEmitter()
    logger_started = threading.Event()
    logger_release = threading.Event()

    def previous(signum: int, frame: object) -> None:
        previous_calls.append((signum, frame))

    monkeypatch.setattr(
        system_signal,
        "global_worker",
        SimpleNamespace(
            node=SimpleNamespace(get_logs_dir_path=lambda: "/tmp/ray-logs"),
            _fluxway_system_signal_reporter_installed=False,
            _fluxway_system_signal_emitter=emitter,
        ),
    )
    monkeypatch.setattr(
        system_signal.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "node",
            get_job_id=lambda: "job",
            get_actor_id=lambda: "actor",
            get_worker_id=lambda: "worker",
        ),
    )
    monkeypatch.setattr(
        system_signal,
        "get_event_logger",
        lambda _source, _logs_dir: SimpleNamespace(
            error=lambda *_args, **_kwargs: (
                logger_started.set(),
                logger_release.wait(timeout=1),
            )
        ),
    )
    monkeypatch.setattr(system_signal.signal, "getsignal", lambda _signum: previous)
    monkeypatch.setattr(
        system_signal.signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )

    system_signal.install_system_signal_reporter(spmd_id="Train", rank=2)
    frame = object()
    for _ in range(system_signal._EVENT_QUEUE_CAPACITY + 1):
        installed[system_signal.signal.SIGTERM](system_signal.signal.SIGTERM, frame)

    assert logger_started.wait(timeout=1)
    assert emitter.dropped_events >= 1
    assert len(previous_calls) == system_signal._EVENT_QUEUE_CAPACITY + 1
    logger_release.set()
