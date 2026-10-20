from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import textwrap


def _wait_for_ready(child: subprocess.Popen[str]) -> None:
    assert child.stdout is not None
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        events = selector.select(timeout=15)
    assert events, "child did not reach the idle priority-thread boundary"
    assert child.stdout.readline() == "READY\n"


def test_sigterm_releases_idle_priority_threads_before_interpreter_join() -> None:
    fluxway_root = Path(__file__).parents[2]
    child_source = textwrap.dedent(
        f"""
        import signal
        import sys
        import threading
        import time

        sys.path.insert(0, {str(fluxway_root)!r})
        from fluxway.spmd.worker.runtime import _PriorityThreadExecutor

        worker_count = 4
        executor = _PriorityThreadExecutor(
            max_workers=worker_count,
            thread_name_prefix="test-priority-exit",
        )
        started = threading.Barrier(worker_count + 1)
        release = threading.Event()

        def occupy_worker():
            started.wait(timeout=2)
            release.wait(timeout=2)

        futures = [
            executor.submit((0, 0, index, "test"), occupy_worker)
            for index in range(worker_count)
        ]
        started.wait(timeout=2)
        release.set()
        for future in futures:
            future.result(timeout=2)
        time.sleep(0.05)
        print("READY", flush=True)

        def ray_style_sigterm(_signum, _frame):
            raise SystemExit(23)

        signal.signal(signal.SIGTERM, ray_style_sigterm)
        signal.pause()
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for_ready(child)
    os.kill(child.pid, signal.SIGTERM)
    stdout, stderr = child.communicate(timeout=1.5)
    assert child.returncode == 23, (stdout, stderr)
