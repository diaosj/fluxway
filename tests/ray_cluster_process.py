from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from multiprocessing import Pipe
from multiprocessing.connection import Connection


@contextmanager
def local_ray_cluster(**options):
    import ray

    if ray.is_initialized():
        raise RuntimeError("isolated cluster requires a disconnected Ray client")
    connection, child = Pipe()
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(child.fileno()),
                json.dumps(options),
            ],
            pass_fds=(child.fileno(),),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        child.close()
        ready = False
        try:
            if not connection.poll(90):
                raise TimeoutError("Ray cluster readiness timed out")
            response = connection.recv()
            if response["status"] != "ready":
                raise RuntimeError(response["error"])
            ready = True
            ray.init(address=response["address"], log_to_driver=False)
            yield
        finally:
            ray.shutdown()
            if ready and process.poll() is None:
                connection.send("stop")
            connection.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
            if process.returncode:
                output.seek(0)
                raise RuntimeError(
                    f"Ray cluster bootstrap failed ({process.returncode}): {output.read()}"
                )


def main():
    import ray

    with Connection(int(sys.argv[1])) as connection:
        try:
            context = ray.init(include_dashboard=False, **json.loads(sys.argv[2]))
        except Exception as error:
            connection.send({"status": "error", "error": repr(error)})
            raise
        try:
            connection.send(
                {"status": "ready", "address": context.address_info["gcs_address"]}
            )
            if connection.recv() != "stop":
                raise ValueError("invalid cluster shutdown command")
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
