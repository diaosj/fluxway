"""Fluxway Benchmark: Core Tensor Operations

Systematic benchmark across operation x data size x backend x device (CPU/GPU).
Tests the six core communication-layer operations — argsort, take, bincount,
repeat_interleave, cumsum, and their pipeline composition — on every available
backend (JAX CPU/GPU, NumPy, Numba CPU, CuPy, Numba CUDA).

Results guide the per-operation backend dispatch table in
``fluxway.common.backend``.

Usage:
    python -m tests.benchmark.benchmark_ops [options]

Environment:
    - GPU:  NVIDIA CUDA capable device (for GPU backends)
    - Dependencies: numpy (required), jax / numba / cupy (optional)
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import importlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np


# =============================================================================
# Constants
# =============================================================================

DATA_SIZES: list[int] = [1024, 10240, 102400, 1048576, 10485760]
OPERATIONS: list[str] = [
    "argsort",
    "take",
    "bincount",
    "repeat_interleave",
    "cumsum",
    "pipeline",
]
DEFAULT_WORLD_SIZE: int = 8
DEFAULT_D: int = 64  # feature dim for take


# =============================================================================
# Data structures
# =============================================================================


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration for a single benchmark run."""

    op_name: str
    backend_name: str
    device: str  # "cpu" or "gpu"
    size: int
    world_size: int


@dataclasses.dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    config: BenchmarkConfig
    first_run_latency_ms: float
    steady_state_throughput_elems_per_sec: float
    peak_gpu_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.config.op_name,
            "backend": self.config.backend_name,
            "device": self.config.device,
            "size": self.config.size,
            "world_size": self.config.world_size,
            "first_run_latency_ms": self.first_run_latency_ms,
            "steady_state_throughput_elems_per_sec": (
                self.steady_state_throughput_elems_per_sec
            ),
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }


# =============================================================================
# Timing utilities
# =============================================================================


@contextlib.contextmanager
def benchmark_context(warmup: int = 0, runs: int = 10) -> Iterator[list[float]]:
    """Context manager that records execution times over multiple runs.

    Args:
        warmup: Number of warmup iterations (not included in results).
        runs: Number of measured iterations.

    Yields:
        A list that will be populated with measured latencies in seconds.
    """
    timings: list[float] = []

    class _Recorder:
        def __init__(self, timing_list: list[float]):
            self._timings = timing_list
            self._warmup = warmup
            self._runs = runs
            self._count = 0

        def record(self, fn: Callable[[], Any]) -> Any:
            """Execute *fn* and record its latency."""
            for _ in range(self._warmup):
                fn()
            for _ in range(self._runs):
                t0 = time.perf_counter()
                result = fn()
                t1 = time.perf_counter()
                self._timings.append(t1 - t0)
                self._count += 1
            return result

    recorder = _Recorder(timings)
    yield recorder

    if not timings:
        raise RuntimeError("No timings recorded.")


def benchmark(
    func: Callable | None = None,
    *,
    warmup: int = 0,
    runs: int = 10,
) -> Callable:
    """Decorator that benchmarks a callable.

    The decorated function must accept no arguments. If the underlying
    operation needs inputs, wrap it in a closure before decorating.
    """

    def decorator(f: Callable[[], Any]) -> Callable[[], list[float]]:
        @functools.wraps(f)
        def wrapper() -> list[float]:
            # Warmup
            for _ in range(warmup):
                f()
            timings: list[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                f()
                t1 = time.perf_counter()
                timings.append(t1 - t0)
            return timings

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def median_throughput(timings_sec: Sequence[float], num_elements: int) -> float:
    """Return median throughput in elements/second."""
    if not timings_sec:
        return 0.0
    median_latency = statistics.median(timings_sec)
    if median_latency <= 0:
        return float("inf")
    return num_elements / median_latency


# =============================================================================
# GPU memory measurement
# =============================================================================


def measure_gpu_memory() -> float | None:
    """Measure peak GPU memory usage in MiB.

    Tries multiple strategies in order:
      1. nvidia-ml-py / pynvml
      2. PyTorch torch.cuda.max_memory_allocated
      3. JAX device memory stats

    Returns None if no GPU measurement strategy succeeds.
    """
    # Strategy 1: NVML
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        pass

    # Strategy 2: PyTorch
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    except Exception:
        pass

    # Strategy 3: JAX
    try:
        jax = importlib.import_module("jax")
        for d in jax.devices():
            if d.platform == "gpu":
                stats = d.memory_stats()
                if stats:
                    return stats.get("peak_bytes_in_use", 0) / (1024 * 1024)
    except Exception:
        pass

    return None


def reset_gpu_memory_stats() -> None:
    """Reset peak GPU memory statistics if supported."""
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass

    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
    except Exception:
        pass


# =============================================================================
# Input generators
# =============================================================================


def generate_argsort_input(
    N: int, world_size: int, rng: np.random.Generator
) -> tuple[Any, ...]:
    """Generate input for argsort: x shape=[N], dtype=int32, values in [0, ws)."""
    x = rng.integers(0, world_size, size=(N,), dtype=np.int32)
    return (x,)


def generate_take_input(N: int, D: int, rng: np.random.Generator) -> tuple[Any, ...]:
    """Generate input for take: data shape=[N, D], indices shape=[N]."""
    data = rng.random(size=(N, D), dtype=np.float32)
    indices = rng.integers(0, N, size=(N,), dtype=np.int32)
    return (data, indices)


def generate_bincount_input(
    N: int, world_size: int, rng: np.random.Generator
) -> tuple[Any, ...]:
    """Generate input for bincount: x shape=[N], minlength=world_size."""
    x = rng.integers(0, world_size, size=(N,), dtype=np.int32)
    return (x, world_size)


def generate_repeat_interleave_input(
    N: int, rng: np.random.Generator
) -> tuple[Any, ...]:
    """Generate input for repeat_interleave: x shape=[N], repeats shape=[N]."""
    x = rng.random(size=(N,), dtype=np.float32)
    # repeats in [1, 4] so total length is bounded
    repeats = rng.integers(1, 5, size=(N,), dtype=np.int32)
    total = int(repeats.sum())
    return (x, repeats, total)


def generate_cumsum_input(world_size: int, rng: np.random.Generator) -> tuple[Any, ...]:
    """Generate input for cumsum: x shape=[world_size]."""
    x = rng.random(size=(world_size,), dtype=np.float32)
    return (x, 0)


# =============================================================================
# Backend abstraction
# =============================================================================


class BenchmarkBackend:
    """Abstract base for a benchmark backend."""

    name: str
    device: str  # "cpu" or "gpu"
    warmup_runs: int = 0
    measure_runs: int = 10

    def is_available(self) -> bool:
        return True

    def synchronize(self, value: Any | None = None) -> None:
        """Block until all async work completes (e.g. CUDA kernels).

        Args:
            value: Optional array/value to wait on (used by JAX).
        """
        pass

    def to_backend_array(self, arr: np.ndarray) -> Any:
        """Convert a NumPy array to the backend's native array type."""
        return arr

    def argsort(self, x: Any) -> Any:
        raise NotImplementedError(f"{self.name}.argsort")

    def take(self, data: Any, indices: Any) -> Any:
        raise NotImplementedError(f"{self.name}.take")

    def bincount(self, x: Any, minlength: int) -> Any:
        raise NotImplementedError(f"{self.name}.bincount")

    def repeat_interleave(self, x: Any, repeats: Any, total_repeat_length: int) -> Any:
        raise NotImplementedError(f"{self.name}.repeat_interleave")

    def cumsum(self, x: Any, dim: int) -> Any:
        raise NotImplementedError(f"{self.name}.cumsum")

    def pipeline(
        self,
        x: Any,
        data: Any,
        world_size: int,
    ) -> Any:
        """argsort -> take -> bincount -> cumsum pipeline."""
        raise NotImplementedError(f"{self.name}.pipeline")


# ---------------------------------------------------------------------------
# JAX CPU
# ---------------------------------------------------------------------------


class JAXCPUBackend(BenchmarkBackend):
    name = "jax_cpu"
    device = "cpu"
    warmup_runs = 3
    measure_runs = 10

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self._cpu = jax.devices("cpu")[0]

    def is_available(self) -> bool:
        try:
            import jax

            jax.devices("cpu")
            return True
        except Exception:
            return False

    def synchronize(self, value: Any | None = None) -> None:
        self._jax.block_until_ready(value)

    def to_backend_array(self, arr: np.ndarray) -> Any:
        return self._jax.device_put(self._jnp.asarray(arr), self._cpu)

    def argsort(self, x: Any) -> Any:
        return self._jnp.argsort(x)

    def take(self, data: Any, indices: Any) -> Any:
        return data[indices]

    def bincount(self, x: Any, minlength: int) -> Any:
        return self._jnp.bincount(x, length=minlength)

    def repeat_interleave(self, x: Any, repeats: Any, total_repeat_length: int) -> Any:
        return self._jnp.repeat(x, repeats, total_repeat_length=total_repeat_length)

    def cumsum(self, x: Any, dim: int) -> Any:
        return self._jnp.cumsum(x, axis=dim)

    def pipeline(self, x: Any, data: Any, world_size: int) -> Any:
        order = self._jnp.argsort(x)
        gathered = data[order]
        flat = gathered.reshape(-1)
        counts = self._jnp.bincount(
            self._jnp.astype(flat, self._jnp.int32), length=world_size
        )
        return self._jnp.cumsum(counts)


# ---------------------------------------------------------------------------
# JAX GPU
# ---------------------------------------------------------------------------


class JAXGPUBackend(BenchmarkBackend):
    name = "jax_gpu"
    device = "gpu"
    warmup_runs = 3
    measure_runs = 10

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("No GPU devices available for JAX.")
        self._gpu = gpus[0]

    def is_available(self) -> bool:
        try:
            import jax

            return len(jax.devices("gpu")) > 0
        except Exception:
            return False

    def synchronize(self, value: Any | None = None) -> None:
        self._jax.block_until_ready(value)

    def to_backend_array(self, arr: np.ndarray) -> Any:
        return self._jax.device_put(self._jnp.asarray(arr), self._gpu)

    def argsort(self, x: Any) -> Any:
        return self._jnp.argsort(x)

    def take(self, data: Any, indices: Any) -> Any:
        return data[indices]

    def bincount(self, x: Any, minlength: int) -> Any:
        return self._jnp.bincount(x, length=minlength)

    def repeat_interleave(self, x: Any, repeats: Any, total_repeat_length: int) -> Any:
        return self._jnp.repeat(x, repeats, total_repeat_length=total_repeat_length)

    def cumsum(self, x: Any, dim: int) -> Any:
        return self._jnp.cumsum(x, axis=dim)

    def pipeline(self, x: Any, data: Any, world_size: int) -> Any:
        order = self._jnp.argsort(x)
        gathered = data[order]
        flat = gathered.reshape(-1)
        counts = self._jnp.bincount(
            self._jnp.astype(flat, self._jnp.int32), length=world_size
        )
        return self._jnp.cumsum(counts)


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------


class NumPyBackend(BenchmarkBackend):
    name = "numpy"
    device = "cpu"
    warmup_runs = 0
    measure_runs = 10

    def to_backend_array(self, arr: np.ndarray) -> Any:
        return arr

    def argsort(self, x: Any) -> Any:
        return np.argsort(x)

    def take(self, data: Any, indices: Any) -> Any:
        return data[indices]

    def bincount(self, x: Any, minlength: int) -> Any:
        return np.bincount(x, minlength=minlength)

    def repeat_interleave(self, x: Any, repeats: Any, total_repeat_length: int) -> Any:
        return np.repeat(x, repeats)

    def cumsum(self, x: Any, dim: int) -> Any:
        return np.cumsum(x, axis=dim)

    def pipeline(self, x: Any, data: Any, world_size: int) -> Any:
        order = np.argsort(x)
        gathered = data[order]
        flat = gathered.reshape(-1)
        counts = np.bincount(flat.astype(np.int32), minlength=world_size)
        return np.cumsum(counts)


# ---------------------------------------------------------------------------
# Numba CPU (placeholder)
# ---------------------------------------------------------------------------


class NumbaCPUBackend(BenchmarkBackend):
    name = "numba_cpu"
    device = "cpu"
    warmup_runs = 3
    measure_runs = 10

    def is_available(self) -> bool:
        try:
            import numba  # noqa: F401

            return True
        except ImportError:
            return False

    def to_backend_array(self, arr: np.ndarray) -> Any:
        return arr

    def argsort(self, x: Any) -> Any:
        raise NotImplementedError("numba_cpu.argsort not yet implemented")

    def take(self, data: Any, indices: Any) -> Any:
        raise NotImplementedError("numba_cpu.take not yet implemented")

    def bincount(self, x: Any, minlength: int) -> Any:
        raise NotImplementedError("numba_cpu.bincount not yet implemented")

    def repeat_interleave(self, x: Any, repeats: Any, total_repeat_length: int) -> Any:
        raise NotImplementedError("numba_cpu.repeat_interleave not yet implemented")

    def cumsum(self, x: Any, dim: int) -> Any:
        raise NotImplementedError("numba_cpu.cumsum not yet implemented")

    def pipeline(self, x: Any, data: Any, world_size: int) -> Any:
        raise NotImplementedError("numba_cpu.pipeline not yet implemented")


# ---------------------------------------------------------------------------
# CuPy (placeholder)
# ---------------------------------------------------------------------------


class CuPyBenchBackend(BenchmarkBackend):
    name = "cupy"
    device = "gpu"
    warmup_runs = 0
    measure_runs = 10

    def is_available(self) -> bool:
        try:
            import cupy  # noqa: F401

            return True
        except ImportError:
            return False

    def synchronize(self) -> None:
        import cupy

        cupy.cuda.Device(0).synchronize()

    def to_backend_array(self, arr: np.ndarray) -> Any:
        import cupy

        return cupy.asarray(arr)

    def argsort(self, x: Any) -> Any:
        raise NotImplementedError("cupy.argsort not yet implemented")

    def take(self, data: Any, indices: Any) -> Any:
        raise NotImplementedError("cupy.take not yet implemented")

    def bincount(self, x: Any, minlength: int) -> Any:
        raise NotImplementedError("cupy.bincount not yet implemented")

    def repeat_interleave(self, x: Any, repeats: Any, total_repeat_length: int) -> Any:
        raise NotImplementedError("cupy.repeat_interleave not yet implemented")

    def cumsum(self, x: Any, dim: int) -> Any:
        raise NotImplementedError("cupy.cumsum not yet implemented")

    def pipeline(self, x: Any, data: Any, world_size: int) -> Any:
        raise NotImplementedError("cupy.pipeline not yet implemented")


# =============================================================================
# BenchmarkRunner
# =============================================================================


class BenchmarkRunner:
    """Orchestrates registration and execution of backend benchmarks."""

    def __init__(self) -> None:
        self._backends: dict[str, type[BenchmarkBackend]] = {}
        self._op_gens: dict[str, Callable] = {
            "argsort": generate_argsort_input,
            "take": generate_take_input,
            "bincount": generate_bincount_input,
            "repeat_interleave": generate_repeat_interleave_input,
            "cumsum": generate_cumsum_input,
        }

    def register_backend(self, backend_cls: type[BenchmarkBackend]) -> None:
        inst = backend_cls()
        self._backends[inst.name] = backend_cls

    def list_available_backends(self) -> list[str]:
        return [name for name, cls in self._backends.items() if cls().is_available()]

    def _run_single(
        self,
        backend: BenchmarkBackend,
        op_name: str,
        size: int,
        world_size: int,
        rng: np.random.Generator,
    ) -> BenchmarkResult:
        # Build inputs
        if op_name == "argsort":
            inputs = generate_argsort_input(size, world_size, rng)
            backend_inputs = tuple(backend.to_backend_array(arr) for arr in inputs)

            def fn():
                return backend.argsort(backend_inputs[0])

        elif op_name == "take":
            inputs = generate_take_input(size, DEFAULT_D, rng)
            backend_inputs = tuple(backend.to_backend_array(arr) for arr in inputs)

            def fn():
                return backend.take(backend_inputs[0], backend_inputs[1])

        elif op_name == "bincount":
            inputs = generate_bincount_input(size, world_size, rng)
            backend_inputs = tuple(
                backend.to_backend_array(arr) if isinstance(arr, np.ndarray) else arr
                for arr in inputs
            )

            def fn():
                return backend.bincount(backend_inputs[0], backend_inputs[1])

        elif op_name == "repeat_interleave":
            inputs = generate_repeat_interleave_input(size, rng)
            backend_inputs = tuple(
                backend.to_backend_array(arr) if isinstance(arr, np.ndarray) else arr
                for arr in inputs
            )

            def fn():
                return backend.repeat_interleave(
                    backend_inputs[0],
                    backend_inputs[1],
                    backend_inputs[2],
                )

        elif op_name == "cumsum":
            inputs = generate_cumsum_input(world_size, rng)
            backend_inputs = tuple(
                backend.to_backend_array(arr) if isinstance(arr, np.ndarray) else arr
                for arr in inputs
            )

            def fn():
                return backend.cumsum(backend_inputs[0], backend_inputs[1])

        elif op_name == "pipeline":
            # Pipeline needs argsort input (x) and take data
            argsort_in = generate_argsort_input(size, world_size, rng)
            take_in = generate_take_input(size, DEFAULT_D, rng)
            x = backend.to_backend_array(argsort_in[0])
            data = backend.to_backend_array(take_in[0])

            def fn():
                return backend.pipeline(x, data, world_size)

        else:
            raise ValueError(f"Unknown operation: {op_name}")

        # Ensure any async transfer is done before timing
        backend.synchronize()

        # First-run latency (includes JIT/compilation if any)
        reset_gpu_memory_stats()
        t0 = time.perf_counter()
        result = fn()
        backend.synchronize(result)
        t1 = time.perf_counter()
        first_latency_ms = (t1 - t0) * 1000.0

        # Steady-state throughput
        with benchmark_context(
            warmup=backend.warmup_runs, runs=backend.measure_runs
        ) as recorder:
            recorder.record(lambda: backend.synchronize(fn()))

        peak_mem = measure_gpu_memory() if backend.device == "gpu" else None

        config = BenchmarkConfig(
            op_name=op_name,
            backend_name=backend.name,
            device=backend.device,
            size=size,
            world_size=world_size,
        )
        throughput = median_throughput(recorder._timings, size)

        return BenchmarkResult(
            config=config,
            first_run_latency_ms=first_latency_ms,
            steady_state_throughput_elems_per_sec=throughput,
            peak_gpu_memory_mb=peak_mem,
        )

    def run(
        self,
        backends: Sequence[str] | None = None,
        ops: Sequence[str] | None = None,
        sizes: Sequence[int] | None = None,
        world_size: int = DEFAULT_WORLD_SIZE,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run the benchmark matrix and return results."""
        if backends is None:
            backends = self.list_available_backends()
        if ops is None:
            ops = OPERATIONS.copy()
        if sizes is None:
            sizes = DATA_SIZES.copy()

        rng = np.random.default_rng(seed)
        results: list[BenchmarkResult] = []

        for backend_name in backends:
            backend_cls = self._backends.get(backend_name)
            if backend_cls is None:
                print(
                    f"[WARN] Backend '{backend_name}' not registered, skipping.",
                    file=sys.stderr,
                )
                continue
            backend = backend_cls()
            if not backend.is_available():
                print(
                    f"[WARN] Backend '{backend_name}' not available, skipping.",
                    file=sys.stderr,
                )
                continue

            for op_name in ops:
                if op_name not in OPERATIONS:
                    print(
                        f"[WARN] Operation '{op_name}' unknown, skipping.",
                        file=sys.stderr,
                    )
                    continue
                for size in sizes:
                    print(
                        f"Running {op_name} on {backend_name} "
                        f"(size={size}, ws={world_size}) ..."
                    )
                    try:
                        result = self._run_single(
                            backend, op_name, size, world_size, rng
                        )
                        results.append(result)
                    except NotImplementedError as e:
                        print(
                            f"[SKIP] {op_name}/{backend_name}: {e}",
                            file=sys.stderr,
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] {op_name}/{backend_name}: {e}",
                            file=sys.stderr,
                        )

        return results


# =============================================================================
# CLI
# =============================================================================


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark communication backends")
    parser.add_argument(
        "--backends",
        type=_parse_str_list,
        default=None,
        help="Comma-separated backend names (default: all available)",
    )
    parser.add_argument(
        "--ops",
        type=_parse_str_list,
        default=None,
        help=f"Comma-separated ops (default: all). Choices: {OPERATIONS}",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_int_list,
        default=None,
        help=f"Comma-separated sizes (default: {DATA_SIZES})",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=DEFAULT_WORLD_SIZE,
        help="world_size parameter for relevant ops",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for input generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List registered backends and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runner = BenchmarkRunner()
    runner.register_backend(JAXCPUBackend)
    runner.register_backend(JAXGPUBackend)
    runner.register_backend(NumPyBackend)
    runner.register_backend(NumbaCPUBackend)
    runner.register_backend(CuPyBenchBackend)

    if args.list_backends:
        print("Registered backends:")
        for name in sorted(runner._backends.keys()):
            avail = (
                "available"
                if runner._backends[name]().is_available()
                else "not available"
            )
            print(f"  - {name} ({avail})")
        return 0

    results = runner.run(
        backends=args.backends,
        ops=args.ops,
        sizes=args.sizes,
        world_size=args.world_size,
        seed=args.seed,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(
        f"{'Op':<18} {'Backend':<12} {'Device':<6} "
        f"{'Size':>10} {'1st(ms)':>12} {'Throughput':>16} {'GPU(MB)':>10}"
    )
    print("-" * 80)
    for r in results:
        gpu_str = (
            f"{r.peak_gpu_memory_mb:.1f}" if r.peak_gpu_memory_mb is not None else "N/A"
        )
        print(
            f"{r.config.op_name:<18} {r.config.backend_name:<12} "
            f"{r.config.device:<6} {r.config.size:>10} "
            f"{r.first_run_latency_ms:>12.3f} "
            f"{r.steady_state_throughput_elems_per_sec:>16,.0f} "
            f"{gpu_str:>10}"
        )
    print("=" * 80)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
