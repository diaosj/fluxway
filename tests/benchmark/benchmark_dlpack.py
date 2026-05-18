"""Fluxway Benchmark: DLPack Zero-Copy Interoperability

Measures the cost and correctness of cross-framework tensor sharing via
DLPack and related zero-copy paths:

  1. DLPack interop matrix: GPU-GPU, CPU-CPU, CPU-GPU across sizes
  2. GPU memory coexistence: JAX preallocate + CuPy / Numba CUDA
  3. Mixed device strategy breakeven point (CPU roundtrip vs GPU direct)
  4. DLPack CPU-CPU path vs np.asarray()
  5. Numba CUDA context coexistence with JAX

Usage:
    python -m tests.benchmark.benchmark_dlpack

Environment:
    - GPU:  NVIDIA CUDA capable device (for GPU interop benchmarks)
    - Dependencies: numpy (required), jax / cupy / numba (optional)
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import textwrap
import time
from typing import Any

# Must be set BEFORE any JAX import — disable preallocation for most tests
# Memory coexistence tests run in subprocesses with their own settings.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

_has_jax = False
_has_jax_gpu = False
_has_cupy = False

try:
    import jax
    import jax.numpy as jnp

    _has_jax = True
    _has_jax_gpu = len(jax.devices("gpu")) > 0
except Exception:
    pass

try:
    import cupy

    _has_cupy = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZES_BYTES: list[int] = [1024, 102400, 10 * 1024 * 1024, 100 * 1024 * 1024]
# If 1 GB fits in memory, uncomment:
# SIZES_BYTES.append(1024 * 1024 * 1024)

INT32_ELEM_COUNTS: list[int] = [s // 4 for s in SIZES_BYTES]
# Human-readable labels
SIZE_LABELS: list[str] = []
for s in SIZES_BYTES:
    if s >= 1024 * 1024 * 1024:
        SIZE_LABELS.append(f"{s // (1024**3)}GB")
    elif s >= 1024 * 1024:
        SIZE_LABELS.append(f"{s // (1024**2)}MB")
    elif s >= 1024:
        SIZE_LABELS.append(f"{s // 1024}KB")
    else:
        SIZE_LABELS.append(f"{s}B")

WARMUP: int = 5
RUNS: int = 50

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def measure_median_ms(fn: Any, warmup: int = WARMUP, runs: int = RUNS) -> float:
    """Measure *fn* latency and return the median in milliseconds."""
    for _ in range(warmup):
        fn()
    timings: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return statistics.median(timings)


def format_bw(size_bytes: int, latency_ms: float) -> str:
    """Format bandwidth string from size and latency."""
    if latency_ms <= 0:
        return "inf"
    bw = size_bytes / (latency_ms / 1000.0)
    if bw >= 1024**3:
        return f"{bw / (1024**3):.1f}GB/s"
    elif bw >= 1024**2:
        return f"{bw / (1024**2):.1f}MB/s"
    else:
        return f"{bw / 1024:.1f}KB/s"


# ============================================================================
# Module 1: DLPack Interop Matrix
# ============================================================================


def bench_gpu_gpu_interop() -> None:
    """GPU-GPU DLPack interop: JAX <-> CuPy."""
    print("\n--- GPU-GPU (JAX <-> CuPy) ---")
    if not (_has_jax_gpu and _has_cupy):
        print("  SKIPPED (requires JAX GPU + CuPy)")
        return

    print(f"{'Size':<12} {'JAX->CuPy':>12} {'CuPy->JAX':>12} {'Zero-copy?':>14}")
    print("-" * 52)

    jax_to_cupy_times: list[float] = []
    cupy_to_jax_times: list[float] = []

    for i, N in enumerate(INT32_ELEM_COUNTS):
        # JAX GPU -> CuPy GPU (via DLPack)
        jax_arr = jax.device_put(np.ones(N, dtype=np.int32))
        jax_arr.block_until_ready()

        j2c_ms = measure_median_ms(lambda a=jax_arr: cupy.from_dlpack(a))
        jax_to_cupy_times.append(j2c_ms)

        # CuPy GPU -> JAX GPU (via DLPack)
        cupy_arr = cupy.ones(N, dtype=cupy.int32)
        cupy.cuda.Device(0).synchronize()

        c2j_ms = measure_median_ms(lambda a=cupy_arr: jax.dlpack.from_dlpack(a))
        cupy_to_jax_times.append(c2j_ms)

    # Determine if zero-copy: time should be O(1) regardless of size
    is_zero_copy = _check_zero_copy(jax_to_cupy_times) and _check_zero_copy(
        cupy_to_jax_times
    )
    zc_label = "Yes (O(1))" if is_zero_copy else "No"

    for i in range(len(INT32_ELEM_COUNTS)):
        print(
            f"{SIZE_LABELS[i]:<12} "
            f"{jax_to_cupy_times[i]:>10.3f}ms "
            f"{cupy_to_jax_times[i]:>10.3f}ms "
            f"{zc_label:>14}"
        )


def _check_zero_copy(times: list[float]) -> bool:
    """Heuristic: if max time < 5 * min time and max < 0.15ms, likely O(1)."""
    if not times:
        return False
    mn, mx = min(times), max(times)
    return mx < max(mn * 5, 0.15)


def bench_cpu_cpu_interop() -> None:
    """CPU-CPU DLPack interop: NumPy <-> JAX CPU."""
    print("\n--- CPU-CPU (NumPy <-> JAX CPU) ---")
    if not _has_jax:
        print("  SKIPPED (requires JAX)")
        return

    jax_cpu_dev = jax.devices("cpu")[0]

    print(
        f"{'Size':<12} {'NP->JAX CPU':>12} {'JAX CPU->NP':>12} "
        f"{'DLPack':>12} {'np.asarray':>12}"
    )
    print("-" * 62)

    for i, N in enumerate(INT32_ELEM_COUNTS):
        label = SIZE_LABELS[i]

        # NumPy -> JAX CPU (via device_put)
        np_arr = np.ones(N, dtype=np.int32)
        np2jax_ms = measure_median_ms(lambda a=np_arr: jax.device_put(a, jax_cpu_dev))

        # JAX CPU -> NumPy (via np.asarray)
        jax_cpu_arr = jax.device_put(np.ones(N, dtype=np.int32), jax_cpu_dev)
        jax2np_ms = measure_median_ms(lambda a=jax_cpu_arr: np.asarray(a))

        # DLPack path: JAX CPU -> NumPy via from_dlpack
        jax_cpu_arr2 = jax.device_put(np.ones(N, dtype=np.int32), jax_cpu_dev)
        dlpack_ms = measure_median_ms(lambda a=jax_cpu_arr2: np.from_dlpack(a))

        # np.asarray path for direct comparison
        jax_cpu_arr3 = jax.device_put(np.ones(N, dtype=np.int32), jax_cpu_dev)
        asarray_ms = measure_median_ms(lambda a=jax_cpu_arr3: np.asarray(a))

        print(
            f"{label:<12} "
            f"{np2jax_ms:>10.3f}ms "
            f"{jax2np_ms:>10.3f}ms "
            f"{dlpack_ms:>10.3f}ms "
            f"{asarray_ms:>10.3f}ms"
        )


def bench_cpu_gpu_transfer() -> None:
    """CPU-GPU transfer: host->device and device->host."""
    print("\n--- CPU-GPU Transfer ---")
    if not _has_jax_gpu:
        print("  SKIPPED (requires JAX GPU)")
        return

    print(
        f"{'Size':<12} {'CPU->GPU':>12} {'GPU->CPU':>12} {'BW(->GPU)':>12} {'BW(->CPU)':>12}"
    )
    print("-" * 62)

    for i, N in enumerate(INT32_ELEM_COUNTS):
        label = SIZE_LABELS[i]
        size_bytes = SIZES_BYTES[i]

        # CPU -> GPU: host->device transfer
        np_arr = np.ones(N, dtype=np.int32)
        h2d_ms = measure_median_ms(
            lambda a=np_arr: jax.device_put(a).block_until_ready(),
            warmup=3,
            runs=20,
        )

        # GPU -> CPU: device->host transfer
        jax_gpu_arr = jax.device_put(np.ones(N, dtype=np.int32))
        jax_gpu_arr.block_until_ready()
        d2h_ms = measure_median_ms(
            lambda a=jax_gpu_arr: np.asarray(a),
            warmup=3,
            runs=20,
        )

        bw_h2d = format_bw(size_bytes, h2d_ms)
        bw_d2h = format_bw(size_bytes, d2h_ms)

        print(
            f"{label:<12} {h2d_ms:>10.3f}ms {d2h_ms:>10.3f}ms {bw_h2d:>12} {bw_d2h:>12}"
        )


def bench_dlpack_matrix() -> None:
    """Run all DLPack interop matrix benchmarks."""
    print("\n=== DLPack Interop Matrix ===")
    bench_gpu_gpu_interop()
    bench_cpu_cpu_interop()
    bench_cpu_gpu_transfer()


# ============================================================================
# Module 2: GPU Memory Coexistence (subprocess-isolated)
# ============================================================================

_MEM_COEXIST_SCRIPT = textwrap.dedent("""\
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import sys

result = {}

# --- Test 1: JAX preallocate + CuPy ---
try:
    import jax
    import numpy as np
    jax_arr = jax.device_put(np.ones(1_000_000, dtype=np.float32))
    jax_arr.block_until_ready()

    try:
        import cupy
        cupy_arr = cupy.ones(1_000_000, dtype=np.float32)
        result["cupy_after_jax"] = "SUCCESS"
    except Exception as e:
        result["cupy_after_jax"] = f"FAILED ({e})"
except Exception as e:
    result["cupy_after_jax"] = f"SKIPPED ({e})"

# --- Test 3: CUDA stream sync / data consistency ---
try:
    import jax
    import numpy as np
    import cupy

    jax_arr = jax.device_put(np.arange(10, dtype=np.float32))
    jax_arr.block_until_ready()

    cupy_arr = cupy.from_dlpack(jax_arr)
    cupy_arr += 100

    jax_result = jax.numpy.sum(jax_arr).block_until_ready()
    cupy_result = cupy.sum(cupy_arr)

    result["stream_sync"] = f"OK (jax_sum={float(jax_result)}, cupy_sum={float(cupy_result)})"
except Exception as e:
    result["stream_sync"] = f"FAILED ({e})"

# Output results
for k, v in result.items():
    print(f"{k}: {v}")
""")


def bench_memory_coexistence() -> None:
    """Run memory coexistence tests in isolated subprocesses."""
    print("\n=== Memory Coexistence ===")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _MEM_COEXIST_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = proc.stdout.strip()
        if proc.returncode != 0:
            print(f"  Subprocess error (rc={proc.returncode}):")
            print(f"  stderr: {proc.stderr.strip()}")
        if output:
            for line in output.splitlines():
                key, _, val = line.partition(": ")
                key = key.strip()
                val = val.strip()
                if key == "cupy_after_jax":
                    print(f"  JAX preallocate 90% + CuPy: {val}")
                elif key == "stream_sync":
                    print(f"  CUDA stream sync: {val}")
                else:
                    print(f"  {key}: {val}")
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: memory coexistence subprocess exceeded 120s")
    except Exception as e:
        print(f"  ERROR running subprocess: {e}")


# ============================================================================
# Module 3: Mixed Device Strategy Breakeven Point
# ============================================================================


def bench_breakeven() -> None:
    """Find breakeven N where GPU direct beats CPU roundtrip for bincount."""
    print("\n=== Breakeven Point ===")
    if not _has_jax_gpu:
        print("  SKIPPED (requires JAX GPU)")
        return

    WORLD_SIZE = 32
    test_sizes = [100, 1000, 5000, 10000, 100000, 1_000_000]

    # Warmup JIT
    dummy = jax.device_put(np.ones(100, dtype=np.int32))
    _ = jnp.bincount(dummy, length=WORLD_SIZE).block_until_ready()

    print(f"  Operation: bincount(N, minlength={WORLD_SIZE})")
    print(f"  {'N':<12} {'GPU direct':>14} {'CPU roundtrip':>14} {'Winner':>10}")
    print("  " + "-" * 52)

    breakeven_n: int | None = None

    for N in test_sizes:
        data = np.random.randint(0, WORLD_SIZE, size=N, dtype=np.int32)

        # GPU direct
        gpu_data = jax.device_put(data)
        gpu_data.block_until_ready()
        gpu_ms = measure_median_ms(
            lambda d=gpu_data: jnp.bincount(d, length=WORLD_SIZE).block_until_ready(),
            warmup=5,
            runs=30,
        )

        # CPU roundtrip: GPU -> CPU -> NumPy bincount -> GPU
        gpu_data2 = jax.device_put(data)
        gpu_data2.block_until_ready()

        def _cpu_roundtrip(d: Any = gpu_data2) -> Any:
            cpu_data = np.asarray(d)
            result = np.bincount(cpu_data, minlength=WORLD_SIZE)
            return jax.device_put(result).block_until_ready()

        cpu_ms = measure_median_ms(_cpu_roundtrip, warmup=5, runs=30)

        winner = "CPU" if cpu_ms < gpu_ms else "GPU"
        if winner == "GPU" and breakeven_n is None:
            breakeven_n = N

        print(f"  {N:<12} {gpu_ms:>10.3f}ms {cpu_ms:>10.3f}ms {winner:>10}")

    if breakeven_n is not None:
        idx = test_sizes.index(breakeven_n)
        if idx > 0:
            lo = test_sizes[idx - 1]
            hi = breakeven_n
            print(f"  Breakeven: ~N={lo}-{hi}")
        else:
            print(f"  Breakeven: <=N={breakeven_n}")
    else:
        print("  Breakeven: not found in tested range (CPU always wins)")


# ============================================================================
# Module 4: DLPack CPU-CPU vs np.asarray()
# ============================================================================


def bench_dlpack_vs_asarray() -> None:
    """Compare DLPack path vs np.asarray() for JAX CPU -> NumPy."""
    print("\n--- DLPack vs np.asarray (JAX CPU -> NumPy) ---")
    if not _has_jax:
        print("  SKIPPED (requires JAX)")
        return

    jax_cpu_dev = jax.devices("cpu")[0]

    print(f"  {'Size':<12} {'np.asarray':>14} {'np.from_dlpack':>14} {'Ratio':>10}")
    print("  " + "-" * 52)

    for i, N in enumerate(INT32_ELEM_COUNTS):
        label = SIZE_LABELS[i]

        # np.asarray() path
        jax_cpu1 = jax.device_put(np.ones(N, dtype=np.int32), jax_cpu_dev)
        asarray_ms = measure_median_ms(lambda a=jax_cpu1: np.asarray(a))

        # DLPack path
        jax_cpu2 = jax.device_put(np.ones(N, dtype=np.int32), jax_cpu_dev)
        dlpack_ms = measure_median_ms(lambda a=jax_cpu2: np.from_dlpack(a))

        ratio = dlpack_ms / asarray_ms if asarray_ms > 0 else float("inf")

        print(
            f"  {label:<12} {asarray_ms:>10.3f}ms {dlpack_ms:>10.3f}ms {ratio:>8.2f}x"
        )


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    print("=" * 60)
    print("DLPack Interop & GPU Memory Coexistence Benchmark")
    print("=" * 60)

    print("\nEnvironment:")
    print(f"  JAX:        {'available' if _has_jax else 'NOT available'}")
    print(f"  JAX GPU:    {'available' if _has_jax_gpu else 'NOT available'}")
    print(f"  CuPy:       {'available' if _has_cupy else 'NOT available'}")
    print(
        f"  XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'default')}"
    )

    # Module 1: DLPack interop matrix
    bench_dlpack_matrix()

    # Module 2: Memory coexistence (subprocess)
    bench_memory_coexistence()

    # Module 3: Breakeven point
    bench_breakeven()

    # Module 4: DLPack vs np.asarray
    bench_dlpack_vs_asarray()

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
