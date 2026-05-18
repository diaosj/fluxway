"""Fluxway Benchmark: Multi-Backend Full-Dimension Comparison

Comprehensive comparison across 5 dimensions:
  1. CPU Performance:     JAX CPU (jax.jit) vs Hybrid CPU (@numba.jit)
  2. GPU Performance:     JAX GPU vs Hybrid (native+RT) vs CuPy GPU (all native)
  3. Dynamic Shape:       JAX recompilation vs Hybrid parameterized vs CuPy
  4. Pipeline:            argsort->take->bincount->cumsum end-to-end
  5. Cold Start:          First-call latency including compilation

Operations tested:
  - argsort, take(D=64/128), bincount, cumsum, repeat_interleave
  - N = [1024, 10240, 102400, 1048576]

Usage:
    python -m tests.benchmark.benchmark_backends

Environment:
    - GPU:  NVIDIA CUDA capable device (for GPU benchmarks)
    - Dependencies: numpy, jax, numba (required), cupy (optional)
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import sys
import subprocess

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SIZES = [1024, 10240, 102400, 1048576]
D_SIZES = [64, 128]
WORLD_SIZE = 32
WARMUP = 3
REPEATS = 10
SEED = 42
_MAX_MEMORY_MB = 8000  # 8 GB

_rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

HAS_JAX = False
HAS_JAX_GPU = False
HAS_NUMBA = False
HAS_CUPY = False

_jax = None
_jnp = None
_jax_gpu_dev = None
_jax_cpu_dev = None
_cp = None

try:
    import jax
    import jax.numpy as jnp

    _jax = jax
    _jnp = jnp
    HAS_JAX = True
    try:
        _jax_gpu_dev = jax.devices("gpu")[0]
        HAS_JAX_GPU = True
    except Exception:
        pass
    _jax_cpu_dev = jax.devices("cpu")[0]
except ImportError:
    pass

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    pass

try:
    import cupy as cp

    _cp = cp
    HAS_CUPY = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Early exit
# ---------------------------------------------------------------------------

if not HAS_JAX:
    print("ERROR: JAX is not installed. This benchmark requires JAX.")
    sys.exit(1)
if not HAS_NUMBA:
    print("ERROR: Numba is not installed. This benchmark requires Numba.")
    sys.exit(1)
if not HAS_JAX_GPU:
    print("ERROR: JAX GPU is not available. This benchmark requires GPU.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CuPy op availability (use subprocess to avoid SIGABRT from CUB/thrust)
# ---------------------------------------------------------------------------

_cupy_op_available = {}

if HAS_CUPY:
    for _op_name, _op_code in [
        (
            "argsort",
            "import cupy as cp,numpy as np; x=cp.asarray(np.array([2,0,1],dtype=np.int32)); cp.argsort(x); cp.cuda.Stream.null.synchronize()",
        ),
        (
            "take",
            "import cupy as cp,numpy as np; d=cp.asarray(np.random.randn(4,3).astype(np.float32)); i=cp.asarray(np.array([0,1],dtype=np.int32)); d[i]; cp.cuda.Stream.null.synchronize()",
        ),
        (
            "bincount",
            "import cupy as cp,numpy as np; x=cp.asarray(np.array([0,1,2,0],dtype=np.int32)); cp.bincount(x,minlength=4); cp.cuda.Stream.null.synchronize()",
        ),
        (
            "cumsum",
            "import cupy as cp,numpy as np; x=cp.asarray(np.array([1.,2.,3.],dtype=np.float32)); cp.cumsum(x); cp.cuda.Stream.null.synchronize()",
        ),
        (
            "repeat_interleave",
            "import cupy as cp,numpy as np; x=cp.asarray(np.array([1.,2.],dtype=np.float32)); r=cp.asarray(np.array([2,3],dtype=np.int32)); cp.repeat(x,r); cp.cuda.Stream.null.synchronize()",
        ),
    ]:
        _cupy_op_available[_op_name] = False
        try:
            _result = subprocess.run(
                [sys.executable, "-c", _op_code],
                capture_output=True,
                timeout=30,
            )
            if _result.returncode == 0:
                _cupy_op_available[_op_name] = True
        except Exception:
            pass

    # Pipeline needs argsort+take+bincount+cumsum
    _cupy_op_available["pipeline"] = (
        _cupy_op_available.get("argsort", False)
        and _cupy_op_available.get("take", False)
        and _cupy_op_available.get("bincount", False)
        and _cupy_op_available.get("cumsum", False)
    )

    def _cupy_pipeline(dest, data, world_size):
        """Pipeline: all native CuPy GPU ops."""
        sort_idx = _cp.argsort(dest)
        reordered = data[sort_idx]
        sorted_dest = dest[sort_idx]
        counts = _cp.bincount(sorted_dest.astype(_cp.int32), minlength=world_size)
        offsets = _cp.cumsum(counts)
        return reordered, counts, offsets


# ===========================================================================
# JAX JIT helpers
# ===========================================================================

_jit_argsort = _jax.jit(_jnp.argsort)
_jit_take = _jax.jit(lambda data, indices: data[indices])
_jit_cumsum = _jax.jit(lambda x: _jnp.cumsum(x, axis=0))


def _make_jit_bincount(length):
    return _jax.jit(lambda x: _jnp.bincount(x, length=length))


_jit_bincount_32 = _make_jit_bincount(WORLD_SIZE)


def _make_jit_ri(total):
    return _jax.jit(
        lambda x, repeats: _jnp.repeat(x, repeats, total_repeat_length=total)
    )


@_jax.jit(static_argnums=(2,))
def _jax_pipeline(data, dest, world_size):  # noqa
    sort_idx = _jnp.argsort(dest)
    reordered = _jnp.take(data, sort_idx, axis=0)
    sorted_dest = dest[sort_idx]
    counts = _jnp.bincount(sorted_dest, minlength=world_size, length=world_size)
    offsets = _jnp.cumsum(counts)
    return reordered, counts, offsets


# ===========================================================================
# Numba CPU helpers
# ===========================================================================


@numba.jit(nopython=True, cache=True)
def _nb_argsort_cpu(x):
    """Counting-sort based argsort for integer arrays (fast for small range)."""
    n = x.shape[0]
    max_val = 0
    for i in range(n):
        if x[i] > max_val:
            max_val = x[i]
    bc = np.zeros(max_val + 1, dtype=np.int32)
    for i in range(n):
        bc[x[i]] += 1
    offs = np.zeros(max_val + 1, dtype=np.int32)
    total = 0
    for v in range(max_val + 1):
        offs[v] = total
        total += bc[v]
    result = np.zeros(n, dtype=np.int32)
    tmp = offs.copy()
    for i in range(n):
        v = x[i]
        result[tmp[v]] = i
        tmp[v] += 1
    return result


@numba.jit(nopython=True, cache=True)
def _nb_take_cpu(data, indices):
    n = indices.shape[0]
    d = data.shape[1]
    out = np.empty((n, d), dtype=data.dtype)
    for i in range(n):
        idx = indices[i]
        for j in range(d):
            out[i, j] = data[idx, j]
    return out


@numba.jit(nopython=True, cache=True)
def _nb_bincount_cpu(x, minlength):
    out = np.zeros(minlength, dtype=np.int32)
    for i in range(x.shape[0]):
        v = x[i]
        if v < minlength:
            out[v] += 1
    return out


@numba.jit(nopython=True, cache=True)
def _nb_cumsum_cpu(x):
    n = x.shape[0]
    out = np.empty(n, dtype=x.dtype)
    s = np.float32(0)
    for i in range(n):
        s += x[i]
        out[i] = s
    return out


@numba.jit(nopython=True, cache=True)
def _nb_repeat_interleave_cpu(x, repeats, out):
    pos = 0
    for i in range(x.shape[0]):
        r = repeats[i]
        for _ in range(r):
            out[pos] = x[i]
            pos += 1
    return out


# ===========================================================================
# Utility functions
# ===========================================================================


def _estimate_memory_mb(n, d=None, elem_bytes=4):
    if d is not None:
        elems = 2 * n * d + n
    else:
        elems = 2 * n
    return elems * elem_bytes / (1024 * 1024)


def _fmt(v):
    if v is None:
        return ""
    if v >= 10000:
        return f"{v / 1000:.1f}s"
    if v >= 1000:
        return f"{v:.0f}ms"
    if v >= 1:
        return f"{v:.2f}ms"
    if v >= 0.001:
        return f"{v:.3f}ms"
    return f"{v:.4f}ms"


def _fmt_n(n):
    if n >= 1048576:
        return f"{n // 1048576}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


def _winner2(a_val, b_val, a_name="A", b_name="B"):
    if a_val is None or b_val is None:
        return "-"
    if a_val < b_val * 0.9:
        return a_name
    if b_val < a_val * 0.9:
        return b_name
    return "Tie"


def _best_backend(vals: dict):
    """Return the best backend name from a dict of {name: median_ms}."""
    # Filter out None values
    vals = {k: v for k, v in vals.items() if v is not None}
    if len(vals) < 2:
        return "-"
    sorted_v = sorted(vals.items(), key=lambda x: x[1])
    if sorted_v[0][1] < sorted_v[1][1] * 0.9:
        return sorted_v[0][0]
    return "Tie"


# ===========================================================================
# Benchmark runner
# ===========================================================================


def bench_fn(fn, warmup=WARMUP, repeats=REPEATS, sync_fn=None):
    """Benchmark a single function call. Returns (first_ms, median_ms)."""
    if sync_fn:
        sync_fn()
    t0 = time.perf_counter()
    result = fn()
    if sync_fn:
        sync_fn(result)
    first_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        result = fn()
        if sync_fn:
            sync_fn(result)

    times = []
    for _ in range(repeats):
        if sync_fn:
            sync_fn()
        t0 = time.perf_counter()
        result = fn()
        if sync_fn:
            sync_fn(result)
        times.append(time.perf_counter() - t0)

    return first_ms, np.median(times) * 1000


# ===========================================================================
# Section 1: CPU Performance
# ===========================================================================


def run_cpu_benchmarks():
    print("\n=== CPU Performance ===")
    print("-" * 90)
    print(
        f"{'Operation':<22} | {'N':>6} | {'JAX CPU':>12} | {'Hybrid CPU':>12} | {'Winner':>8}"
    )
    print("-" * 90)

    jax_dev = _jax_cpu_dev

    def jax_to(a):
        return _jax.device_put(np.asarray(a), jax_dev)

    def jax_sync(v=None):
        return v.block_until_ready() if v is not None else None

    # --- argsort ---
    for n in N_SIZES:
        x_np = _rng.integers(0, WORLD_SIZE, size=(n,), dtype=np.int32)
        x_jax = jax_to(x_np)
        _, jax_med = bench_fn(lambda _x=x_jax: _jit_argsort(_x), sync_fn=jax_sync)
        nb_first, nb_med = bench_fn(lambda _x=x_np: _nb_argsort_cpu(_x))
        w = _winner2(jax_med, nb_med, "JAX", "Numba")
        print(
            f"{'argsort':<22} | {_fmt_n(n):>6} | {_fmt(jax_med):>12} | {_fmt(nb_med):>12} | {w:>8}"
        )

    # --- take D=64, D=128 ---
    for d in D_SIZES:
        for n in N_SIZES:
            if _estimate_memory_mb(n, d) > _MAX_MEMORY_MB:
                continue
            data_np = _rng.random((n, d), dtype=np.float32)
            idx_np = _rng.integers(0, n, (n,), dtype=np.int32)
            data_jax = jax_to(data_np)
            idx_jax = jax_to(idx_np)
            _, jax_med = bench_fn(
                lambda _d=data_jax, _i=idx_jax: _jit_take(_d, _i), sync_fn=jax_sync
            )
            _, nb_med = bench_fn(lambda _d=data_np, _i=idx_np: _nb_take_cpu(_d, _i))
            w = _winner2(jax_med, nb_med, "JAX", "Numba")
            print(
                f"{'take D=' + str(d):<22} | {_fmt_n(n):>6} | {_fmt(jax_med):>12} | {_fmt(nb_med):>12} | {w:>8}"
            )

    # --- bincount ---
    for n in N_SIZES:
        x_np = _rng.integers(0, WORLD_SIZE, size=(n,), dtype=np.int32)
        x_jax = jax_to(x_np)
        _, jax_med = bench_fn(lambda _x=x_jax: _jit_bincount_32(_x), sync_fn=jax_sync)
        _, nb_med = bench_fn(lambda _x=x_np: _nb_bincount_cpu(_x, WORLD_SIZE))
        w = _winner2(jax_med, nb_med, "JAX", "Numba")
        print(
            f"{'bincount':<22} | {_fmt_n(n):>6} | {_fmt(jax_med):>12} | {_fmt(nb_med):>12} | {w:>8}"
        )

    # --- cumsum ---
    for n in N_SIZES:
        x_np = _rng.random((n,), dtype=np.float32)
        x_jax = jax_to(x_np)
        _, jax_med = bench_fn(lambda _x=x_jax: _jit_cumsum(_x), sync_fn=jax_sync)
        _, nb_med = bench_fn(lambda _x=x_np: _nb_cumsum_cpu(_x))
        w = _winner2(jax_med, nb_med, "JAX", "Numba")
        print(
            f"{'cumsum':<22} | {_fmt_n(n):>6} | {_fmt(jax_med):>12} | {_fmt(nb_med):>12} | {w:>8}"
        )

    # --- repeat_interleave ---
    for n in N_SIZES:
        x_np = _rng.random((n,), dtype=np.float32)
        rep_np = _rng.integers(1, 5, (n,), dtype=np.int32)
        total = int(rep_np.sum())
        x_jax = jax_to(x_np)
        rep_jax = jax_to(rep_np)
        jax_ri = _make_jit_ri(total)
        _, jax_med = bench_fn(
            lambda _ri=jax_ri, _x=x_jax, _r=rep_jax: _ri(_x, _r), sync_fn=jax_sync
        )
        out_np = np.empty(total, dtype=np.float32)
        _, nb_med = bench_fn(
            lambda _x=x_np, _r=rep_np, _o=out_np: _nb_repeat_interleave_cpu(_x, _r, _o)
        )
        w = _winner2(jax_med, nb_med, "JAX", "Numba")
        print(
            f"{'repeat_interleave':<22} | {_fmt_n(n):>6} | {_fmt(jax_med):>12} | {_fmt(nb_med):>12} | {w:>8}"
        )


# ===========================================================================
# Main
# ===========================================================================


def run_all():
    print("=" * 80)
    print("Multi-Backend CPU Comparison Benchmark")
    print("=" * 80)
    print(f"\nN sizes: {N_SIZES}")
    print(f"D sizes: {D_SIZES}  (take only)")
    print(f"world_size: {WORLD_SIZE}")
    print(f"warmup: {WARMUP}, repeats: {REPEATS}")
    print("\n--- Backend Availability ---")
    print("  JAX CPU:      available")
    print(f"  Hybrid CPU:   {'available' if HAS_NUMBA else 'NOT AVAILABLE'}")

    run_cpu_benchmarks()

    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
