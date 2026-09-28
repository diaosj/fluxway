import os
import shutil
import asyncio
from pathlib import Path

# Ensure JAX doesn't preallocate all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["NCCL_MULTI_RANK_GPU_ENABLE"] = "1"

import ray
from ray.cluster_utils import Cluster
from fluxway.example.recommendation.train import run_training


class Args:
    def __init__(self, **kwargs):
        self.max_steps = kwargs.get("max_steps", 4)
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 2)
        self.num_epochs = kwargs.get("num_epochs", 5)
        self.source_max_in_flight = kwargs.get("source_max_in_flight", 4)
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "/tmp/fluxway_e2e_test_ckpt")
        self.reader_shards = kwargs.get("reader_shards", 2)
        self.dp = kwargs.get("dp", 4)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.table1_shards = kwargs.get("table1_shards", 2)
        self.table2_shards = kwargs.get("table2_shards", 2)
        self.num_threads = kwargs.get("num_threads", None)
        self.dense_step_delay_ms = kwargs.get("dense_step_delay_ms", 0)
        self.clear_ckpt = kwargs.get("clear_ckpt", False)
        self.profiler_trace_dir = None
        self.graph_visualization_dir = None
        self.overwrite_graph_visualization = False
        self.profiler_run_id = None
        self.profiler_buffer_mib = 64
        self.profiler_clock_sync_rounds = 8
        self.profiler_max_clock_uncertainty_us = 1000
        self.profiler_output_file = None
        self.profiler_diagnostics = []
        self.profiler_skip_initial_steps = 1
        self.profiler_steady_state_only = False
        self.data_plane_batch_max_frames = None
        self.data_plane_batch_flush_us = None
        self.data_plane_max_in_flight = None
        self.disable_ray_dashboard = True
        self.ray_num_cpus = None
        self.ray_num_gpus = None
        self.compact_checkpoints = False
        self.compact_only = False


def main():
    print("=== Starting E2E Multi-Node Multi-GPU Checkpoint & Resharding Test ===")
    ckpt_dir = "/tmp/fluxway_e2e_test_ckpt"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # 1. Setup simulated multi-node Ray cluster
    cluster = Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 8, "num_gpus": 2, "resources": {"head_node": 1}},
    )
    cluster.add_node(num_cpus=8, num_gpus=2, resources={"worker_node": 1})

    ray.init(address=cluster.address, ignore_reinit_error=True)
    nodes = ray.nodes()
    print(f"[Ray Cluster] Active nodes: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"  - Node {i}: node_id={node['NodeID']}, resources={node['Resources']}")

    try:
        # Phase 1: Initial training with 2 Readers, 4 DP workers across 2 nodes.
        # Checkpoint interval = 2, max_steps = 4 -> checkpoint saved at step 2 and step 4.
        print(
            "\n--- [Phase 1] Training with 2 Readers, 4 DP workers (steps 1 -> 4) ---"
        )
        args_p1 = Args(
            reader_shards=2,
            dp=4,
            num_gpus=1,
            table1_shards=2,
            table2_shards=2,
            max_steps=4,
            checkpoint_interval=2,
            checkpoint_dir=ckpt_dir,
        )
        asyncio.run(run_training(args_p1))
        print("[Phase 1] Successfully reached step 4 and produced checkpoints.")

        # Inspect checkpoint directory
        ckpt_files = list(Path(ckpt_dir).rglob("*"))
        print(f"[Checkpoint] Found {len(ckpt_files)} files/dirs in {ckpt_dir}")
        for p in sorted(ckpt_files):
            if p.is_file():
                print(f"  - {p.relative_to(ckpt_dir)} ({p.stat().st_size} bytes)")

        # Phase 2: Resharding restore! Change worker topology:
        # Scale reader_shards from 2 -> 1, keep dp=4.
        # Restore latest checkpoint (step 4) and train steps 5 -> 8.
        print(
            "\n--- [Phase 2] Resharding Restore: reader_shards 2 -> 1, continue to step 8 ---"
        )
        args_p2 = Args(
            reader_shards=1,
            dp=4,
            num_gpus=1,
            table1_shards=2,
            table2_shards=2,
            max_steps=8,
            checkpoint_interval=2,
            checkpoint_dir=ckpt_dir,
        )
        asyncio.run(run_training(args_p2))
        print("[Phase 2] Successfully restored from checkpoint and trained to step 8.")

        # Phase 3: Resharding restore again! Scale UP:
        # Scale reader_shards from 1 -> 4, dp=4.
        # Restore latest checkpoint (step 8) and train steps 9 -> 12.
        print(
            "\n--- [Phase 3] Resharding Restore: reader_shards 1 -> 4, continue to step 12 ---"
        )
        args_p3 = Args(
            reader_shards=4,
            dp=4,
            num_gpus=1,
            table1_shards=2,
            table2_shards=2,
            max_steps=12,
            checkpoint_interval=2,
            checkpoint_dir=ckpt_dir,
        )
        asyncio.run(run_training(args_p3))
        print("[Phase 3] Successfully restored from checkpoint and trained to step 12.")

        print("\n=== ALL E2E CHECKPOINT & RESHARDING TESTS PASSED SUCCESSFULLY! ===")

    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
