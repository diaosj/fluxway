import argparse

from fluxway import launch
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import AsyncStepConfig
from fluxway.control_plane.controller import create_controller
from fluxway.control_plane.graph.functional import FunctionalGraph

from .reader import TEMinimalReader
from .spmd import TEMinimalSPMD


async def _print_resource_allocation_summary(controller):
    summary = await controller.get_resource_allocation_summary.remote()
    if not summary:
        print("[Fluxway] Controller resource summary is empty.")
        return

    print("[Fluxway] Controller resource summary:")
    for spmd_id in sorted(summary):
        item = summary[spmd_id]
        print(
            f"  - {spmd_id}: bundles={item['bundle_count']} "
            f"workers={item['worker_count']} total_cpu={item['total_cpu']} "
            f"total_memory_gb={item['total_memory_gb']} "
            f"driver={item['driver_bundle']} worker={item['worker_bundle']}"
        )


async def run_training(args):
    if args.precision != "bf16":
        raise RuntimeError("TE minimal MVP only supports --precision bf16.")

    mesh = DeviceMesh(device_shape=(1,), axis_names=("data",))

    with FunctionalGraph("TE_Minimal_Training_Flow") as graph:
        reader = TEMinimalReader(
            name="TEMinimalReader",
            mesh=mesh,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
        )
        model = TEMinimalSPMD(
            name="TEMinimalModel",
            mesh=mesh,
            resource={"num_gpus": args.num_gpus, "num_cpus": 2, "memory_gb": 8},
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            lr=args.lr,
            precision=args.precision,
        )

        h_batch = reader.read()
        model.step(h_batch)

        task_graph = graph.compile()

    print(f"[Fluxway] Graph compiled. Nodes: {list(task_graph.nodes.keys())}")

    if args.compile_only:
        print("[Fluxway] Compile-only mode completed.")
        return

    async_step_config = AsyncStepConfig(max_steps=args.max_steps)
    controller = await create_controller(
        task_graph,
        async_step_config=async_step_config,
        checkpoint_config=None,
    )
    await _print_resource_allocation_summary(controller)

    print(f"[Fluxway] Running to reach total {args.max_steps} steps ...")
    completed = await controller.wait_for_completion.remote()

    if completed:
        print("[Fluxway] Training completed up to target steps.")
    else:
        print("[Fluxway] Training failed or timed out.")


def main():
    parser = argparse.ArgumentParser(description="Fluxway TE Minimal Example")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ffn-hidden-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", choices=["bf16"], default="bf16")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    launch(run_training, args)


if __name__ == "__main__":
    main()

