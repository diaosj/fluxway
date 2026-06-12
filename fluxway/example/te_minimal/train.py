import argparse
import asyncio

from fluxway import launch
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import AsyncStepConfig
from fluxway.control_plane.controller import create_controller
from fluxway.control_plane.graph.functional import FunctionalGraph

from .reader import TEMinimalReader
from .spmd import TEMinimalParallelSPMD, TEMinimalSPMD


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


async def _wait_for_step_target(controller, expected_ops, target_steps: int):
    last_progress = None
    while True:
        progress = await controller.get_step_progress_summary.remote()
        if progress != last_progress:
            print(f"[Fluxway] Step progress: {progress}", flush=True)
            last_progress = progress

        reached = True
        for spmd_id, op_names in expected_ops.items():
            op_progress = progress.get(spmd_id, {})
            for op_name in op_names:
                if op_progress.get(op_name, 0) < target_steps:
                    reached = False
                    break
            if not reached:
                break

        if reached:
            return progress

        await asyncio.sleep(1.0)


async def run_training(args):
    if args.precision != "bf16":
        raise RuntimeError("TE minimal example currently only supports --precision bf16.")
    if args.num_gpus != 1:
        raise RuntimeError(
            "--num-gpus is GPUs per Fluxway worker. Use --num-gpus 1 and set "
            "--dp/--tp to create multiple workers, e.g. --dp 2 --tp 4."
        )

    model_mesh = DeviceMesh(device_shape=(args.dp, args.tp), axis_names=("dp", "tp"))

    with FunctionalGraph("TE_Minimal_Training_Flow") as graph:
        model_cls = TEMinimalSPMD if args.dp * args.tp == 1 else TEMinimalParallelSPMD
        model = model_cls(
            name="TEMinimalModel",
            mesh=model_mesh,
            resource={"num_gpus": args.num_gpus, "num_cpus": 2, "memory_gb": 8},
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            batch_size=args.batch_size,
            lr=args.lr,
            precision=args.precision,
            dp_size=args.dp,
            tp_size=args.tp,
        )

        if args.dp * args.tp == 1:
            reader_mesh = DeviceMesh(device_shape=(1, 1), axis_names=("dp", "tp"))
            reader = TEMinimalReader(
                name="TEMinimalReader",
                mesh=reader_mesh,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                dp_size=1,
                tp_size=1,
            )
            h_batch = reader.read()
            model.step(h_batch)
        else:
            model.step_synthetic()

        task_graph = graph.compile()

    print(f"[Fluxway] Graph compiled. Nodes: {list(task_graph.nodes.keys())}")
    print(
        f"[Fluxway] TE minimal mesh: reader_dp={args.dp}, reader_tp={args.tp}, "
        f"model_dp={args.dp}, model_tp={args.tp}, model_workers={args.dp * args.tp}"
    )

    if args.compile_only:
        print("[Fluxway] Compile-only mode completed.")
        return

    expected_ops = {
        "TEMinimalModel": sorted(
            task_graph.nodes["TEMinimalModel"].spmd_config.operator_metas.keys()
        )
    }

    async_step_config = AsyncStepConfig(max_steps=args.max_steps)
    controller = await create_controller(
        task_graph,
        async_step_config=async_step_config,
        checkpoint_config=None,
    )
    await _print_resource_allocation_summary(controller)

    print(f"[Fluxway] Running to reach total {args.max_steps} steps ...")
    await _wait_for_step_target(controller, expected_ops, args.max_steps)
    await controller.stop.remote(save_checkpoint=False)
    print("[Fluxway] Training completed up to target steps.")


def main():
    parser = argparse.ArgumentParser(description="Fluxway TE Minimal Example")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
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
