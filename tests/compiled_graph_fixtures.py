from __future__ import annotations

from collections.abc import Mapping

from fluxway.common.compiled_graph import (
    DependencySpec,
    CompiledGraph,
    SPMDGraphSpec,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types.configs import SPMDConfig
from fluxway.resources import ResourceProfile


class CompiledGraphFixtureSPMD:
    pass


COMPILED_GRAPH_FIXTURE_PROFILE = ResourceProfile(base={"CPU": 1})


type SPMDDeclarations = Mapping[
    str,
    tuple[DeviceMesh, Mapping[str, OperatorSpec]],
]


def compiled_spmd_spec(
    spmd_id: str,
    mesh: DeviceMesh,
    operators: Mapping[str, OperatorSpec],
    *,
    spmd_class: type = CompiledGraphFixtureSPMD,
    resource_profile: ResourceProfile = COMPILED_GRAPH_FIXTURE_PROFILE,
) -> SPMDGraphSpec:
    return SPMDGraphSpec(
        spmd_id=spmd_id,
        spmd_class=spmd_class,
        mesh=mesh,
        resource_profile=resource_profile,
        operators=operators,
    )


def compile_test_graph(
    *,
    name: str,
    spmds: SPMDDeclarations,
    dependencies: tuple[DependencySpec, ...] = (),
) -> CompiledGraph:
    return compile_graph(
        name=name,
        spmds=tuple(
            compiled_spmd_spec(spmd_id, mesh, operators)
            for spmd_id, (mesh, operators) in spmds.items()
        ),
        dependencies=dependencies,
    )


def spmd_config_from_graph(
    plan: CompiledGraph,
    spmd_id: str,
) -> SPMDConfig:
    spmd_plan = plan.spmd(spmd_id)
    return SPMDConfig(
        spmd_id=spmd_id,
        graph_view=spmd_plan,
        mesh=spmd_plan.mesh,
        operator_specs={
            operator.key.op_id: operator.spec for operator in spmd_plan.operators
        },
    )


def single_spmd_config(
    *,
    name: str,
    spmd_id: str,
    mesh: DeviceMesh,
    operators: Mapping[str, OperatorSpec],
) -> SPMDConfig:
    plan = compile_test_graph(
        name=name,
        spmds={spmd_id: (mesh, operators)},
    )
    return spmd_config_from_graph(plan, spmd_id)
