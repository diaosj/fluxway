import os

import pytest

from fluxway.common.ingestion import LayoutUnitDescriptor
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    TailLayoutMode,
    LayoutCompileRequest,
    AssignmentPolicy,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
)

# Suppress Ray warning regarding accelerator visible devices override on CPU-only runs
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


@pytest.fixture
def source_layout():
    def compile_layout(
        *,
        demands=(2, 2),
        cardinality=12,
        revision="r",
        epoch=0,
        start_step=0,
        terminal=True,
    ):
        return AlgebraicLayoutCompiler().compile(
            LayoutCompileRequest(
                source_identity="source",
                view_identity=revision,
                schema_digest="schema",
                lineage_digest=revision,
                units=(
                    LayoutUnitDescriptor(
                        "unit",
                        revision,
                        cardinality,
                        "rows",
                        "SplittableSpan",
                        "node",
                        "file",
                        0,
                        b"metadata",
                    ),
                ),
                topology=ReaderTopologyDescriptor(
                    sum(demands),
                    tuple(
                        ReaderGroupDemand(str(i), (RankDemand(str(i), n),))
                        for i, n in enumerate(demands)
                    ),
                ),
                tail_layout_mode=(
                    TailLayoutMode.REJECT_REMAINDER
                    if terminal
                    else TailLayoutMode.RETAIN_PENDING
                ),
                assignment_policy=AssignmentPolicy.default(),
            ),
            epoch=epoch,
            start_step=start_step,
        )

    return compile_layout
