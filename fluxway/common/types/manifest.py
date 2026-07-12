from __future__ import annotations

from hashlib import sha256
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator


ArgumentKind = Literal["data", "control", "static"]


class ManifestSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OperatorArguments(ManifestSchema):
    data: tuple[str, ...] = ()
    control: tuple[str, ...] = ()
    static: tuple[str, ...] = ()

    @property
    def signature_id(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(payload.encode("utf-8")).hexdigest()


class OperatorFacts(ManifestSchema):
    spmd_id: str
    op_id: str
    function_id: str
    inputs: OperatorArguments
    outputs: OperatorArguments
    input_signature_id: str
    output_signature_id: str

    @model_validator(mode="after")
    def validate_signatures(self) -> OperatorFacts:
        if self.input_signature_id != self.inputs.signature_id:
            raise ValueError("input_signature_id does not match operator inputs.")
        if self.output_signature_id != self.outputs.signature_id:
            raise ValueError("output_signature_id does not match operator outputs.")
        return self


class ManifestOperatorFacts(ManifestSchema):
    manifest_id: str
    operator: OperatorFacts


class ResourceFacts(ManifestSchema):
    num_cpus: float
    num_gpus: int
    memory_gb: float
    count: int


class SPMDFacts(ManifestSchema):
    _operators_by_id: dict[str, OperatorFacts] = PrivateAttr(default_factory=dict)

    spmd_id: str
    class_id: str
    description: str
    mesh: tuple[tuple[str, int], ...]
    resource: ResourceFacts | None
    priority: str | None
    placement_strategy: str | None
    operators: tuple[OperatorFacts, ...]

    @model_validator(mode="after")
    def index_operators(self) -> SPMDFacts:
        operators: dict[str, OperatorFacts] = {}
        for operator in self.operators:
            if operator.spmd_id != self.spmd_id:
                raise ValueError(
                    f"Operator {operator.op_id!r} has spmd_id "
                    f"{operator.spmd_id!r}, expected {self.spmd_id!r}."
                )
            if operator.op_id in operators:
                raise ValueError(f"Duplicate operator ID: {operator.op_id!r}.")
            operators[operator.op_id] = operator
        self._operators_by_id.clear()
        self._operators_by_id.update(operators)
        return self

    def operator_facts(self, op_id: str) -> OperatorFacts:
        try:
            return self._operators_by_id[op_id]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self.spmd_id!r} has no operator {op_id!r}."
            ) from exc


class OperatorPort(ManifestSchema):
    spmd_id: str
    op_id: str
    name: str
    kind: ArgumentKind = "data"


class TransferFacts(ManifestSchema):
    source: OperatorPort
    target: OperatorPort
    tuuid: int
    local_only: bool = False
    condition_id: str | None = None
    symmetric_to: tuple[str, str, str, str] | None = None


class ComputationGraphManifest(ManifestSchema):
    _runtime_facts_by_identity: dict[tuple[str, str], ManifestOperatorFacts] = (
        PrivateAttr(default_factory=dict)
    )

    schema_version: Literal[1] = 1
    manifest_id: str
    graph_name: str
    spmds: tuple[SPMDFacts, ...]
    transfers: tuple[TransferFacts, ...]

    @classmethod
    def create(
        cls,
        *,
        graph_name: str,
        spmds: tuple[SPMDFacts, ...],
        transfers: tuple[TransferFacts, ...],
    ) -> ComputationGraphManifest:
        return cls(
            manifest_id=cls._content_id(
                graph_name=graph_name,
                spmds=spmds,
                transfers=transfers,
            ),
            graph_name=graph_name,
            spmds=spmds,
            transfers=transfers,
        )

    @classmethod
    def _content_id(
        cls,
        *,
        graph_name: str,
        spmds: tuple[SPMDFacts, ...],
        transfers: tuple[TransferFacts, ...],
    ) -> str:
        content = {
            "schema_version": 1,
            "graph_name": graph_name,
            "spmds": [item.model_dump(mode="json") for item in spmds],
            "transfers": [item.model_dump(mode="json") for item in transfers],
        }
        encoded = json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @model_validator(mode="after")
    def validate_integrity(self) -> ComputationGraphManifest:
        expected_id = self._content_id(
            graph_name=self.graph_name,
            spmds=self.spmds,
            transfers=self.transfers,
        )
        if self.manifest_id != expected_id:
            raise ValueError("manifest_id does not match manifest content.")

        operators: dict[tuple[str, str], OperatorFacts] = {}
        spmd_ids: set[str] = set()
        for spmd in self.spmds:
            if spmd.spmd_id in spmd_ids:
                raise ValueError(f"Duplicate spmd_id: {spmd.spmd_id!r}.")
            spmd_ids.add(spmd.spmd_id)
            for operator in spmd.operators:
                key = (operator.spmd_id, operator.op_id)
                operators[key] = operator

        for transfer in self.transfers:
            self._validate_port(transfer.source, operators, output=True)
            self._validate_port(transfer.target, operators, output=False)
        self._runtime_facts_by_identity.clear()
        self._runtime_facts_by_identity.update(
            {
                identity: ManifestOperatorFacts(
                    manifest_id=self.manifest_id,
                    operator=operator,
                )
                for identity, operator in operators.items()
            }
        )
        return self

    def runtime_operator_facts(self, spmd_id: str, op_id: str) -> ManifestOperatorFacts:
        identity = (spmd_id, op_id)
        try:
            return self._runtime_facts_by_identity[identity]
        except KeyError as exc:
            raise ValueError(f"Manifest has no operator {identity!r}.") from exc

    @staticmethod
    def _validate_port(
        port: OperatorPort,
        operators: dict[tuple[str, str], OperatorFacts],
        *,
        output: bool,
    ) -> None:
        key = (port.spmd_id, port.op_id)
        operator = operators.get(key)
        if operator is None:
            raise ValueError(f"Transfer references unknown operator: {key!r}.")
        arguments = operator.outputs if output else operator.inputs
        names = getattr(arguments, port.kind)
        if port.name not in names:
            direction = "output" if output else "input"
            raise ValueError(
                f"Transfer references unknown {port.kind} {direction} "
                f"{port.name!r} on operator {key!r}."
            )
