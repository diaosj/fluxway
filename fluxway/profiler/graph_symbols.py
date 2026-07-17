from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fluxway.common.compiled_graph import CompiledGraph, CompiledOperator, OperatorKey
from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.operator_policy import ArgumentType
from fluxway.profiler.errors import (
    GraphTraceSymbolConflict,
    MissingGraphTraceSymbol,
)


type ParameterPaths = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class OperatorTraceSymbol:
    key: OperatorKey
    function_id: str
    input_data_ports: tuple[str, ...]
    input_control_ports: tuple[str, ...]
    input_static_arguments: tuple[str, ...]
    output_data_ports: tuple[str, ...]
    output_control_ports: tuple[str, ...]
    output_static_arguments: tuple[str, ...]
    input_parameter_paths: ParameterPaths
    output_parameter_paths: ParameterPaths

    @property
    def qualified_id(self) -> str:
        return f"{self.key.spmd_id}.{self.key.op_id}"


@dataclass(frozen=True, slots=True)
class LogicalTensorTraceSymbol:
    logical_tuuid: int
    source_operator: OperatorKey
    source_port: str

    @property
    def producer(self) -> str:
        return (
            f"{self.source_operator.spmd_id}.{self.source_operator.op_id}:"
            f"{self.source_port}"
        )


@dataclass(frozen=True, slots=True)
class TransferTraceSymbol:
    transfer_code: int
    logical_tuuid: int
    source_operator: OperatorKey
    source_port: str
    target_operator: OperatorKey
    target_port: str


@dataclass(frozen=True, slots=True)
class GraphTraceSymbols:
    graph_id: str
    generation: int
    operators: Mapping[OperatorKey, OperatorTraceSymbol]
    transfers: Mapping[int, TransferTraceSymbol]
    logical_tensors: Mapping[int, LogicalTensorTraceSymbol]
    _operators_by_spmd_and_id: Mapping[tuple[str, str], OperatorTraceSymbol] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("GraphTraceSymbols requires a non-empty graph_id.")
        if self.generation < 0:
            raise ValueError("GraphTraceSymbols generation must be non-negative.")
        for key, symbol in self.operators.items():
            if key != symbol.key:
                raise GraphTraceSymbolConflict(
                    symbol_kind="operator",
                    identity=key,
                    detail=f"mapping key differs from symbol key {symbol.key!r}",
                )
        for transfer_code, symbol in self.transfers.items():
            if transfer_code != symbol.transfer_code:
                raise GraphTraceSymbolConflict(
                    symbol_kind="transfer",
                    identity=transfer_code,
                    detail=(
                        "mapping key differs from symbol transfer code "
                        f"{symbol.transfer_code}"
                    ),
                )
        for logical_tuuid, symbol in self.logical_tensors.items():
            if logical_tuuid != symbol.logical_tuuid:
                raise GraphTraceSymbolConflict(
                    symbol_kind="logical_tensor",
                    identity=logical_tuuid,
                    detail=(
                        "mapping key differs from symbol logical TUUID "
                        f"{symbol.logical_tuuid}"
                    ),
                )
        _validate_symbol_references(
            operators=self.operators,
            transfers=self.transfers,
            logical_tensors=self.logical_tensors,
        )
        operators = freeze_mapping(self.operators)
        transfers = freeze_mapping(self.transfers)
        logical_tensors = freeze_mapping(self.logical_tensors)
        operator_index = freeze_mapping(
            {(key.spmd_id, key.op_id): symbol for key, symbol in operators.items()}
        )
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "transfers", transfers)
        object.__setattr__(self, "logical_tensors", logical_tensors)
        object.__setattr__(self, "_operators_by_spmd_and_id", operator_index)

    @classmethod
    def from_compiled_graph(cls, graph: CompiledGraph) -> GraphTraceSymbols:
        operators: dict[OperatorKey, OperatorTraceSymbol] = {}
        logical_tensors: dict[int, LogicalTensorTraceSymbol] = {}
        for spmd in graph.spmds:
            for compiled_operator in spmd.operators:
                symbol = _operator_symbol(compiled_operator)
                _insert_unique(
                    operators,
                    compiled_operator.key,
                    symbol,
                    symbol_kind="operator",
                )
                output_ports = symbol.output_data_ports
                output_uuids = compiled_operator.output_uuids
                if len(output_ports) != len(output_uuids):
                    raise GraphTraceSymbolConflict(
                        symbol_kind="logical_tensor",
                        identity=compiled_operator.key,
                        detail=(
                            f"output port count {len(output_ports)} differs from "
                            f"logical tensor count {len(output_uuids)}"
                        ),
                    )
                for logical_tuuid, source_port in zip(
                    output_uuids,
                    output_ports,
                    strict=True,
                ):
                    _insert_unique(
                        logical_tensors,
                        logical_tuuid,
                        LogicalTensorTraceSymbol(
                            logical_tuuid=logical_tuuid,
                            source_operator=compiled_operator.key,
                            source_port=source_port,
                        ),
                        symbol_kind="logical_tensor",
                    )

        transfers: dict[int, TransferTraceSymbol] = {}
        for dependency in graph.dependencies:
            for transfer in dependency.transfers:
                _insert_unique(
                    transfers,
                    transfer.transfer_code,
                    TransferTraceSymbol(
                        transfer_code=transfer.transfer_code,
                        logical_tuuid=transfer.logical_tuuid,
                        source_operator=dependency.source_operator.key,
                        source_port=transfer.spec.source.name,
                        target_operator=dependency.target_operator.key,
                        target_port=transfer.spec.target.name,
                    ),
                    symbol_kind="transfer",
                )
        return cls(
            graph_id=graph.graph_id,
            generation=graph.generation,
            operators=operators,
            transfers=transfers,
            logical_tensors=logical_tensors,
        )

    def operator(self, key: OperatorKey) -> OperatorTraceSymbol:
        try:
            return self.operators[key]
        except KeyError as exc:
            raise MissingGraphTraceSymbol(
                symbol_kind="operator",
                identity=(key.spmd_id, key.op_id),
            ) from exc

    def operator_by_id(self, spmd_id: str, op_id: str) -> OperatorTraceSymbol:
        try:
            return self._operators_by_spmd_and_id[(spmd_id, op_id)]
        except KeyError as exc:
            raise MissingGraphTraceSymbol(
                symbol_kind="operator",
                identity=(spmd_id, op_id),
            ) from exc

    def transfer(self, transfer_code: int) -> TransferTraceSymbol:
        try:
            return self.transfers[transfer_code]
        except KeyError as exc:
            raise MissingGraphTraceSymbol(
                symbol_kind="transfer",
                identity=transfer_code,
            ) from exc

    def tensor(self, logical_tuuid: int) -> LogicalTensorTraceSymbol:
        try:
            return self.logical_tensors[logical_tuuid]
        except KeyError as exc:
            raise MissingGraphTraceSymbol(
                symbol_kind="logical_tensor",
                identity=logical_tuuid,
            ) from exc


def _operator_symbol(compiled_operator: CompiledOperator) -> OperatorTraceSymbol:
    return OperatorTraceSymbol(
        key=compiled_operator.key,
        function_id=compiled_operator.function_id,
        input_data_ports=tuple(
            compiled_operator.get_arguments(False, ArgumentType.DATA)
        ),
        input_control_ports=tuple(
            compiled_operator.get_arguments(False, ArgumentType.CONTROL)
        ),
        input_static_arguments=tuple(
            compiled_operator.get_arguments(False, ArgumentType.STATIC)
        ),
        output_data_ports=tuple(
            compiled_operator.get_arguments(True, ArgumentType.DATA)
        ),
        output_control_ports=tuple(
            compiled_operator.get_arguments(True, ArgumentType.CONTROL)
        ),
        output_static_arguments=tuple(
            compiled_operator.get_arguments(True, ArgumentType.STATIC)
        ),
        input_parameter_paths=_parameter_paths(compiled_operator.spec.input_param_map),
        output_parameter_paths=_parameter_paths(
            compiled_operator.spec.output_param_map
        ),
    )


def _parameter_paths(mapping: Mapping[str, tuple[str, ...]]) -> ParameterPaths:
    return tuple((name, tuple(paths)) for name, paths in mapping.items())


def _validate_symbol_references(
    *,
    operators: Mapping[OperatorKey, OperatorTraceSymbol],
    transfers: Mapping[int, TransferTraceSymbol],
    logical_tensors: Mapping[int, LogicalTensorTraceSymbol],
) -> None:
    for logical_tuuid, tensor in logical_tensors.items():
        source = operators.get(tensor.source_operator)
        if source is None:
            raise GraphTraceSymbolConflict(
                symbol_kind="logical_tensor",
                identity=logical_tuuid,
                detail=f"unknown source operator {tensor.source_operator!r}",
            )
        if tensor.source_port not in source.output_data_ports:
            raise GraphTraceSymbolConflict(
                symbol_kind="logical_tensor",
                identity=logical_tuuid,
                detail=f"unknown source output port {tensor.source_port!r}",
            )
    for transfer_code, transfer in transfers.items():
        source = operators.get(transfer.source_operator)
        target = operators.get(transfer.target_operator)
        if source is None:
            raise GraphTraceSymbolConflict(
                symbol_kind="transfer",
                identity=transfer_code,
                detail=f"unknown source operator {transfer.source_operator!r}",
            )
        if target is None:
            raise GraphTraceSymbolConflict(
                symbol_kind="transfer",
                identity=transfer_code,
                detail=f"unknown target operator {transfer.target_operator!r}",
            )
        if transfer.source_port not in source.output_data_ports:
            raise GraphTraceSymbolConflict(
                symbol_kind="transfer",
                identity=transfer_code,
                detail=f"unknown source output port {transfer.source_port!r}",
            )
        if transfer.target_port not in target.input_data_ports:
            raise GraphTraceSymbolConflict(
                symbol_kind="transfer",
                identity=transfer_code,
                detail=f"unknown target input port {transfer.target_port!r}",
            )
        tensor = logical_tensors.get(transfer.logical_tuuid)
        if tensor is None:
            raise GraphTraceSymbolConflict(
                symbol_kind="transfer",
                identity=transfer_code,
                detail=f"unknown logical TUUID {transfer.logical_tuuid}",
            )
        if (
            tensor.source_operator != transfer.source_operator
            or tensor.source_port != transfer.source_port
        ):
            raise GraphTraceSymbolConflict(
                symbol_kind="transfer",
                identity=transfer_code,
                detail="source endpoint differs from logical tensor producer",
            )


def _insert_unique[K, V](
    mapping: dict[K, V],
    identity: K,
    symbol: V,
    *,
    symbol_kind: str,
) -> None:
    existing = mapping.get(identity)
    if existing is not None and existing != symbol:
        raise GraphTraceSymbolConflict(
            symbol_kind=symbol_kind,
            identity=identity,
            detail="identity maps to multiple symbols",
        )
    mapping[identity] = symbol


def format_parameter_paths(paths: ParameterPaths) -> str:
    return "; ".join(
        f"{parameter} -> [{', '.join(field_paths)}]" for parameter, field_paths in paths
    )
