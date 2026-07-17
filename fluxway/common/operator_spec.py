from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    field_validator,
    model_validator,
)

from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.operator_policy import (
    ArgumentType,
    AsyncPolicy,
    CompiledAsyncPolicy,
    PeriodicBarrierConfig,
    StandardBarrierConfig,
)
from fluxway.common.tree_spec import TreeSpec
from fluxway.common.types.sharding import ShardingConfig, ShardingSpec


type StringTupleMap = Annotated[
    Mapping[str, tuple[str, ...]],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]
type ShardingMap = Annotated[
    Mapping[str, ShardingSpec],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]
type TreeMap = Annotated[
    Mapping[str, TreeSpec],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]


class OperatorSpec(BaseModel):
    _runtime_policy: CompiledAsyncPolicy = PrivateAttr()

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    name: str = Field(min_length=1)
    owner: type | None = None
    method_name: str = Field(default="", min_length=1)
    requires_worker_process: bool = False
    is_source: bool | None = None
    is_interleaved: bool = False
    interleave_unit_size: int = Field(default=1, gt=0)
    async_policy: AsyncPolicy = Field(default_factory=AsyncPolicy.sync)
    input_arguments: StringTupleMap = Field(default_factory=dict)
    output_arguments: StringTupleMap = Field(default_factory=dict)
    input_sharding: ShardingMap = Field(default_factory=dict)
    output_sharding: ShardingMap = Field(default_factory=dict)
    input_param_map: StringTupleMap = Field(default_factory=dict)
    output_param_map: StringTupleMap = Field(default_factory=dict)
    input_treedefs: TreeMap = Field(default_factory=dict)
    output_treedef: TreeSpec | None = None
    barrier_config: StandardBarrierConfig | PeriodicBarrierConfig = Field(
        default_factory=StandardBarrierConfig
    )

    @model_validator(mode="before")
    @classmethod
    def default_method_name(cls, value: Any) -> Any:
        if (
            isinstance(value, Mapping)
            and not value.get("method_name")
            and value.get("name")
        ):
            return {**value, "method_name": value["name"]}
        return value

    @field_validator("owner")
    @classmethod
    def validate_owner(cls, value: type | None) -> type | None:
        if value is not None and not isinstance(value, type):
            raise TypeError("operator owner must be a class")
        return value

    @field_validator("input_arguments", "output_arguments", mode="before")
    @classmethod
    def validate_argument_map(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("input_sharding", "output_sharding", mode="before")
    @classmethod
    def validate_sharding_map(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            name: (
                spec
                if isinstance(spec, ShardingSpec)
                else (
                    ShardingSpec.model_validate(spec)
                    if isinstance(spec, Mapping) and "routing" in spec
                    else ShardingConfig.model_validate(spec).to_spec()
                )
            )
            for name, spec in value.items()
        }

    @field_validator("input_treedefs", mode="before")
    @classmethod
    def validate_input_trees(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            name: (
                tree
                if isinstance(tree, TreeSpec)
                else (
                    TreeSpec.model_validate(tree)
                    if isinstance(tree, Mapping) and "type" in tree
                    else TreeSpec.from_runtime(tree)
                )
            )
            for name, tree in value.items()
        }

    @field_validator("output_treedef", mode="before")
    @classmethod
    def validate_output_tree(cls, value: Any) -> Any:
        if value is None or isinstance(value, TreeSpec):
            return value
        if isinstance(value, Mapping) and "type" in value:
            return TreeSpec.model_validate(value)
        return TreeSpec.from_runtime(value)

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_runtime_policy", self.async_policy.compile())

    @property
    def runtime_policy(self) -> CompiledAsyncPolicy:
        return self._runtime_policy

    def get_arguments(self, is_output: bool, arg_type: ArgumentType) -> list[str]:
        arguments = self.output_arguments if is_output else self.input_arguments
        if arg_type.value in arguments:
            return list(arguments[arg_type.value])
        if arg_type == ArgumentType.DATA:
            sharding = self.output_sharding if is_output else self.input_sharding
            return list(sharding)
        return []

    def state_names(self, *, is_output: bool) -> list[str]:
        return self.get_arguments(
            is_output=is_output,
            arg_type=ArgumentType.CONTROL,
        )

    @property
    def owner_class_module(self) -> str:
        return "" if self.owner is None else self.owner.__module__

    @property
    def owner_class_qualname(self) -> str:
        return "" if self.owner is None else self.owner.__qualname__

    def updated(self, **updates: Any) -> OperatorSpec:
        values = {name: getattr(self, name) for name in self.__class__.model_fields}
        values.update(updates)
        return self.__class__.model_validate(values)

    def resolve_sharding_references(
        self,
        *,
        input_resolver: Any,
        output_resolver: Any,
        required: bool = True,
    ) -> OperatorSpec:
        input_sharding = {
            name: spec.resolved(input_resolver, required=required)
            for name, spec in self.input_sharding.items()
        }
        output_sharding = {
            name: spec.resolved(output_resolver, required=required)
            for name, spec in self.output_sharding.items()
        }
        if all(
            input_sharding[name] is spec for name, spec in self.input_sharding.items()
        ) and all(
            output_sharding[name] is spec for name, spec in self.output_sharding.items()
        ):
            return self
        return self.updated(
            input_sharding=input_sharding,
            output_sharding=output_sharding,
        )
