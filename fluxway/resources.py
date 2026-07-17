from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, DecimalException
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
)


class FrozenResourceVector(Mapping[str, Decimal]):
    def __init__(self, value: Mapping[str, Decimal]) -> None:
        self._items = tuple(sorted(value.items()))
        self._values = dict(self._items)

    def __getitem__(self, key: str) -> Decimal:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenResourceVector:
        return self

    def __reduce__(self):
        return FrozenResourceVector, (dict(self._items),)


def exact_decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, float):
        raise ValueError(f"{label} must use int or Decimal, not float")
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
        raise ValueError(f"{label} must use int or Decimal")
    try:
        decimal = Decimal(value)
    except DecimalException as error:
        raise ValueError(f"{label} must be a finite decimal number") from error
    if not decimal.is_finite():
        raise ValueError(f"{label} must be finite")
    return decimal


def normalize_resource_vector(value: Any) -> dict[str, Decimal]:
    if not isinstance(value, Mapping):
        raise ValueError("resource vector must be a mapping")
    result: dict[str, Decimal] = {}
    for resource, quantity in value.items():
        if not isinstance(resource, str) or not resource:
            raise ValueError("resource names must be non-empty strings")
        decimal = exact_decimal(quantity, label="resource quantities")
        if decimal < 0:
            raise ValueError("resource quantities must be non-negative")
        result[resource] = decimal
    return dict(sorted(result.items()))


def _freeze_resource_vector(
    value: Mapping[str, Decimal],
) -> Mapping[str, Decimal]:
    return FrozenResourceVector(value)


type ResourceVector = Annotated[
    Mapping[str, Decimal],
    AfterValidator(_freeze_resource_vector),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]


class ResourceModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class ResourceScale(ResourceModel):
    step: ResourceVector
    max_steps: int = Field(ge=0)
    weight: Decimal = Field(default=Decimal(1), gt=Decimal(0))

    @field_validator("step", mode="before")
    @classmethod
    def validate_step(cls, value: Any) -> dict[str, Decimal]:
        result = normalize_resource_vector(value)
        if not result or all(quantity == 0 for quantity in result.values()):
            raise ValueError("scale step must contain a positive resource quantity")
        return result

    @field_validator("weight", mode="before")
    @classmethod
    def validate_weight(cls, value: Any) -> Decimal:
        return exact_decimal(value, label="weight")


class ResourceProfile(ResourceModel):
    base: ResourceVector
    scale: ResourceScale | None = None

    @field_validator("base", mode="before")
    @classmethod
    def validate_base(cls, value: Any) -> dict[str, Decimal]:
        return normalize_resource_vector(value)

    @property
    def minimum(self) -> Mapping[str, Decimal]:
        return self.base

    @property
    def maximum(self) -> Mapping[str, Decimal]:
        steps = self.scale.max_steps if self.scale is not None else 0
        return self.resources_at(steps)

    def resources_at(self, steps: int) -> Mapping[str, Decimal]:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("resource scale steps must be an integer")
        maximum_steps = self.scale.max_steps if self.scale is not None else 0
        if not 0 <= steps <= maximum_steps:
            raise ValueError(
                f"resource scale steps must be between 0 and {maximum_steps}"
            )
        resources = dict(self.base)
        if self.scale is not None:
            for resource, quantity in self.scale.step.items():
                resources[resource] = resources.get(resource, Decimal(0)) + (
                    quantity * steps
                )
        return FrozenResourceVector(resources)
