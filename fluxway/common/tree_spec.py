from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class TreeSpec(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    type: Literal["leaf", "data_type", "dict", "list", "tuple"]
    data_type: type | None = None
    keys: tuple[str, ...] = ()
    children: tuple[TreeSpec, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> TreeSpec:
        if self.type == "leaf":
            if self.data_type is not None or self.keys or self.children:
                raise ValueError("leaf TreeSpec cannot have metadata or children")
            return self
        if self.type == "data_type":
            if self.data_type is None:
                raise ValueError("data_type TreeSpec requires a class")
            if len(self.keys) != len(self.children):
                raise ValueError("data_type TreeSpec keys and children must align")
            return self
        if self.data_type is not None:
            raise ValueError(f"{self.type} TreeSpec cannot have a data type")
        if self.type == "dict" and len(self.keys) != len(self.children):
            raise ValueError("dict TreeSpec keys and children must align")
        if self.type in ("list", "tuple") and self.keys:
            raise ValueError(f"{self.type} TreeSpec cannot have keys")
        return self

    @classmethod
    def from_runtime(cls, value: Any) -> TreeSpec:
        if value is None:
            return cls(type="leaf")
        if not isinstance(value, tuple) or not value:
            raise TypeError("runtime tree definitions must be tuples or None")
        tree_type = value[0]
        if tree_type == "data_type":
            data_type, keys, children = value[1], value[2], value[3]
            if not isinstance(data_type, type):
                raise TypeError("data_type tree definition must contain a class")
            return cls(
                type="data_type",
                data_type=data_type,
                keys=tuple(keys),
                children=tuple(cls.from_runtime(child) for child in children),
            )
        if tree_type == "dict":
            return cls(
                type="dict",
                keys=tuple(value[1]),
                children=tuple(cls.from_runtime(child) for child in value[2]),
            )
        if tree_type in ("list", "tuple"):
            return cls(
                type=tree_type,
                children=tuple(cls.from_runtime(child) for child in value[2]),
            )
        raise ValueError(f"unsupported runtime tree definition type {tree_type!r}")

    def materialize(self) -> Any:
        if self.type == "leaf":
            return None
        children = [child.materialize() for child in self.children]
        if self.type == "data_type":
            if self.data_type is None:
                raise RuntimeError("validated data_type TreeSpec lost its class")
            return ("data_type", self.data_type, list(self.keys), children)
        if self.type == "dict":
            return ("dict", list(self.keys), children)
        return (self.type, len(children), children)
