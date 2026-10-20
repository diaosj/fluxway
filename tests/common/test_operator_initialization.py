import pytest

from fluxway.common.operator_initialization import (
    OperatorInitializationError,
    operator_initialization_arguments,
)


class ExplicitInitializer:
    def __init__(self, required, optional=1):
        self.required = required
        self.optional = optional


class VariadicInitializer:
    def __init__(self, **values):
        self.values = values


class UninspectableInitializer:
    def __init__(self):
        pass


UninspectableInitializer.__init__.__signature__ = "invalid"  # type: ignore[attr-defined]


def test_explicit_initializer_receives_only_declared_static_arguments() -> None:
    assert operator_initialization_arguments(
        ExplicitInitializer,
        {"required": 2, "optional": 3, "unrelated": 4},
    ) == {"required": 2, "optional": 3}


def test_variadic_initializer_receives_all_static_arguments() -> None:
    values = {"first": 1, "second": 2}

    assert operator_initialization_arguments(VariadicInitializer, values) == values


def test_uninspectable_initializer_fails_instead_of_dropping_arguments() -> None:
    with pytest.raises(OperatorInitializationError, match="UninspectableInitializer"):
        operator_initialization_arguments(UninspectableInitializer, {"required": 1})
