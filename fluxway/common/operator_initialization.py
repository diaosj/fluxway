import inspect
from collections.abc import Mapping
from typing import Any


class OperatorInitializationError(TypeError):
    pass


def operator_initialization_arguments(
    owner: type,
    static_args: Mapping[str, Any],
) -> dict[str, Any]:
    initializer = inspect.getattr_static(owner, "__init__")
    try:
        signature = inspect.signature(initializer)
    except (TypeError, ValueError) as error:
        raise OperatorInitializationError(
            f"cannot inspect initializer for {owner.__module__}.{owner.__qualname__}"
        ) from error
    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(static_args)
    accepted = set(parameters) - {"self", "cls"}
    return {name: value for name, value in static_args.items() if name in accepted}
