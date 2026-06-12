"""Transformer Engine lazy import helpers."""


def is_te_available() -> bool:
    try:
        import transformer_engine.pytorch  # noqa: F401

        return True
    except Exception:
        return False


def require_te():
    try:
        import transformer_engine.pytorch as te

        return te
    except Exception as exc:
        raise RuntimeError(
            "Transformer Engine is required for TE minimal example. "
            "Please install transformer-engine with PyTorch support."
        ) from exc

