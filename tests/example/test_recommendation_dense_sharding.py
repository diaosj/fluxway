import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def test_dense_gradient_matches_parameter_sharding_before_accumulation():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("multiple JAX devices are required")

    from fluxway.example.recommendation.dense_spmd import _place_gradient_like

    mesh = Mesh(np.asarray(devices), ("dp",))
    reference = jax.device_put(
        jnp.zeros((1,), dtype=jnp.float32), NamedSharding(mesh, P())
    )
    local_gradient = jax.device_put(jnp.ones((1,), dtype=jnp.float32), devices[0])

    placed = _place_gradient_like(local_gradient, reference)
    updated = reference + placed

    assert placed.sharding == reference.sharding
    assert placed.devices() == reference.devices()
    np.testing.assert_array_equal(np.asarray(updated), np.ones((1,), dtype=np.float32))
