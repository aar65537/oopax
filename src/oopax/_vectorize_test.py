# Copyright 2024 the OOPax Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import chex
import equinox as eqx
import jax.numpy as jnp
import pytest

import oopax
from oopax.testing.cache import clear_caches
from oopax.types import Array


class Selecter(oopax.Module):
    index: Array = oopax.field(signature="()")

    @oopax.vectorize("(n)->()")
    def __call__(self, arr: Array, *, extra: str) -> Array:
        del extra
        return arr[self.index]


@pytest.fixture()
def selecter(shape: tuple[int, ...]) -> Selecter:
    return Selecter(jnp.zeros(shape, int))


@pytest.fixture(
    params=[(), (10,), (5, 3)],
    ids=["batch_shape=()", "batch_shape=(10,)", "batch_shape=(5,3)"],
)
def batch_shape(request: pytest.FixtureRequest) -> tuple[int, ...]:
    return request.param


@pytest.fixture()
def arr(shape: tuple[int, ...], batch_shape: tuple[int, ...]) -> Array:
    return jnp.zeros((*batch_shape, *shape, 10))


def test_vectorize(jit: bool, selecter: Selecter, arr: Array) -> None:
    clear_caches()

    call = selecter.__class__.__call__
    call = eqx.filter_jit(chex.assert_max_traces(call, 1)) if jit else call

    for _ in range(10):
        value = call(selecter, arr, extra="hi")
        assert jnp.equal(value, 0).all()

    if not jit:
        return

    with pytest.raises(AssertionError):
        call(selecter, jnp.zeros((*selecter.index.shape, 2)), extra="hi")
