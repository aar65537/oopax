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
from oopax.types import Array, Update


class Updater(oopax.Module):
    value: Array

    @oopax.update
    def __call__(self, new_value: Array) -> tuple[Update, Array]:
        return ({"value": new_value}, self.value)


@pytest.fixture()
def updater() -> Updater:
    return Updater(jnp.zeros(10))


def test_update(jit: bool, updater: Updater) -> None:
    clear_caches()

    call = updater.__class__.__call__
    call = eqx.filter_jit(chex.assert_max_traces(call, 1)) if jit else call

    for index in range(10):
        last_value = updater.value
        updater, result = call(updater, jnp.ones(10) * (index + 1))
        next_value = updater.value

        assert jnp.equal(last_value, result).all()
        assert jnp.equal(next_value - last_value, 1).all()

    if not jit:
        return

    with pytest.raises(AssertionError):
        call(updater, jnp.zeros(20))
