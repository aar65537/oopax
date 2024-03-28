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
import jax
import jax.numpy as jnp
import pytest

import oopax
from oopax.testing.cache import clear_caches
from oopax.types import PRNGKeyArray, Update


class Injecter(oopax.Module):
    key: PRNGKeyArray

    @oopax.update
    @oopax.inject_key
    def __call__(self, key: PRNGKeyArray) -> tuple[Update, PRNGKeyArray]:
        return ({}, key)


@pytest.fixture()
def injecter(key: PRNGKeyArray, shape: tuple[int, ...]) -> Injecter:
    return Injecter(key=jax.random.split(key, shape))


def test_inject_key(key: PRNGKeyArray, jit: bool, injecter: Injecter) -> None:
    clear_caches()

    call = injecter.__class__.__call__
    call = eqx.filter_jit(chex.assert_max_traces(call, 1)) if jit else call

    for _ in range(10):
        last_key = injecter.key
        injecter, result = call(injecter)
        next_key = injecter.key

        assert not jnp.equal(last_key, result).all()
        assert not jnp.equal(result, next_key).all()
        assert not jnp.equal(last_key, next_key).all()

    if not jit:
        return

    with pytest.raises(AssertionError):
        call(Injecter(jax.random.split(key, (120,))))
