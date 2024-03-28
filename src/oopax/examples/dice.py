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


from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

import oopax
from oopax.types import Array, MapTree, PRNGKeyArray


class Dice(eqx.Module):
    key: PRNGKeyArray
    hist: Array
    weights: Array = oopax.field(signature="(n)")

    def __init__(
        self, key: PRNGKeyArray, *, n_sides: int = 6, weights: Array | None = None
    ) -> None:
        weights = jnp.ones((n_sides,), float) if weights is None else weights

        self.key = key
        self.weights = jax.nn.softmax(jnp.log(weights))
        self.hist = jnp.zeros((self.n_sides), int)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weights.shape[:-1]

    @property
    def n_sides(self) -> int:
        return self.weights.shape[-1]

    @eqx.filter_jit
    @oopax.strip_output
    @oopax.capture_update
    def reset(self) -> tuple[MapTree]:
        return ({"hist": jnp.zeros((self.n_sides,), int)},)

    @eqx.filter_jit
    @oopax.capture_update
    @oopax.consume_key
    def __call__(self, key: PRNGKeyArray, *args: int) -> tuple[MapTree, Array]:
        shape = (*args, *self.shape)
        key = jax.random.split(key, shape)
        result = self._call(key)
        hist = self.hist

        for index in range(self.n_sides):
            hist = hist.at[index].add((result == index).sum())

        return {"hist": hist}, result

    @partial(oopax.vectorize, signature="(2)->()")
    def _call(self, key: PRNGKeyArray) -> Array:
        return jax.random.choice(
            key, jnp.arange(self.n_sides, dtype=int), (), p=self.weights
        )
