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


import equinox as eqx
import jax
import jax.numpy as jnp

import oopax
from oopax.types import Array, MapTree, PRNGKeyArray


class DiceRoller(eqx.Module):
    key: PRNGKeyArray
    weights: Array
    hist: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        n_sides: int = 6,
        weights: Array | None = None,
    ) -> None:
        if weights is None:
            weights = jnp.ones((n_sides,), float)
        else:
            n_sides = weights.shape[-1]

        self.weights = jax.nn.softmax(jnp.log(weights))
        self.key = jax.random.split(key, self.batch_shape)
        self.hist = jnp.zeros((*self.batch_shape, n_sides), int)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.weights.shape[:-1]

    @property
    def n_sides(self) -> int:
        return self.weights.shape[-1]

    @eqx.filter_jit
    @oopax.strip_output
    @oopax.capture_update
    @oopax.auto_vmap
    def reset(self) -> tuple[MapTree, None]:
        return {"hist": jnp.zeros((self.n_sides,), int)}, None

    @eqx.filter_jit
    @oopax.capture_update
    @oopax.auto_vmap
    @oopax.consume_key
    def roll(self, key: PRNGKeyArray, n_rolls: int = 1) -> tuple[MapTree, Array]:
        if n_rolls < 1:
            msg = "n_rolls must be greater than or equal to one."
            raise ValueError(msg)

        if n_rolls == 1:
            hist, result = self._roll(self.hist, key)
        else:
            keys = jax.random.split(key, n_rolls)
            hist, result = jax.lax.scan(self._roll, self.hist, keys)

        return {"hist": hist}, result

    def _roll(self, hist: Array, key: PRNGKeyArray) -> tuple[Array, Array]:
        result = jax.random.choice(
            key, jnp.arange(self.n_sides, dtype=int), (), p=self.weights
        )
        return hist.at[result].add(1), result
