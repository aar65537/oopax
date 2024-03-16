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


import jax
import jax.numpy as jnp
import pytest
from oopax.examples import DiceRoller

pytestmark = [
    pytest.mark.parametrize("jit", [True, False]),
]


def main() -> None:
    key = jax.random.PRNGKey(0)
    weights = jnp.arange(600).reshape(10, 10, 6) % 7
    roller = DiceRoller(key, weights=weights)
    # roller = DiceRoller(key)
    print(roller.weights.sum((0, 1)) / roller.weights.sum())
    for _ in range(10):
        roller, roll = roller.roll(10**3)
        print(roll.shape)
        print(roller.hist.sum((0, 1)) / roller.hist.sum())

    # print(jnp.histogram(roller.hist))
    roller = roller.reset()
    print(roller.hist.sum())


if __name__ == "__main__":
    main()
