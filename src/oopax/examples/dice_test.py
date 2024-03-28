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

from oopax.examples.dice import Dice
from oopax.testing.cache import clear_caches
from oopax.types import Array, PRNGKeyArray


@pytest.fixture(params=[6, 12], ids=["n_sides=6", "n_sides=12"])
def n_sides(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(
    params=[(), (10,), (5, 3)], ids=["shape=()", "shape=(10,)", "shape=(5,3)"]
)
def shape(request: pytest.FixtureRequest) -> tuple[int, ...]:
    return request.param


@pytest.fixture(
    params=[None, jnp.arange(6), jnp.arange(35).reshape(7, 5) % 4],
    ids=["weights=None", "weights=(6,)", "weights=(7,5)"],
)
def weights(request: pytest.FixtureRequest) -> Array | None:
    return request.param


@pytest.fixture()
def dice(key: PRNGKeyArray, n_sides: int, weights: Array | None) -> Dice:
    return Dice(key, n_sides=n_sides, weights=weights)


def test_dice_roll(jit: bool, dice: Dice, shape: tuple[int, ...]) -> None:
    clear_caches()

    roll = dice.__class__.__call__
    roll = eqx.filter_jit(chex.assert_max_traces(roll, 1)) if jit else roll

    for _ in range(10):
        old_hist = dice.hist
        dice, result = roll(dice, *shape)

        hist_diff = dice.hist - old_hist
        if result.shape == ():
            assert hist_diff[result] == 1
            assert hist_diff.sum() == 1
        else:
            prod = 1
            for axis in result.shape:
                prod *= axis
            assert hist_diff.sum() == prod

    if not jit:
        return

    with pytest.raises(AssertionError):
        roll(dice, 120)


def test_dice_reset(key: PRNGKeyArray, jit: bool, dice: Dice) -> None:
    clear_caches()

    reset = dice.__class__.reset
    reset = eqx.filter_jit(chex.assert_max_traces(reset, 1)) if jit else reset

    dice, _ = dice()
    assert dice.hist.sum() > 0

    dice = reset(dice)
    assert dice.hist.sum() == 0

    if not jit:
        return

    with pytest.raises(AssertionError):
        dice = reset(Dice(key, n_sides=100))
