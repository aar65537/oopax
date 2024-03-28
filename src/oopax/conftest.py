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
import pytest

from oopax.types import PRNGKeyArray

SEED = 0


@pytest.fixture()
def key() -> PRNGKeyArray:
    return jax.random.PRNGKey(SEED)


@pytest.fixture(params=[True, False], ids=["jit", "no jit"])
def jit(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(
    params=[(), (10,), (5, 3)], ids=["shape=()", "shape=(10,)", "shape=(5,3)"]
)
def shape(request: pytest.FixtureRequest) -> tuple[int, ...]:
    return request.param
