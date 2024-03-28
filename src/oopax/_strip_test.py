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

import oopax
from oopax.testing.cache import clear_caches


def fn() -> tuple[None]:
    return (None,)


def test_strip(jit: bool) -> None:
    clear_caches()

    call = oopax.strip(fn)
    call = eqx.filter_jit(chex.assert_max_traces(call, 1)) if jit else call

    for _ in range(10):
        assert fn() == (None,)
        assert call() is None
