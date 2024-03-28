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

from collections.abc import Callable
from functools import wraps
from typing import Concatenate

import equinox as eqx

from oopax.types import FlatUpdate, P, T, Ts, Update


def update(
    fn: Callable[Concatenate[T, P], tuple[Update, *Ts]],
) -> Callable[Concatenate[T, P], tuple[T, *Ts]]:
    @wraps(fn)
    def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> tuple[T, *Ts]:
        update, *output = fn(module, *args, **kwargs)

        def where_fn(_module: T) -> FlatUpdate:
            return [getattr(_module, attr) for attr in update]

        dynamic_module, static_module = eqx.partition(module, eqx.is_array)
        dynamic_update = eqx.filter(list(update.values()), eqx.is_array)
        dynamic_new_module = eqx.tree_at(where_fn, dynamic_module, dynamic_update)
        new_module: T = eqx.combine(dynamic_new_module, static_module)

        return (new_module, *output)  # type: ignore

    return inner
