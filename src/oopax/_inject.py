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
from functools import partial, wraps
from typing import Any, Concatenate

import jax
import jax.numpy as jnp

from oopax.types import ArrayTree, P, PRNGKeyArray, T, Ts, U, Update


def inject(
    fn: Callable[Concatenate[T, U, P], tuple[Update, *Ts]],
    *,
    attr: str,
    split: Callable[Concatenate[T, P], tuple[U, U]],
) -> Callable[Concatenate[T, P], tuple[Update, *Ts]]:
    @wraps(fn)
    def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> tuple[Update, *Ts]:
        next_attr, sub_attr = split(module, *args, **kwargs)
        update, *output = fn(module, sub_attr, *args, **kwargs)

        if attr in update:
            msg = (
                f"Update from '{fn.__qualname__}' already contains "
                f"attribute '{attr}'."
            )
            raise ValueError(msg)

        update = {attr: next_attr, **update}
        return (update, *output)  # type: ignore

    del inner.__wrapped__
    return inner


def inject_key(
    fn: Callable[Concatenate[T, PRNGKeyArray, P], tuple[Update, *Ts]],
    *,
    key: str = "key",
) -> Callable[Concatenate[T, P], tuple[Update, *Ts]]:
    split_key = partial(_split_key, key=key)
    return inject(fn, attr=key, split=split_key)  # type: ignore


def _split_key(
    module: ArrayTree, *args: Any, key: str, **kwargs: Any
) -> tuple[PRNGKeyArray, PRNGKeyArray]:
    del args, kwargs

    old_key: PRNGKeyArray = getattr(module, key)
    next_keys = _split(old_key)
    next_key, sub_key = next_keys.take(0, -2), next_keys.take(1, -2)

    return (next_key, sub_key)


_split: Callable[[PRNGKeyArray], PRNGKeyArray]
_split = jnp.vectorize(jax.random.split, signature="(2)->(2,2)")
