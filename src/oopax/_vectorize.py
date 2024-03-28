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

import dataclasses
from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate

import jax.numpy as jnp

from oopax.types import P, T, U


def vectorize(
    signature: str = "->()",
) -> Callable[[Callable[Concatenate[T, P], U]], Callable[Concatenate[T, P], U]]:
    def decorator(fn: Callable[Concatenate[T, P], U]) -> Callable[Concatenate[T, P], U]:
        @wraps(fn)
        def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> U:
            _signature = signature

            fields = dataclasses.fields(module)  # type: ignore
            module_args = []
            module_exclude = []

            for field in fields:
                if "signature" in field.metadata:
                    module_args.append(getattr(module, field.name))
                    _signature = field.metadata["signature"] + "," + _signature
                else:
                    module_exclude.append(getattr(module, field.name))

            _excluded = {len(module_args)} | kwargs.keys()

            def elementary_fn(*args: Any, **kwargs: P.kwargs) -> U:  # type: ignore
                _module = object.__new__(module.__class__)

                n_args = 0
                n_excludes = 0
                excludes = args[len(module_args)]

                for field in fields:
                    if "signature" in field.metadata:
                        object.__setattr__(_module, field.name, args[n_args])  # noqa: PLC2801
                        n_args += 1
                    else:
                        object.__setattr__(_module, field.name, excludes[n_excludes])  # noqa: PLC2801
                        n_excludes += 1

                return fn(_module, *args[len(module_args) + 1 :], **kwargs)

            vectorized_fn = jnp.vectorize(
                elementary_fn, excluded=_excluded, signature=_signature
            )

            return vectorized_fn(*module_args, module_exclude, *args, **kwargs)

        return inner

    return decorator
