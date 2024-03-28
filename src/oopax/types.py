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

from collections.abc import Iterable, Mapping
from typing import Any, ParamSpec, TypeAlias, TypeVar, TypeVarTuple

from jaxtyping import Array, ArrayLike, PRNGKeyArray

ArrayTree = Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
Update: TypeAlias = Mapping[str, ArrayTree]
FlatUpdate: TypeAlias = Iterable[ArrayTree]

T = TypeVar("T")
U = TypeVar("U")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")

__all__ = [
    "Array",
    "ArrayLike",
    "ArrayTree",
    "FlatUpdate",
    "P",
    "PRNGKeyArray",
    "T",
    "Ts",
    "U",
    "Update",
]
