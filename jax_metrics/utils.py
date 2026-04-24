import re
import typing as tp

import jax
import jax.numpy as jnp

from jax_metrics import types


def _flatten_names(inputs: tp.Any) -> tp.List[tp.Tuple[str, tp.Any, bool]]:
    return [
        ("/".join(map(str, path)), value, parent_iterable)
        for path, value, parent_iterable in _flatten_names_helper((), inputs, True)
    ]


def _flatten_names_helper(
    path: types.PathLike, inputs: tp.Any, parent_iterable: bool
) -> tp.Iterable[tp.Tuple[types.PathLike, tp.Any, bool]]:
    if isinstance(inputs, (tp.Tuple, tp.List)):
        for i, value in enumerate(inputs):
            yield from _flatten_names_helper(path, value, True)
    elif isinstance(inputs, tp.Dict):
        for name, value in inputs.items():
            yield from _flatten_names_helper(path + (name,), value, False)
    else:
        yield (path, inputs, parent_iterable)


def _get_name(obj) -> str:
    pass


def _lower_snake_case(s: str) -> str:
    pass


def _unique_name(
    names: tp.Set[str],
    name: str,
):
    if name in names:
        match = re.match(r"(.*?)(\d*)$", name)
        assert match is not None

        name = match[1]
        num_part = match[2]

        i = int(num_part) if num_part else 2
        str_template = f"{{name}}{{i:0{len(num_part)}}}"

        while str_template.format(name=name, i=i) in names:
            i += 1

        name = str_template.format(name=name, i=i)

    names.add(name)
    return name


def _unique_names(
    names: tp.Iterable[str],
    *,
    existing_names: tp.Optional[tp.Set[str]] = None,
) -> tp.Iterable[str]:
    pass
