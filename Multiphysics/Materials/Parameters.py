# Parameter.py
from __future__ import annotations

import json
from dataclasses import asdict, fields, make_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Type, TypeVar, Union

T = TypeVar("T")
JsonSrc = Union[str, Path, Mapping[str, Any]]


def make_param_type(
    name: str,
    field_names: Sequence[str],
    *,
    frozen: bool = True,
    slots: bool = True,
) -> Type[Any]:
    """
     dataclass :
      - : obj.E
      - : frozen=True
      - : slots=True

    field_names: 
    """
    if not field_names:
        raise ValueError("field_names cannot be empty")


    seen = set()
    ordered = []
    for k in field_names:
        if not isinstance(k, str) or not k:
            raise TypeError(f"Invalid field name: {k!r}")
        if k in seen:
            continue
        seen.add(k)
        ordered.append(k)


    cls = make_dataclass(
        cls_name=name,
        fields=[(k, Any) for k in ordered],
        frozen=frozen,
        slots=slots,
    )


    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_str(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_json_file(self, path: Union[str, Path], *, indent: int = 2) -> None:
        Path(path).write_text(self.to_json_str(indent=indent), encoding="utf-8")

    setattr(cls, "to_dict", to_dict)
    setattr(cls, "to_json_str", to_json_str)
    setattr(cls, "to_json_file", to_json_file)
    return cls


def _load_json_src(src: JsonSrc) -> Mapping[str, Any]:
    """
    src :
      - dict-like
      - JSON  Path / str
      - JSON 
    """
    if isinstance(src, Mapping):
        return src

    if isinstance(src, (str, Path)):
        s = str(src)
        p = Path(s)
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))


        return json.loads(s)

    raise TypeError(f"Unsupported JSON source type: {type(src)}")


def build_params(
    ParamType: Type[T],
    data: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> T:
    """
     dict 
    strict=True: data  defaults  data 
    strict=False: 
    """
    field_set = {f.name for f in fields(ParamType)}
    data_keys = set(data.keys())

    extra = data_keys - field_set
    if extra and strict:
        raise KeyError(f"Unknown keys for {ParamType.__name__}: {sorted(extra)}")

    merged: dict[str, Any] = {}
    if defaults:
        merged.update(defaults)
    merged.update({k: data[k] for k in data.keys() if (k in field_set) or (not strict)})

    missing = [k for k in field_set if k not in merged]
    if missing:
        raise KeyError(f"Missing keys for {ParamType.__name__}: {sorted(missing)}")


    kwargs = {k: merged[k] for k in field_set}
    return ParamType(**kwargs)


def from_json(
    name: str,
    src: JsonSrc,
    *,
    field_names: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
    infer_order: str = "as_is",
    strict: bool = True,
) -> Any:
    """
    :  JSON 

    field_names:
      - None:  JSON  key 
      -  list: 

    infer_order:
      - "as_is":  JSON dict Python 3.7+ 
      - "sorted": 
    """
    data = dict(_load_json_src(src))

    if field_names is None:
        keys = list(data.keys())
        if infer_order == "sorted":
            keys = sorted(keys)
        elif infer_order != "as_is":
            raise ValueError("infer_order must be 'as_is' or 'sorted'")
        field_names = keys

    ParamType = make_param_type(name, field_names)
    return build_params(ParamType, data, defaults=defaults, strict=strict)


def make_params(
    name: str,
    field_names: Sequence[str],
    /,
    **kwargs: Any,
) -> Any:
    """
    :  kwargs
    """
    ParamType = make_param_type(name, field_names)
    return build_params(ParamType, kwargs, strict=True)

