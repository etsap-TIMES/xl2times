"""Adapter utilities for converting :class:`TimesModel` to `times-data` objects."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from .datatypes import TimesModel


def to_times_data_payload(model: TimesModel) -> dict[str, Any]:
    """Convert a :class:`TimesModel` to a plain mapping for `times-data` consumption.

    Returns a mapping containing all ``TimesModel`` table fields and metadata,
    plus a nested ``model_config`` dictionary with ``regions``,
    ``internal_regions``, ``external_regions``, ``periods``, and ``start_year``.

    DataFrames are copied so downstream conversion logic can mutate payload
    tables without mutating the original ``TimesModel`` object.
    """
    periods: list[int] = []
    if not model.time_periods.empty and "m" in model.time_periods.columns:
        periods = [int(year) for year in model.time_periods["m"].dropna().tolist()]
    internal_regions = sorted(model.internal_regions)
    all_regions = sorted(model.all_regions)

    return {
        "internal_regions": internal_regions,
        "all_regions": all_regions,
        "processes": model.processes.copy(),
        "commodities": model.commodities.copy(),
        "commodity_groups": model.commodity_groups.copy(),
        "topology": model.topology.copy(),
        "implied_topology": model.implied_topology.copy(),
        "trade": model.trade.copy(),
        "attributes": model.attributes.copy(),
        "user_constraints": model.user_constraints.copy(),
        "uc_attributes": model.uc_attributes.copy(),
        "ts_tslvl": model.ts_tslvl.copy(),
        "ts_map": model.ts_map.copy(),
        "time_periods": model.time_periods.copy(),
        "units": model.units.copy(),
        "start_year": model.start_year,
        "files": list(model.files),
        "data_modules": list(model.data_modules),
        "custom_psets": model.custom_psets.copy(),
        "user_psets": model.user_psets.copy(),
        "user_csets": model.user_csets.copy(),
        "cases": dict(model.cases),
        "model_config": {
            "regions": all_regions,
            "internal_regions": internal_regions,
            "external_regions": sorted(model.external_regions),
            "periods": periods,
            "start_year": model.start_year,
        },
    }


def to_times_data_model(model: TimesModel, model_cls: type[Any] | None = None) -> Any:
    """Convert a :class:`TimesModel` into a `times-data` ``Model`` object.

    If ``model_cls`` is ``None``, this function attempts to import the
    `times-data` ``Model`` class from common module paths. If no class is
    found, ``ImportError`` is raised.
    """
    resolved_model_cls = model_cls or _resolve_times_data_model_class()
    payload = to_times_data_payload(model)
    return _instantiate_model(resolved_model_cls, payload)


def _resolve_times_data_model_class() -> type[Any]:
    """Resolve `times-data` Model class from common import paths."""
    candidates = ("times_data", "times_data.model", "times_data.models")
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        model_cls = getattr(module, "Model", None)
        if isinstance(model_cls, type):
            return model_cls
    raise ImportError(
        "Could not import 'times-data' Model class. Install 'times-data' or provide "
        "'model_cls' explicitly to 'to_times_data_model'."
    )


def _instantiate_model(model_cls: type[Any], payload: dict[str, Any]) -> Any:
    """Instantiate model using supported constructor patterns."""
    model_validate = getattr(model_cls, "model_validate", None)
    if callable(model_validate):
        return model_validate(payload)

    for method_name in ("from_dict", "from_mapping", "from_tables"):
        method = getattr(model_cls, method_name, None)
        if callable(method):
            return method(payload)

    try:
        signature = inspect.signature(model_cls)
    except (TypeError, ValueError):
        return model_cls(payload)

    params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params
    )
    if accepts_kwargs:
        return model_cls(**payload)

    supported_keys = {
        parameter.name
        for parameter in params
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    kwargs = {key: value for key, value in payload.items() if key in supported_keys}
    if kwargs:
        return model_cls(**kwargs)
    if len(params) == 1:
        return model_cls(payload)
    raise TypeError(
        f"Could not instantiate {model_cls.__name__} with adapter payload keys. "
        "Pass a custom `model_cls` with a supported constructor."
    )
