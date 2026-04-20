import pandas as pd
import pytest

import xl2times.times_data_adapter as adapter
from xl2times.datatypes import TimesModel


def _sample_model() -> TimesModel:
    model = TimesModel()
    model.internal_regions = {"REG1"}
    model.all_regions = {"REG1", "REG2"}
    model.processes = pd.DataFrame([{"process": "P1", "region": "REG1"}])
    model.commodities = pd.DataFrame([{"commodity": "C1", "region": "REG1"}])
    model.topology = pd.DataFrame(
        [{"region": "REG1", "process": "P1", "commodity": "C1", "io": "IN"}]
    )
    model.time_periods = pd.DataFrame([{"m": 2020}])
    model.start_year = 2020
    model.files = ["demo_model"]
    model.data_modules = ["base"]
    model.cases = {"BASE": "1"}
    return model


def test_to_times_data_payload_contains_model_config_and_copies():
    model = _sample_model()

    payload = adapter.to_times_data_payload(model)

    assert payload["model_config"]["regions"] == ["REG1", "REG2"]
    assert payload["model_config"]["external_regions"] == ["REG2"]
    assert payload["model_config"]["periods"] == [2020]
    assert payload["processes"].equals(model.processes)
    assert payload["processes"] is not model.processes


def test_to_times_data_model_prefers_model_validate():
    model = _sample_model()

    class ValidatingModel:
        called_with = None

        @classmethod
        def model_validate(cls, data):
            cls.called_with = data
            return cls()

    converted = adapter.to_times_data_model(model, model_cls=ValidatingModel)

    assert isinstance(converted, ValidatingModel)
    assert ValidatingModel.called_with["start_year"] == 2020


def test_to_times_data_model_filters_constructor_kwargs():
    model = _sample_model()

    class MinimalModel:
        def __init__(self, processes, model_config):
            self.processes = processes
            self.model_config = model_config

    converted = adapter.to_times_data_model(model, model_cls=MinimalModel)

    assert converted.processes.equals(model.processes)
    assert converted.model_config["start_year"] == 2020


def test_to_times_data_model_raises_helpful_import_error(monkeypatch):
    model = _sample_model()

    def _raise_import_error():
        raise ImportError("missing times-data")

    monkeypatch.setattr(adapter, "_resolve_times_data_model_class", _raise_import_error)

    with pytest.raises(ImportError, match="missing times-data"):
        adapter.to_times_data_model(model)
