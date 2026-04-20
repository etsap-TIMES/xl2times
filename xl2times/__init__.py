# TODO is it better to have version here or in pyproject.toml (or both)?
__version__ = "0.3.0"

from .times_data_adapter import to_times_data_model, to_times_data_payload

__all__ = ["to_times_data_model", "to_times_data_payload", "__version__"]
