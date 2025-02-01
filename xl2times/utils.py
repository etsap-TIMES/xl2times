from __future__ import (
    annotations,
)

# see https://loguru.readthedocs.io/en/stable/api/type_hints.html#module-autodoc_stub_file.loguru
import functools
import gzip
import os
import pickle
import sys
from dataclasses import replace
from math import floor, log10
from pathlib import Path, PurePath

import numpy
import pandas as pd
from loguru import logger
from more_itertools import one
from pandas.core.frame import DataFrame

from . import datatypes

# prevent excessive number of processes in Windows and high cpu-count machines
# TODO make this a cli param or global setting?
max_workers: int = 4 if os.name == "nt" else min(16, os.cpu_count() or 16)


def apply_composite_tag(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
    """Handles table level declarations.

    Declarations can be included in the table tag and will apply to all data that doesn't
    have a different value for that index specified. For example, ~FI_T: DEMAND would
    assign DEMAND as the attribute for all values in the table that don't have an
    attribute specification at the column or row level. After applying the declaration
    this function will return the modified table with the simplified table tag (e.g.
    ~FI_T).

    See page 15 of https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
    for more context.

    Parameters
    ----------
    table
        Table in EmbeddedXlTable format.

    Returns
    -------
    datatypes.EmbeddedXlTable
        Table in EmbeddedXlTable format with declarations applied
        and table tag simplified.
    """
    defaults = table.defaults
    if not defaults:
        return table

    df = table.dataframe
    # Check for ANSWER-style defaults
    if "=" in defaults:
        # Split multiple comma-separated defaults / make defaults a list
        defaults = defaults.split(",")
        # Check whether there are invalid values on the list
        invalid_defaults = [default for default in defaults if "=" not in default]
        if invalid_defaults:
            logger.warning(f"Expected ANSWER-style defaults, got {invalid_defaults}")
        defaults = [default.split("=") for default in defaults if "=" in default]
        # TODO: check whether a column is allowed in a particular table type
        for col, val in defaults:
            colname = col.lower()
            if colname in df.columns:
                df[colname] = df[colname].fillna(val.upper())
            else:
                df[colname] = val.upper()
    else:
        # TODO: Resolve the default value (it doesn't have to be an attribute)
        if "attribute" not in df.columns:
            df["attribute"] = pd.NA
        df["attribute"] = df["attribute"].fillna(defaults.upper())
    return replace(table, dataframe=df)


def explode(df: DataFrame, data_columns: list[str]) -> tuple[DataFrame, pd.Series]:
    """Transpose the 'data_columns' in each row into a column of values, replicating the
    other columns. The name for the new column is "VALUE".

    Parameters
    ----------
    df
        Dataframe to be exploded.
    data_columns
        Names of the columns to be exploded.

    Returns
    -------
    tuple[DataFrame,pd.Series]
        Tuple with the exploded dataframe and a Series of the original
        column name for each value in each new row.
    """
    # Handle duplicate columns (https://pandas.pydata.org/docs/user_guide/duplicates.html)
    if len(set(data_columns)) < len(data_columns):
        cols = df.columns.to_list()
        data_cols_idx = [idx for idx, val in enumerate(cols) if val in data_columns]
        data = df.iloc[:, data_cols_idx].values.tolist()
    else:
        data = df[data_columns].values.tolist()

    other_columns = [
        colname for colname in df.columns.values if colname not in data_columns
    ]
    df = df[other_columns]
    value_column = "value"
    df = df.assign(value=data)
    nrows = df.shape[0]
    df = df.explode(value_column, ignore_index=True)
    names = pd.Series(data_columns * nrows, index=df.index, dtype=str)

    return df, names


def single_table(
    tables: list[datatypes.EmbeddedXlTable], tag: str
) -> datatypes.EmbeddedXlTable:
    """Make sure exactly one table in 'tables' has the given table tag, and return it.
    If there are none or more than one raise an error.

    Parameters
    ----------
    tables
        List of tables in EmbeddedXlTable format.
    tag
        Tag name.

    Returns
    -------
    datatypes.EmbeddedXlTable
        Table with the given tag in EmbeddedXlTable format.
    """
    return one(table for table in tables if table.tag == tag)


def single_column(
    tables: list[datatypes.EmbeddedXlTable], tag: str, colname: str
) -> numpy.ndarray:
    """Make sure exactly one table in 'tables' has the given table tag, and return the
    values for the given column name. If there are none or more than one raise an error.

    Parameters
    ----------
    tables
        List of tables in EmbeddedXlTable format.
    tag
        Tag name.
    colname
        Column name to return the values of.

    Returns
    -------
    numpy.ndarray
        Values for the column in the given table.
    """
    return single_table(tables, tag).dataframe[colname].values


def merge_columns(
    tables: list[datatypes.EmbeddedXlTable], tag: str, colname: str
) -> numpy.ndarray:
    """Return a list with all the values belonging to a column 'colname' from a table
    with the given tag.

    Parameters
    ----------
    tables
        List of tables in EmbeddedXlTable format.
    tag
        Tag name to select tables
    colname
        Column name to select values.

    Returns
    -------
    numpy.ndarray
        List of values for the given column name and tag.
    """
    columns = [table.dataframe[colname].values for table in tables if table.tag == tag]
    return numpy.concatenate(columns)


def missing_value_inherit(df: DataFrame, colname: str) -> None:
    """For each None value in the specifed column of the dataframe, replace it with the
    last non-None value. If no previous non-None value is found leave it as it is. This
    function modifies the supplied dataframe and returns None.

    Parameters
    ----------
    df
        Dataframe to be filled in.
    colname
        Name of the column to be filled in.

    Returns
    -------
    None
        None. The dataframe is filled in in place.
    """
    # TODO: should we use pandas.DataFrame.fillna(method="ffill") instead?
    last = None
    for index, value in df[colname].items():
        if value is None:
            df.loc[index, colname] = last
        else:
            last = value


def get_scalar(table_tag: str, tables: list[datatypes.EmbeddedXlTable]):
    table = one(filter(lambda t: t.tag == table_tag, tables))
    if table.dataframe.shape[0] != 1 or table.dataframe.shape[1] != 1:
        raise ValueError("Not scalar table")
    return table.dataframe["value"].values[0]


def has_negative_patterns(pattern: str) -> bool:
    if len(pattern) == 0:
        return False
    return pattern[0] == "-" or ",-" in pattern


def remove_negative_patterns(pattern: str) -> str:
    # Remove trailing commas
    pattern = pattern.rstrip(",")
    if len(pattern) == 0:
        return pattern
    return ",".join([word for word in pattern.split(",") if word[0] != "-"])


def remove_positive_patterns(pattern: str) -> str:
    # Remove trailing commas
    pattern = pattern.rstrip(",")
    if len(pattern) == 0:
        return pattern
    return ",".join([word[1:] for word in pattern.split(",") if word[0] == "-"])


def remove_whitespace(pattern: str) -> str:
    return ",".join([word.strip() for word in pattern.split(",")])


@functools.lru_cache(maxsize=int(1e6))
def create_regexp(pattern: str, combined: bool = True) -> str:
    pattern = remove_whitespace(pattern)
    # Exclude negative patterns
    if has_negative_patterns(pattern):
        pattern = remove_negative_patterns(pattern)
    # Handle comma-separated values
    pattern = pattern.replace(",", r"$|^")
    if len(pattern) == 0:
        return r".*"  # matches everything
    # Substite VEDA wildcards with regex patterns; escape metacharacters.
    # ("_", ".") and ("[.]", "_") are meant to apply one after another to handle
    # the usage of "_" equivalent to "?" and "[_]" as literal "_".
    substitute = [(".", "\\."), ("_", "."), ("[.]", "_"), ("*", ".*"), ("?", ".")]
    for old, new in substitute:
        pattern = pattern.replace(old, new)
    # Do not match substrings
    pattern = rf"^{pattern}$"
    return pattern


@functools.lru_cache(maxsize=int(1e6))
def create_negative_regexp(pattern: str) -> str:
    pattern = remove_whitespace(pattern)
    # Exclude positive patterns
    pattern = remove_positive_patterns(pattern)
    if len(pattern) == 0:
        pattern = r"^$"  # matches nothing
    return create_regexp(pattern)


def round_sig(x, sig_figs):
    if x == 0.0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + sig_figs - 1)


# Get entry point file name as default log name
default_log_name = Path(sys.argv[0]).stem
default_log_name = "log" if default_log_name == "" else default_log_name


def _case_insensitive_match(path: str, pattern: str) -> bool:
    """Do case-insensitive path match. Convert to lowercase first, because
    case_sensitive parameter in match is not available before Python 3.12.
    """
    return PurePath(path.lower()).match(pattern.lower())


def is_veda_based(files: list[str]) -> bool:
    """Determine whether the model follows Veda file structure.
    This function does not verify file extensions.
    """
    marker = "SysSettings.*"

    matches = [file for file in files if _case_insensitive_match(file, marker)]

    if len(matches) == 1:
        return True
    elif len(matches) > 1:
        raise ValueError(f"Only one {marker} expected. Multiple detected: {matches}")
    else:
        return False


def filter_veda_filename_patterns(files: list[str]) -> list[str]:
    """Filter files by patterns recognised by Veda.
    This function does not verify file extensions.
    """
    legal_paths = (
        "BY_Trans.*",
        "LMA*.*",
        "Set*.*",
        "SysSettings.*",
        "VT_*.*",
        "SubRES_TMPL/SubRES_*.*",
        "SuppXLS/Demands/Dem_Alloc+Series.*",
        "SuppXLS/Demands/ScenDem_*.*",
        "SuppXLS/ParScenFiles/Scen_Par-*.*",
        "SuppXLS/Scen_*.*",
        "SuppXLS/Trades/ScenTrade_*.*",
    )
    # Generate a set of fiels that match the patterns
    filtered = {
        file
        for file in files
        for legal_path in legal_paths
        if _case_insensitive_match(file, legal_path)
    }
    # Return as a list
    return list(filtered)


def set_log_level(level: int | None) -> str:
    """Sets the log level, in order of priority, to the provided int `level`, the
    `LOGURU_LEVEL` environment variable, or `WARNING` by default.

    E.g. `os.environ["LOGURU_LEVEL"] = "INFO"`
    Available levels are `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, and
    `CRITICAL`. Default is `SUCCESS` which is level `0`, and higher levels are more
    verbose.
    """
    level_map = {
        3: "TRACE",
        2: "DEBUG",
        1: "INFO",
        0: "SUCCESS",
        -1: "WARNING",
        -2: "ERROR",
        -3: "CRITICAL",
    }
    # First priority is argument `level`
    if level is not None:
        return level_map[level]
    # Second, if env var is set, let's roll with that
    env_level = os.getenv("LOGURU_LEVEL")
    if env_level is not None:
        return env_level
    # Default log level
    return level_map[0]


def setup_logger(
    level: int | None, log_name: str = default_log_name, log_dir: str = "."
):
    """Configure loguru.

    Call this once from entrypoints to set up a new logger.
    In non-entrypoint modules, just use `from loguru import logger` directly.

    Log file will be written to `f"{log_dir}/{log_name}.log"`

    Parameters
    ----------
    log_name
        Name of the log. Corresponding log file will be called {log_name}.log. (Default value = default_log_name)
    log_dir
        Directory to write the log file to. Default is the current working directory.
    """
    log_level = set_log_level(level)

    base_format = "<cyan>{time:YYYY-MM-DD HH:mm:ss.SSS}</cyan> | <level>{level: >8}</level> : <level>{message}</level>"
    filename_and_thread = '(<cyan>{name}:{thread.name}:pid-{process}</cyan> "<cyan>{file.path}</cyan>:<cyan>{line}</cyan>")'
    if level is not None and level > 1:
        stdout_format = base_format + filename_and_thread
    else:
        stdout_format = base_format

    logger.remove()
    logger.add(
        sink=sys.stdout,
        diagnose=True,
        level=log_level,
        format=stdout_format,
    )
    logger.add(
        sink=f"{log_dir}/{log_name}.log",
        enqueue=True,
        mode="a+",
        level="INFO",
        format=base_format + filename_and_thread,
        colorize=False,
        serialize=False,
        diagnose=False,
        rotation="20 MB",
        compression="zip",
    )


def save_state(
    config: datatypes.Config,
    tables: dict[str, DataFrame],
    model: datatypes.TimesModel,
    filename: str,
) -> None:
    """Save the state from a transform step to a single pickle file.

    Useful for troubleshooting regressions by diffing with state from another branch.
    """
    pickle.dump({"tables": tables, "model": model}, gzip.open(filename, "wb"))
    logger.debug(f"State saved to {filename}")


def compare_df_dict(
    df_before: dict[str, DataFrame],
    df_after: dict[str, DataFrame],
    sort_cols: bool = True,
    context_rows: int = 2,
) -> None:
    """Simple function to compare two dictionaries of DataFrames.

    Parameters
    ----------
    df_before
        the first dictionary of DataFrames to compare
    df_after
        the second dictionary of DataFrames to compare
    sort_cols
        whether to sort the columns before comparing.  Set True if the column order
        is unimportant. (Default value = True)
    context_rows
        number of rows to show around the first difference (Default value = 2)
    """
    for key in df_before:
        before = df_before[key]
        after = df_after[key]

        if sort_cols:
            before = before.sort_index(axis="columns")
            after = after.sort_index(axis="columns")

        if not before.equals(after):
            # print first line that is different, and its surrounding lines
            for i in range(len(before)):
                if not before.columns.equals(after.columns):
                    logger.warning(
                        f"Table {key} has different columns (or column order):\n"
                        f"BEFORE: {before.columns}\n"
                        f"AFTER: {after.columns}"
                    )
                    break
                if not before.iloc[i].equals(after.iloc[i]):
                    logger.warning(
                        f"Table {key} is different, first difference at row {i}:\n"
                        f"BEFORE:\n{before.iloc[i - context_rows:i + context_rows + 1]}\n"
                        f"AFTER: \n{after.iloc[i - context_rows:i + context_rows + 1]}"
                    )
                    break
        else:
            logger.success(f"Table {key} is the same")


def diff_state(
    filename_before: str, filename_after: str, sort_cols: bool = False
) -> None:
    """Diffs dataframes from two persisted state files created with save_state().

    Typical usage:
    - Save the state from a branch with a regression at some point in the transforms:
    - Switch to `main` branch and save the state from the same point:
    - Diff the two states:

    For example:

    >>> from utils import save_state, diff_state
    >>> save_state(config, tables, model, "branch.pkl.gz")
    >>> save_state(config, tables, model, "main.pkl.gz")
    >>> diff_state("branch.pkl.gz", "main.pkl.gz")
    """
    # TODO also compare config and non-dataframe model attributes?
    before = pickle.load(gzip.open(filename_before, "rb"))
    after = pickle.load(gzip.open(filename_after, "rb"))

    # Compare DFs in the tables dict
    logger.info("Comparing `table` dataframes...")
    compare_df_dict(before["tables"], after["tables"], sort_cols=sort_cols)

    # Compare DFs on the model object
    model_before = before["model"]
    model_after = after["model"]
    dfs_before = {
        a: getattr(model_before, a)
        for a in dir(model_before)
        if isinstance(getattr(model_before, a), pd.DataFrame)
    }
    dfs_after = {
        a: getattr(model_after, a)
        for a in dir(model_after)
        if isinstance(getattr(model_after, a), pd.DataFrame)
    }
    logger.info("Comparing `model` dataframes...")
    compare_df_dict(dfs_before, dfs_after, sort_cols=sort_cols)
