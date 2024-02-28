from __future__ import (
    annotations,
)  # see https://loguru.readthedocs.io/en/stable/api/type_hints.html#module-autodoc_stub_file.loguru

import functools
import os
import re
import sys
from dataclasses import replace
from math import log10, floor
from pathlib import Path
from typing import Iterable, List

import loguru
import numpy
import pandas as pd
from more_itertools import one
from pandas.core.frame import DataFrame

from . import datatypes

# prevent excessive number of processes in Windows and high cpu-count machines
# TODO make this a cli param or global setting?
max_workers: int = 4 if os.name == "nt" else min(16, os.cpu_count() or 16)


def apply_composite_tag(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
    """
    Handles table level declarations. Declarations can be included in the table
    tag and will apply to all data that doesn't have a different value for that
    index specified. For example, ~FI_T: DEMAND would assign DEMAND as the
    attribute for all values in the table that don't have an attribute specification
    at the column or row level. After applying the declaration this function will
    return the modified table with the simplified table tag (e.g. ~FI_T).

    See page 15 of https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
    for more context.

    :param table:      Table in EmbeddedXlTable format.
    :return:           Table in EmbeddedXlTable format with declarations applied
                       and table tag simplified.
    """
    if ":" in table.tag:
        (newtag, varname) = table.tag.split(":")
        varname = varname.strip()
        df = table.dataframe.copy()
        df["attribute"] = df["attribute"].fillna(varname)
        return replace(table, tag=newtag, dataframe=df)
    else:
        return table


def explode(df, data_columns):
    """
    Transpose the 'data_columns' in each row into a column of values, replicating the other
    columns. The name for the new column is "VALUE".

    :param df:              Dataframe to be exploded.
    :param data_columns:    Names of the columns to be exploded.
    :return:                Tuple with the exploded dataframe and a Series of the original
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
    # Remove rows with no VALUE
    index = df[value_column].notna()
    df = df[index]
    names = names[index]
    return df, names


def single_table(tables: List[datatypes.EmbeddedXlTable], tag: str):
    """
    Make sure exactly one table in 'tables' has the given table tag, and return it.
    If there are none or more than one raise an error.

    :param tables:          List of tables in EmbeddedXlTable format.
    :param tag:             Tag name.
    :return:                Table with the given tag in EmbeddedXlTable format.
    """
    return one(table for table in tables if table.tag == tag)


def single_column(tables: List[datatypes.EmbeddedXlTable], tag: str, colname: str):
    """
    Make sure exactly one table in 'tables' has the given table tag, and return the
    values for the given column name. If there are none or more than one raise an error.

    :param tables:          List of tables in EmbeddedXlTable format.
    :param tag:             Tag name.
    :param colname:         Column name to return the values of.
    :return:                Table with the given tag in EmbeddedXlTable format.
    """
    return single_table(tables, tag).dataframe[colname].values


def merge_columns(tables: List[datatypes.EmbeddedXlTable], tag: str, colname: str):
    """
    Return a list with all the values belonging to a column 'colname' from
    a table with the given tag.

    :param tables:          List of tables in EmbeddedXlTable format.
    :param tag:             Tag name to select tables
    :param colname:         Column name to select values.
    :return:                List of values for the given column name and tag.
    """
    columns = [table.dataframe[colname].values for table in tables if table.tag == tag]
    return numpy.concatenate(columns)


def apply_wildcards(
    df: DataFrame, candidates: Iterable[str], wildcard_col: str, output_col: str
):
    """
    Apply wildcards values to a list of candidates. Wildcards are values containing '*'. For example,
    a value containing '*SOLID*' would include all the values in 'candidates' containing 'SOLID' in the middle.



    :param df:              Dataframe containing all values.
    :param candidates:      List of candidate strings to apply the wildcard to.
    :param wildcard_col:    Name of column containing the wildcards.
    :param output_col:      Name of the column to dump the wildcard matches to.
    :return:                A dataframe containing all the wildcard matches on its 'output_col' column.
    """

    wildcard_map = {}
    all_wildcards = df[wildcard_col].unique()
    for wildcard_string in all_wildcards:
        if wildcard_string is None:
            wildcard_map[wildcard_string] = None
        else:
            wildcard_list = wildcard_string.split(",")
            current_list = []
            for wildcard in wildcard_list:
                if wildcard.startswith("-"):
                    wildcard = wildcard[1:]
                    regexp = re.compile(wildcard.replace("*", ".*"))
                    current_list = [s for s in current_list if not regexp.match(s)]
                else:
                    regexp = re.compile(wildcard.replace("*", ".*"))
                    additions = [s for s in candidates if regexp.match(s)]
                    current_list = sorted(set(current_list + additions))
            wildcard_map[wildcard_string] = current_list

    df[output_col] = df[wildcard_col].map(wildcard_map)


def missing_value_inherit(df: DataFrame, colname: str):
    # TODO: should we use pandas.DataFrame.fillna(method="ffill") instead?
    """
    For each None value in the specifed column of the dataframe, replace it with the last
    non-None value. If no previous non-None value is found leave it as it is. This function
    modifies the supplied dataframe and returns None.

    :param df:          Dataframe to be filled in.
    :param colname:     Name of the column to be filled in.
    :return:            None. The dataframe is filled in in place.
    """
    last = None
    for index, value in df[colname].items():
        if value is None:
            df.loc[index, colname] = last
        else:
            last = value


def get_scalar(table_tag: str, tables: List[datatypes.EmbeddedXlTable]):
    table = next(filter(lambda t: t.tag == table_tag, tables))
    if table.dataframe.shape[0] != 1 or table.dataframe.shape[1] != 1:
        raise ValueError("Not scalar table")
    return table.dataframe["value"].values[0]


def has_negative_patterns(pattern):
    return pattern[0] == "-" or ",-" in pattern


def remove_negative_patterns(pattern):
    return ",".join([word for word in pattern.split(",") if word[0] != "-"])


def remove_positive_patterns(pattern):
    return ",".join([word[1:] for word in pattern.split(",") if word[0] == "-"])


@functools.lru_cache(maxsize=int(1e6))
def create_regexp(pattern):
    # exclude negative patterns
    if has_negative_patterns(pattern):
        pattern = remove_negative_patterns(pattern)
    if len(pattern) == 0:
        return re.compile(pattern)  # matches everything
    # Handle VEDA wildcards
    pattern = pattern.replace("*", ".*").replace("?", ".").replace(",", r"$|^")
    # Do not match substrings
    pattern = rf"^{pattern}$"
    return re.compile(pattern)


@functools.lru_cache(maxsize=int(1e6))
def create_negative_regexp(pattern):
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


def get_logger(log_name: str = default_log_name, log_dir: str = ".") -> loguru.Logger:
    """Return a configured loguru logger.

    Call this once from entrypoints to set up a new logger.
    In non-entrypoint modules, just use `from loguru import logger` directly.

    To set the log level, use the `LOGURU_LEVEL` environment variable before or during runtime. E.g. `os.environ["LOGURU_LEVEL"] = "INFO"`
    Available levels are `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, and `CRITICAL`. Default is `INFO`.

    Log file will be written to `f"{log_dir}/{log_name}.log"`

    Parameters:
        log_name (str): Name of the log. Corresponding log file will be called {log_name}.log in the .
        log_dir (str): Directory to write the log file to. Default is the current working directory.
    Returns:
        Logger: A configured loguru logger.
    """
    from loguru import logger

    # set global log level via env var.  Set to INFO if not already set.
    if os.getenv("LOGURU_LEVEL") is None:
        os.environ["LOGURU_LEVEL"] = "INFO"

    log_conf = {
        "handlers": [
            {
                "sink": sys.stdout,
                "diagnose": False,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> : <level>{message}</level> (<cyan>{name}:{"
                'thread.name}:pid-{process}</cyan> "<cyan>{'
                'file.path}</cyan>:<cyan>{line}</cyan>")',
            },
            {
                "sink": f"{log_dir}/{log_name}.log",
                "enqueue": True,
                "mode": "a+",
                "level": "DEBUG",
                "colorize": False,
                "serialize": False,
                "diagnose": True,
                "rotation": "20 MB",
                "compression": "zip",
            },
        ],
    }
    logger.configure(**log_conf)
    return logger
