from pandas.core.frame import DataFrame
import pandas as pd
from dataclasses import replace
from typing import Dict, List
from more_itertools import locate, one
from itertools import groupby
import numpy
import re
import os
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from math import log10, floor
from . import datatypes


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
        df["Attribute"].fillna(varname, inplace=True)
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
    data = df[data_columns].values.tolist()
    other_columns = [
        colname for colname in df.columns.values if colname not in data_columns
    ]
    df = df[other_columns]
    value_column = "VALUE"
    df = df.assign(VALUE=data)
    nrows = df.shape[0]
    df = df.explode(value_column, ignore_index=True)

    names = pd.Series(data_columns * nrows, index=df.index, dtype=str)
    # Remove rows with no VALUE
    filter = df[value_column].notna()
    df = df[filter]
    names = names[filter]
    return df, names


def timeslices(tables: List[datatypes.EmbeddedXlTable]):
    """
    Given a list of tables with a unique table with a time slice tag, return a list
    with all the column names of that table + "ANNUAL".

    :param tables:          List of tables in EmbeddedXlTable format.
    :return:                List of column names of the unique time slice table.
    """
    # TODO merge with other timeslice code

    # No idea why casing of Weekly is special
    cols = single_table(tables, datatypes.Tag.time_slices).dataframe.columns
    timeslices = [col if col == "Weekly" else col.upper() for col in cols]
    timeslices.insert(0, "ANNUAL")
    return timeslices


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
    df: DataFrame, candidates: List[str], wildcard_col: str, output_col: str
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
        if wildcard_string == None:
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
                    current_list = list(set(current_list + additions))
            wildcard_map[wildcard_string] = current_list

    df[output_col] = df[wildcard_col].map(wildcard_map)


def missing_value_inherit(df: DataFrame, colname: str):
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
        if value == None:
            df.loc[index, colname] = last
        else:
            last = value


def get_scalar(table_tag: str, tables: List[datatypes.EmbeddedXlTable]):
    table = next(filter(lambda t: t.tag == table_tag, tables))
    if table.dataframe.shape[0] != 1 or table.dataframe.shape[1] != 1:
        raise ValueError("Not scalar table")
    return table.dataframe["VALUE"].values[0]


def has_negative_patterns(pattern):
    return pattern[0] == "-" or ",-" in pattern


def remove_negative_patterns(pattern):
    return ",".join([word for word in pattern.split(",") if word[0] != "-"])


def remove_positive_patterns(pattern):
    return ",".join([word[1:] for word in pattern.split(",") if word[0] == "-"])


def create_regexp(pattern):
    # exclude negative patterns
    if has_negative_patterns(pattern):
        pattern = remove_negative_patterns(pattern)
    if len(pattern) == 0:
        return re.compile(pattern)  # matches everything
    # escape special characters
    # Backslash must come first
    special = "\\.|^$+()[]{}"
    for c in special:
        pattern = pattern.replace(c, "\\" + c)
    # Handle VEDA wildcards
    pattern = pattern.replace("*", ".*").replace("?", ".").replace(",", "|")
    # Do not match substrings
    pattern = "^" + pattern + "$"
    return re.compile(pattern)


def create_negative_regexp(pattern):
    pattern = remove_positive_patterns(pattern)
    if len(pattern) == 0:
        pattern = "^$"  # matches nothing
    return create_regexp(pattern)


def round_sig(x, sig_figs):
    if x == 0.0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + sig_figs - 1)
