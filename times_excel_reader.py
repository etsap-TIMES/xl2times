from openpyxl import load_workbook
from openpyxl.worksheet.cell_range import CellRange
from pandas.core.frame import DataFrame
import pandas as pd
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, List
from more_itertools import locate, one
from itertools import groupby
import numpy
import re
import os
from concurrent.futures import ProcessPoolExecutor
from math import log10, floor
import time
from functools import reduce
from pathlib import Path
import pickle


@dataclass
class EmbeddedXlTable:
    tag: str
    uc_sets: Dict[str, str]
    sheetname: str
    range: str
    filename: str
    dataframe: DataFrame


@dataclass
class TimesXlMap:
    times_name: str
    times_cols: List[str]
    xl_name: str
    xl_cols: List[str]
    col_map: Dict[str, str]
    filter_rows: bool


class Tag(str, Enum):
    active_p_def = "~ACTIVEPDEF"
    book_regions_map = "~BOOKREGIONS_MAP"
    comemi = "~COMEMI"
    comagg = "~COMAGG"
    currencies = "~CURRENCIES"
    def_units = "~DEFUNITS"
    fi_comm = "~FI_COMM"
    fi_process = "~FI_PROCESS"
    fi_t = "~FI_T"
    start_year = "~STARTYEAR"
    tfm_ava = "~TFM_AVA"
    tfm_comgrp = "~TFM_COMGRP"
    tfm_dins = "~TFM_DINS"
    tfm_fill = "~TFM_FILL"
    tfm_fill_r = "~TFM_FILL-R"
    tfm_ins = "~TFM_INS"
    tfm_ins_ts = "~TFM_INS-TS"
    tfm_topins = "~TFM_TOPINS"
    tfm_upd = "~TFM_UPD"
    todo = "~TODO"
    time_periods = "~TIMEPERIODS"
    time_slices = "~TIMESLICES"
    uc_sets = "~UC_SETS"
    uc_t = "~UC_T"

    @classmethod
    def has_tag(cls, tag):
        return tag in cls._value2member_map_


query_columns = {
    "PSet_Set",
    "PSet_PN",
    "PSet_PD",
    "PSet_CI",
    "PSet_CO",
    "CSet_Set",
    "CSet_CN",
    "CSet_CD",
}


def extract_tables(filename: str) -> List[EmbeddedXlTable]:
    start_time = time.time()

    workbook = load_workbook(filename=filename, data_only=True)

    tables = []
    for sheet in workbook.worksheets:
        # Creating dataframe with dtype=object solves problems with ints being cast to floats
        # https://stackoverflow.com/questions/40251948/stop-pandas-from-converting-int-to-float-due-to-an-insertion-in-another-column
        df = pd.DataFrame(sheet.values, dtype=object)
        uc_sets = {}

        for row_index, row in df.iterrows():
            for colname in df.columns:
                value = str(row[colname])
                if value.startswith("~"):
                    match = re.match(f"~{Tag.uc_sets}:(.*)", value, re.IGNORECASE)
                    if match:
                        parts = match.group(1).split(":")
                        if len(parts) == 2:
                            uc_sets[parts[0].strip()] = parts[1].strip()
                        else:
                            print(
                                f"WARNING: Malformed UC_SET in {sheet.title}, {filename}"
                            )
                    else:
                        col_index = df.columns.get_loc(colname)
                        tables.append(
                            extract_table(
                                row_index, col_index, uc_sets, df, sheet.title, filename
                            )
                        )

    end_time = time.time()
    if end_time - start_time > 2:
        print(f"Loaded {filename} in {end_time-start_time:.2f} seconds")

    return tables


def extract_table(
    tag_row: int,
    tag_col: int,
    uc_sets: Dict[str, str],
    df: DataFrame,
    sheetname: str,
    filename: str,
) -> EmbeddedXlTable:
    # If the cell to the right is not empty then we read a scalar from it
    # Otherwise the row below is the header
    if not cell_is_empty(df.iloc[tag_row, tag_col + 1]):
        range = str(
            CellRange(
                min_col=tag_col + 2,
                min_row=tag_row + 1,
                max_col=tag_col + 2,
                max_row=tag_row + 1,
            )
        )
        table_df = DataFrame(columns=["VALUE"])
        table_df.loc[0] = [df.iloc[tag_row, tag_col + 1]]
        uc_sets = {}
    else:
        header_row = tag_row + 1

        start_col = tag_col
        while start_col > 0 and not cell_is_empty(df.iloc[header_row, start_col - 1]):
            start_col -= 1

        end_col = tag_col
        while end_col < df.shape[1] and not cell_is_empty(df.iloc[header_row, end_col]):
            end_col += 1

        end_row = header_row
        while end_row < df.shape[0] and not are_cells_all_empty(
            df, end_row, start_col, end_col
        ):
            end_row += 1

        # Excel cell numbering starts at 1, while pandas starts at 0
        range = str(
            CellRange(
                min_col=start_col + 1,
                min_row=header_row + 1,
                max_col=end_col + 1,
                max_row=end_row + 1,
            )
        )

        if end_row - header_row == 1 and end_col - start_col == 1:
            # Interpret single cell tables as a single data item with a column name VALUE
            table_df = DataFrame(df.iloc[header_row, start_col:end_col])
            table_df.columns = ["VALUE"]
        else:
            table_df = DataFrame(df.iloc[header_row + 1 : end_row, start_col:end_col])
            # Make all columns names strings as some are integers e.g. years
            table_df.columns = [str(x) for x in df.iloc[header_row, start_col:end_col]]

    table_df.reset_index(drop=True, inplace=True)
    table_df = table_df.applymap(
        lambda cell: cell if not isinstance(cell, float) else round_sig(cell, 15)
    )

    return EmbeddedXlTable(
        filename=filename,
        sheetname=sheetname,
        range=range,
        tag=df.iloc[tag_row, tag_col],
        uc_sets=uc_sets,
        dataframe=table_df,
    )


def are_cells_all_empty(df, row: int, start_col: int, end_col: int) -> bool:
    for col in range(start_col, end_col):
        if not cell_is_empty(df.iloc[row, col]):
            return False
    return True


def cell_is_empty(value) -> bool:
    return (
        value is None
        or (isinstance(value, numpy.float64) and numpy.isnan(value))
        or (isinstance(value, str) and len(value.strip()) == 0)
    )


def remove_comment_rows(table: EmbeddedXlTable) -> EmbeddedXlTable:
    if table.dataframe.size == 0:
        return table

    df = table.dataframe.copy()
    comment_rows = list(
        locate(
            df.iloc[:, 0],
            lambda cell: isinstance(cell, str)
            and (cell.startswith("*") or cell.startswith("\\I:")),
        )
    )
    df.drop(index=comment_rows, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # TODO tidy
    if df.shape[1] > 1:
        comment_rows = list(
            locate(
                df.iloc[:, 1],
                lambda cell: isinstance(cell, str) and cell.startswith("*"),
            )
        )
        df.drop(index=comment_rows, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return replace(table, dataframe=df)


def remove_comment_cols(table: EmbeddedXlTable) -> EmbeddedXlTable:
    if table.dataframe.size == 0:
        return table

    comment_cols = list(
        locate(
            table.dataframe.columns,
            lambda cell: isinstance(cell, str) and cell.startswith("*"),
        )
    )
    df = table.dataframe.drop(table.dataframe.columns[comment_cols], axis=1)
    df.reset_index(drop=True, inplace=True)
    seen = set()
    dupes = [x for x in df.columns if x in seen or seen.add(x)]
    if len(dupes) > 0:
        print(
            f"WARNING: Duplicate columns in {table.range}, {table.sheetname},"
            f" {table.filename}: {','.join(dupes)}"
        )
    return replace(table, dataframe=df)


def remove_tables_with_formulas(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    def is_formula(s):
        return isinstance(s, str) and len(s) > 0 and s[0] == "="

    def has_formulas(table):
        has = table.dataframe.applymap(is_formula).any(axis=None)
        if has:
            print(f"WARNING: Excluding table {table.tag} because it has formulas")
        return has

    return [table for table in tables if not has_formulas(table)]


def normalize_tags_columns_attrs(
    tables: List[EmbeddedXlTable],
) -> List[EmbeddedXlTable]:
    """Normalize (uppercase) tags, column names, and values in attribute columns."""
    # TODO Uppercase column names and attribute values in mapping.txt when reading it
    # TODO Check all string literals left in file
    def normalize(table: EmbeddedXlTable) -> EmbeddedXlTable:
        # Only uppercase upto ':', the rest can be non-uppercase values like regions
        parts = table.tag.split(":")
        # assert len(parts) <= 2
        parts[0] = parts[0].upper()
        newtag = ":".join(parts)

        # TODO continue:
        # df = table.dataframe
        # col_name_map = {x: x.upper() for x in df.columns}

        # return replace(table, tag=newtag, dataframe=df)
        return replace(table, tag=newtag)

    return [normalize(table) for table in tables]


def merge_tables(tables: List[EmbeddedXlTable]) -> Dict[str, DataFrame]:
    """Merge tables of the same types"""
    result = {}
    for key, value in groupby(sorted(tables, key=lambda t: t.tag), lambda t: t.tag):
        group = list(value)
        if not all(
            set(t.dataframe.columns) == set(group[0].dataframe.columns) for t in group
        ):
            cols = [(",".join(g.dataframe.columns.values), g) for g in group]
            cols_groups = [
                (key, list(group))
                for key, group in groupby(
                    sorted(cols, key=lambda ct: ct[0]), lambda ct: ct[0]
                )
            ]
            print(
                f"WARNING: Cannot merge tables with tag {key} as their columns are not identical"
            )
            for (c, table) in cols:
                print(f"  {c} from {table.range}, {table.sheetname}, {table.filename}")
        else:
            df = pd.concat([table.dataframe for table in group], ignore_index=True)
            if "Year" in df.columns.values:
                df["Year"] = df["Year"].astype("Int64")
            result[key] = df
    return result


def apply_composite_tag(table: EmbeddedXlTable) -> EmbeddedXlTable:
    """Process composite tags e.g. ~FI_T: COM_PKRSV"""
    if ":" in table.tag:
        (newtag, varname) = table.tag.split(":")
        varname = varname.strip()
        df = table.dataframe.copy()
        df["Attribute"].fillna(varname, inplace=True)
        return replace(table, tag=newtag, dataframe=df)
    else:
        return table


def explode(df, data_columns):
    data = df[data_columns].values.tolist()
    other_columns = [
        colname for colname in df.columns.values if colname not in data_columns
    ]
    df = df[other_columns]
    value_column = "VALUE"
    # assign is needed here because we want to make a copy
    df = df.assign(VALUE=data)
    nrows = df.shape[0]
    df = df.explode(value_column, ignore_index=True)

    names = pd.Series(data_columns * nrows, index=df.index, dtype=str)
    # Remove rows with no VALUE
    filter = df[value_column].notna()
    df = df[filter]
    names = names[filter]
    return df, names


def timeslices(tables: List[EmbeddedXlTable]):
    # TODO merge with other timeslice code

    # No idea why casing of Weekly is special
    cols = single_table(tables, Tag.time_slices).dataframe.columns
    timeslices = [col if col == "Weekly" else col.upper() for col in cols]
    timeslices.insert(0, "ANNUAL")
    return timeslices


def process_flexible_import_tables(
    tables: List[EmbeddedXlTable],
) -> List[EmbeddedXlTable]:
    legal_values = {
        "LimType": {"LO", "UP", "FX"},
        "TimeSlice": timeslices(tables),
        "Comm-OUT": set(merge_columns(tables, Tag.fi_comm, "CommName")),
        "Region": single_column(tables, Tag.book_regions_map, "Region"),
        "Curr": single_column(tables, Tag.currencies, "Currency"),
        "Other_Indexes": {"Input", "Output"},
    }

    def get_colname(value):
        # TODO make sure to do case-insensitive comparisons when parsing composite column names
        if value.isdigit():
            return "Year", int(value)
        for name, values in legal_values.items():
            if value in values:
                return name, value
        return None, value

    def process_flexible_import_table(table: EmbeddedXlTable) -> EmbeddedXlTable:
        # See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16

        if not table.tag.startswith(Tag.fi_t) and table.tag not in {
            Tag.tfm_upd,
        }:
            return table
        df = table.dataframe
        mapping = {"YEAR": "Year", "CommName": "Comm-IN"}
        df = df.rename(columns=mapping)

        if "CURR" in df.columns.values:
            df.rename(columns={"CURR": "Curr"}, inplace=True)

        nrows = df.shape[0]
        if (
            ("Comm-IN" in df.columns)
            and ("Comm-OUT" in df.columns)
            and (table.tag != Tag.tfm_upd)
        ):
            df["TOP-IN"] = ["IN"] * nrows
            df["TOP-OUT"] = ["OUT"] * nrows

        # Remove any TechDesc column
        if "TechDesc" in df.columns:
            df.drop("TechDesc", axis=1, inplace=True)

        if "CommGrp" in df.columns:
            print(
                f"WARNING: Dropping CommGrp rather than processing it: {table.filename} {table.sheetname} {table.range}"
            )
            df.drop("CommGrp", axis=1, inplace=True)

        # Tag column no longer used to identify data columns
        # https://veda-documentation.readthedocs.io/en/latest/pages/introduction.html#veda2-0-enhanced-features
        known_columns = [
            "Region",
            "TechName",
            "Comm-IN",
            "Comm-IN-A",
            "Comm-OUT",
            "Comm-OUT-A",
            "Attribute",
            "Year",
            "TimeSlice",
            "LimType",
            "Curr",
            "Other_Indexes",
            "Stage",
            "SOW",
            "CommGrp",
        ]
        data_columns = [x for x in df.columns.values if x not in known_columns]

        # Populate index columns
        index_columns = [
            "Region",
            "TechName",
            "Comm-IN",
            "Comm-IN-A",
            "Comm-OUT",
            "Comm-OUT-A",
            "Attribute",
            "Year",
            "TimeSlice",
            "LimType",
            "Curr",
            "Other_Indexes",
        ]
        for colname in index_columns:
            if colname not in df.columns:
                df[colname] = [None] * nrows
        table = replace(table, dataframe=df)

        table = apply_composite_tag(table)
        df = table.dataframe

        attribute = "Attribute"
        if table.tag != Tag.tfm_upd:
            df, attribute_suffix = explode(df, data_columns)

            # Append the data column name to the Attribute column
            if nrows > 0:
                i = df[attribute].notna()
                df.loc[i, attribute] = df.loc[i, attribute] + "~" + attribute_suffix[i]
                i = df[attribute].isna()
                df.loc[i, attribute] = attribute_suffix[i]

        # Handle Attribute containing tilde, such as 'STOCK~2030'
        for attr in df[attribute].unique():
            if "~" in attr:
                i = df[attribute] == attr
                parts = attr.split("~")
                for value in parts:
                    colname, typed_value = get_colname(value)
                    if colname is None:
                        df.loc[i, attribute] = typed_value
                    else:
                        df.loc[i, colname] = typed_value

        # Handle Other_Indexes
        other = "Other_Indexes"
        for attr in df[attribute].unique():
            if attr == "FLO_EMIS":
                i = df[attribute] == attr
                df.loc[i & df[other].isna(), other] = "ACT"
            elif attr == "EFF":
                i = df[attribute] == attr
                df.loc[i, "Comm-IN"] = "ACT"
                df.loc[i, attribute] = "CEFF"
            elif attr == "OUTPUT":
                i = df[attribute] == attr
                df.loc[i, "Comm-IN"] = df.loc[i, "Comm-OUT-A"]
                df.loc[i, attribute] = "CEFF"
            elif attr == "END":
                i = df[attribute] == attr
                df.loc[i, "Year"] = df.loc[i, "VALUE"].astype("int") + 1
                df.loc[i, other] = "EOH"
                df.loc[i, attribute] = "PRC_NOFF"
            elif attr == "TOP-IN":
                i = df[attribute] == attr
                df.loc[i, other] = df.loc[i, "Comm-IN"]
                df.loc[i, attribute] = "IO"
            elif attr == "TOP-OUT":
                i = df[attribute] == attr
                df.loc[i, other] = df.loc[i, "Comm-OUT"]
                df.loc[i, attribute] = "IO"
        filter = ~((df[attribute] == "IO") & df[other].isna())
        df = df[filter]
        df = df.reset_index(drop=True)

        # Should have all index_columns and VALUE
        if table.tag == Tag.fi_t and len(df.columns) != (len(index_columns) + 1):
            raise ValueError(f"len(df.columns) = {len(df.columns)}")

        # TODO: should we have a global list of column name -> type?
        df["Year"] = df["Year"].astype("Int64")

        return replace(table, dataframe=df)

    return [process_flexible_import_table(t) for t in tables]


def process_user_constraint_tables(
    tables: List[EmbeddedXlTable],
) -> List[EmbeddedXlTable]:
    legal_values = {
        # TODO: load these from mapping file?
        "Attribute": {
            "UC_ACT",
            "UC_ATTR",
            "UC_CAP",
            "UC_COMNET",
            "UC_COMPRD",
            "UC_FLO",
            "UC_NCAP",
            "UC_N",
            "UC_RHSRT",
            "UC_RHSRTS",
            "UC_RHSTS",
            "UC_R_EACH",
            "UC_R_SUM",
        },
        "Region": single_column(tables, Tag.book_regions_map, "Region"),
        "LimType": {"LO", "UP"},
        "Side": {"LHS", "RHS"},
    }

    commodity_names = set(merge_columns(tables, Tag.fi_comm, "CommName"))
    process_names = set(merge_columns(tables, Tag.fi_process, "TechName"))

    def get_colname(value):
        # TODO make sure to do case-insensitive comparisons when parsing composite column names
        if value.isdigit():
            return "Year", int(value)
        for name, values in legal_values.items():
            if value in values:
                return name, value
        return None, value

    def process_user_constraint_table(table: EmbeddedXlTable) -> EmbeddedXlTable:
        # See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16

        if not table.tag.startswith(Tag.uc_t):
            return table
        df = table.dataframe

        # TODO: apply table.uc_sets

        # Fill in UC_N blank cells with value from above
        df["UC_N"] = df["UC_N"].ffill()

        if "UC_Desc" in df.columns:
            df.drop("UC_Desc", axis=1, inplace=True)

        if "CSET_CN" in table.dataframe.columns.values:
            df.rename(columns={"CSET_CN": "Cset_CN"}, inplace=True)

        known_columns = [
            "UC_N",
            "Region",
            "Pset_Set",
            "Pset_PN",
            "Pset_PD",
            "Pset_CI",
            "Pset_CO",
            "Cset_CN",
            "Cset_CD",
            "Side",
            "Attribute",
            "UC_ATTR",
            "Year",
            "LimType",
            "Top_Check",
            # TODO remove these?
            "TimeSlice",
            "CommName",
            "TechName",
        ]
        data_columns = [x for x in df.columns.values if x not in known_columns]

        # Populate columns
        nrows = df.shape[0]
        for colname in known_columns:
            if colname not in df.columns:
                df[colname] = [None] * nrows
        table = replace(table, dataframe=df)

        # TODO: detect RHS correctly
        i = df["Side"].isna()
        df.loc[i, "Side"] = "LHS"

        table = apply_composite_tag(table)
        df = table.dataframe
        df, attribute_suffix = explode(df, data_columns)

        # Append the data column name to the Attribute column
        if nrows > 0:
            i = df["Attribute"].notna()
            df.loc[i, "Attribute"] = df.loc[i, "Attribute"] + "~" + attribute_suffix[i]
            i = df["Attribute"].isna()
            df.loc[i, "Attribute"] = attribute_suffix[i]

        # Handle Attribute containing tilde, such as 'STOCK~2030'
        for attr in df["Attribute"].unique():
            if "~" in attr:
                i = df["Attribute"] == attr
                parts = attr.split("~")
                for value in parts:
                    colname, typed_value = get_colname(value)
                    if colname is None:
                        df.loc[i, "Attribute"] = typed_value
                    else:
                        df.loc[i, colname] = typed_value

        apply_wildcards(df, commodity_names, "Cset_CN", "CommName")
        df.drop("Cset_CN", axis=1, inplace=True)
        df = df.explode(["CommName"], ignore_index=True)

        apply_wildcards(df, process_names, "Pset_PN", "TechName")
        df.drop("Pset_PN", axis=1, inplace=True)
        df = df.explode(["TechName"], ignore_index=True)

        # TODO: handle other wildcard columns

        # TODO: should we have a global list of column name -> type?
        df["Year"] = df["Year"].astype("Int64")

        return replace(table, dataframe=df)

    return [process_user_constraint_table(t) for t in tables]


def apply_wildcards(
    df: DataFrame, candidates: List[str], wildcard_col: str, output_col: str
):
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


def merge_columns(tables: List[EmbeddedXlTable], tag: str, colname: str):
    columns = [table.dataframe[colname].values for table in tables if table.tag == tag]
    return numpy.concatenate(columns)


def single_table(tables: List[EmbeddedXlTable], tag: str):
    return one(table for table in tables if table.tag == tag)


def single_column(tables: List[EmbeddedXlTable], tag: str, colname: str):
    return single_table(tables, tag).dataframe[colname].values


def fill_in_missing_values(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    regions = single_column(tables, Tag.book_regions_map, "Region")
    start_year = one(single_column(tables, Tag.start_year, "VALUE"))
    # TODO there are multiple currencies
    currency = single_column(tables, Tag.currencies, "Currency")[0]

    def fill_in_missing_values_inplace(df):
        for colname in df.columns:
            # TODO make this more declarative
            if colname == "Csets" or colname == "TechName":
                missing_value_inherit(df, colname)
            elif colname == "LimType" and table.tag == Tag.fi_comm and False:
                isna = df[colname].isna()
                ismat = df["Csets"] == "MAT"
                df.loc[isna & ismat, colname] = "FX"
                df.loc[isna & ~ismat, colname] = "LO"
            elif colname == "LimType" and (
                table.tag == Tag.fi_t or table.tag.startswith("~TFM")
            ):
                isna = df[colname].isna()
                islo = df["Attribute"].isin({"BS_STIME", "GR_VARGEN", "RCAP_BND"})
                isfx = df["Attribute"].isin(
                    {
                        "ACT_LOSPL",
                        "FLO_SHAR",
                        "MARKAL-REH",
                        "NCAP_CHPR",
                        "VA_Attrib_C",
                        "VA_Attrib_T",
                        "VA_Attrib_TC",
                    }
                )
                df.loc[isna & islo, colname] = "LO"
                df.loc[isna & isfx, colname] = "FX"
                df.loc[isna & ~islo & ~isfx, colname] = "UP"
            elif (
                colname == "TimeSlice" or colname == "Tslvl"
            ):  # or colname == "CTSLvl" or colname == "PeakTS":
                df[colname].fillna(
                    "ANNUAL", inplace=True
                )  # ACT_CSTUP should use DAYNITE
            elif colname == "Region":
                df[colname].fillna(",".join(regions), inplace=True)
            elif colname == "Year":
                df[colname].fillna(start_year, inplace=True)
            elif colname == "Curr":
                df[colname].fillna(currency, inplace=True)

    for table in tables:
        if table.tag == Tag.tfm_upd:
            # Missing values in update tables are wildcards and should not be filled in
            result.append(table)
        else:
            df = table.dataframe.copy()
            fill_in_missing_values_inplace(df)
            result.append(replace(table, dataframe=df))
    return result


def missing_value_inherit(df: DataFrame, colname: str):
    last = None
    for index, value in df[colname].items():
        if value == None:
            df.loc[index, colname] = last
        else:
            last = value


def expand_rows(table: EmbeddedXlTable) -> EmbeddedXlTable:
    """Expand out certain columns with entries containing commas"""

    def has_comma(s):
        return isinstance(s, str) and "," in s

    def split_by_commas(s):
        if has_comma(s):
            return [x.strip() for x in s.split(",")]
        else:
            return s

    df = table.dataframe.copy()
    c = df.applymap(has_comma)
    columns_with_commas = [
        colname
        for colname in c.columns.values
        if colname not in query_columns and c[colname].any()
    ]
    if len(columns_with_commas) > 0:
        # Transform comma-separated strings into lists
        df[columns_with_commas] = df[columns_with_commas].applymap(split_by_commas)
        for colname in columns_with_commas:
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html#pandas.DataFrame.explode
            df = df.explode(colname, ignore_index=True)
    return replace(table, dataframe=df)


def remove_invalid_values(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    # TODO pull this out
    regions = single_column(tables, Tag.book_regions_map, "Region")
    # TODO pull this out
    constraints = {
        "Csets": {"NRG", "MAT", "DEM", "ENV", "FIN"},
        "Region": regions,
    }

    result = []
    for table in tables:
        df = table.dataframe.copy()
        is_valid_list = [
            df[colname].isin(values)
            for colname, values in constraints.items()
            if colname in df.columns
        ]
        if is_valid_list:
            is_valid = reduce(lambda a, b: a & b, is_valid_list)
            df = df[is_valid]
            df.reset_index(drop=True, inplace=True)
        result.append(replace(table, dataframe=df))
    return result


def read_mappings(filename: str) -> List[TimesXlMap]:
    mappings = []
    dropped = []
    with open(filename) as file:
        while True:
            line = file.readline().rstrip()
            if line == "":
                break
            (times, xl) = line.split(" = ")
            (times_name, times_cols_str) = list(filter(None, re.split("\[|\]", times)))
            (xl_name, xl_cols_str) = list(filter(None, re.split("\(|\)", xl)))
            times_cols = times_cols_str.split(",")
            xl_cols = xl_cols_str.split(",")
            filter_rows = False
            if xl_cols[-1].startswith("Attribute:"):
                xl_cols[-1] = xl_cols[-1].replace("Attribute:", "")
                filter_rows = True
            col_map = {}
            for index, value in enumerate(xl_cols):
                col_map[value] = times_cols[index]
            # Uppercase and validate tags:
            if xl_name.startswith("~"):
                xl_name = xl_name.upper()
                assert Tag.has_tag(xl_name), f"Tag {xl_name} not found"
            entry = TimesXlMap(
                times_name=times_name,
                times_cols=times_cols,
                xl_name=xl_name,
                xl_cols=xl_cols,
                col_map=col_map,
                filter_rows=filter_rows,
            )

            # TODO remove: Filter out mappings that are not yet finished
            if entry.xl_name != Tag.todo and not any(
                c.startswith("TODO") for c in entry.xl_cols
            ):
                mappings.append(entry)
            else:
                dropped.append(line)

    if len(dropped) > 0:
        print(f"WARNING: Dropping {len(dropped)} mappings that are not yet complete")
    return mappings


def process_time_periods(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    start_year = get_scalar(Tag.start_year, tables)
    active_pdef = get_scalar(Tag.active_p_def, tables)

    def process_time_periods_table(table: EmbeddedXlTable):
        if table.tag != Tag.time_periods:
            return table

        df = table.dataframe.copy()
        active_series = df[active_pdef]
        # Remove empty rows
        active_series.dropna(inplace=True)

        df = pd.DataFrame({"D": active_series})
        # Start years = start year, then cumulative sum of period durations
        df["B"] = (active_series.cumsum() + start_year).shift(1, fill_value=start_year)
        df["E"] = df.B + df.D - 1
        df["M"] = df.B + ((df.D - 1) // 2)
        df["Year"] = df.M

        return replace(table, dataframe=df.astype(int))

    return [process_time_periods_table(table) for table in tables]


def process_currencies(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    def process_currencies_table(table: EmbeddedXlTable):
        if "Curr" not in table.dataframe.columns:
            return table

        df = table.dataframe.copy()

        # TODO: work out how to implement this correctly, EUR18 etc. do not appear in raw tables
        df["Curr"] = df["Curr"].apply(
            lambda x: None if x is None else x.replace("MEUR20", "EUR")
        )

        return replace(table, dataframe=df)

    return [process_currencies_table(table) for table in tables]


def apply_fixups(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    def apply_fixups_table(table: EmbeddedXlTable):
        if not table.tag.startswith(Tag.fi_t):
            return table

        df = table.dataframe.copy()

        i = df["Attribute"].str.upper() == "FLO_SHAR"
        df.loc[i, "Comm-IN"] = df["Comm-OUT"]

        # Append _NRGI (energy input) to some cells in FLO_SHAR
        i = (df["Attribute"].str.lower() == "share-i") & (
            (df["LimType"] == "UP") | (df["LimType"] == "LO")
        )
        # TODO: looks like NRG may come from ~TFM_Csets
        df.loc[i, "Other_Indexes"] = df["TechName"].astype(str) + "_NRGI"

        # TODO allow multiple columns in mapping
        df["Attribute"] = df["Attribute"].str.replace(
            "Share-I", "FLO_SHAR", case=False, regex=False
        )

        return replace(table, dataframe=df)

    return [apply_fixups_table(table) for table in tables]


def extract_commodity_groups(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    fit_tables = [t for t in tables if t.tag == Tag.fi_t]

    gmap_tables = []
    for fit_table in fit_tables:
        # TODO: looks like NRG may come from ~TFM_Csets
        inputs = fit_table.dataframe[["Region", "TechName", "Comm-IN"]].copy()
        inputs["TechName"] = inputs["TechName"].astype(str) + "_NRGI"
        inputs.rename(columns={"Comm-IN": "Comm"}, inplace=True)

        # TODO: looks like NRG may come from ~TFM_Csets
        outputs = fit_table.dataframe[["Region", "TechName", "Comm-OUT"]].copy()
        outputs["TechName"] = outputs["TechName"].astype(str) + "_NRGO"
        outputs.rename(columns={"Comm-OUT": "Comm"}, inplace=True)

        # TODO: looks like techs in DEMO group are specified via CommGrp column in VT_IE_TRA.xlsx ACT2FLO B6:AF33
        demo = fit_table.dataframe[["Region", "TechName", "Comm-OUT"]].copy()
        demo["TechName"] = demo["TechName"].astype(str) + "_DEMO"
        demo.rename(columns={"Comm-OUT": "Comm"}, inplace=True)

        gmap_tables += [inputs, outputs, demo]

    merged = pd.concat(gmap_tables, ignore_index=True, sort=False)

    # TODO apply renamings from ~TFM_TOPINS e.g. RSDAHT to RSDAHT2

    tables.append(
        EmbeddedXlTable(
            sheetname="",
            range="",
            filename="",
            uc_sets="",
            tag="COM_GMAP",
            dataframe=merged,
        )
    )
    return tables


def get_scalar(table_tag: str, tables: List[EmbeddedXlTable]):
    table = next(filter(lambda t: t.tag == table_tag, tables))
    if table.dataframe.shape[0] != 1 or table.dataframe.shape[1] != 1:
        raise ValueError("Not scalar table")
    return table.dataframe["VALUE"].values[0]


def remove_fill_tables(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    # These tables collect data from elsewhere and update the table itself or a region below
    # The collected data is then presumably consumed via Excel references or vlookups
    # TODO for the moment, assume VEDA has updated these tables but we will need a tool to do this
    result = []
    for table in tables:
        if table.tag != Tag.tfm_fill and not table.tag.startswith(Tag.tfm_fill_r):
            result.append(table)
    return result


def process_commodity_emissions(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    regions = single_column(tables, Tag.book_regions_map, "Region")

    result = []
    for table in tables:
        if table.tag != Tag.comemi:
            result.append(table)
        else:
            df = table.dataframe.copy()
            index_columns = ["Region", "Year", "CommName"]
            data_columns = [
                colname for colname in df.columns.values if colname not in index_columns
            ]
            df, names = explode(df, data_columns)
            df.rename(columns={"VALUE": "EMCB"}, inplace=True)
            df["Other_Indexes"] = names

            if "Region" in df.columns.values:
                df = df.astype({"Region": "string"})
                df["Region"] = df["Region"].map(lambda s: s.split(","))
                df = df.explode("Region", ignore_index=True)
                df = df[df["Region"].isin(regions)]

            nrows = df.shape[0]
            for colname in index_columns:
                if colname not in df.columns:
                    df[colname] = [None] * nrows

            result.append(replace(table, dataframe=df))

    return result


def process_commodities(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    regions = ",".join(single_column(tables, Tag.book_regions_map, "Region"))

    result = []
    for table in tables:
        if table.tag != Tag.fi_comm and table.tag != Tag.fi_comm:
            result.append(table)
        else:
            df = table.dataframe.copy()
            nrows = df.shape[0]
            if "Region" not in table.dataframe.columns.values:
                df.insert(1, "Region", [regions] * nrows)
            if "LimType" not in table.dataframe.columns.values:
                df["LimType"] = [None] * nrows
            if "CSet" in table.dataframe.columns.values:
                df = df.rename(columns={"CSet": "Csets"})
            result.append(replace(table, dataframe=df, tag=Tag.fi_comm))

    return result


def process_years(tables: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    # Datayears is the set of all years in ~FI_T's Year column
    # We ignore values < 1000 because those signify interpolation/extrapolation rules
    # (see Table 8 of Part IV of the Times Documentation)
    datayears = tables[Tag.fi_t]["Year"].where(lambda x: x >= 1000).dropna()
    datayears = datayears.drop_duplicates().sort_values()
    tables["DataYear"] = pd.DataFrame({"Year": datayears})

    # Pastyears is the set of all years before ~StartYear
    start_year = tables[Tag.start_year]["VALUE"][0]
    pastyears = datayears.where(lambda x: x <= start_year).dropna()
    tables["PastYear"] = pd.DataFrame({"Year": pastyears})

    # Modelyears is the union of pastyears and the representative years of the model (middleyears)
    modelyears = pastyears.combine_first(tables[Tag.time_periods]["M"]).drop_duplicates().sort_values()
    tables["ModelYear"] = pd.DataFrame({"Year": modelyears})

    return tables


def process_processes(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    for table in tables:
        if table.tag != Tag.fi_process:
            result.append(table)
        else:
            df = table.dataframe.copy()
            nrows = df.shape[0]
            if "Vintage" not in table.dataframe.columns.values:
                df["Vintage"] = [None] * nrows
            if "Region" not in table.dataframe.columns.values:
                df.insert(1, "Region", [None] * nrows)
            if "Tslvl" not in table.dataframe.columns.values:
                df.insert(6, "Tslvl", ["ANNUAL"] * nrows)
            result.append(replace(table, dataframe=df))

    return result


def process_transform_insert(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    regions = single_column(tables, Tag.book_regions_map, "Region")
    tfm_tags = [Tag.tfm_ins, Tag.tfm_dins, Tag.tfm_topins, Tag.tfm_upd, Tag.tfm_comgrp]

    result = []
    dropped = []
    for table in tables:
        if not any(table.tag.startswith(t) for t in tfm_tags):
            result.append(table)

        elif table.tag in [Tag.tfm_ins, Tag.tfm_ins_ts, Tag.tfm_upd, Tag.tfm_comgrp]:
            df = table.dataframe.copy()
            nrows = df.shape[0]

            # Standardize column names
            known_columns = {
                "Attribute",
                "Year",
                "TimeSlice",
                "LimType",
                "CommGrp",
                "Curr",
                "Stage",
                "SOW",
                "Other_Indexes",
                "AllRegions",
            } | query_columns
            lowercase_cols = df.columns.map(lambda x: x.casefold())
            colmap = {}
            for standard_col in known_columns:
                lowercase_col = standard_col.casefold()
                if lowercase_col in lowercase_cols:
                    i = lowercase_cols.get_loc(lowercase_col)
                    colmap[df.columns[i]] = standard_col
            df.rename(columns=colmap, inplace=True)

            if "AllRegions" in df.columns:
                for region in regions:
                    df[region] = df["AllRegions"]
                df.drop(columns=["AllRegions"], inplace=True)

            if table.tag == Tag.tfm_ins_ts:
                # ~TFM_INS-TS: Regions should be specified in a column with header=Region and columns in data area are YEARS
                if "Region" not in df.columns.values:
                    df["Region"] = [regions] * len(df)
                    df = df.explode(["Region"], ignore_index=True)
            else:
                # Transpose region columns to new VALUE column and add corresponding regions in new Region column
                region_cols = [
                    col_name for col_name in df.columns.values if col_name in regions
                ]
                other_columns = [
                    col_name
                    for col_name in df.columns.values
                    if col_name not in regions
                ]
                data = df[region_cols].values.tolist()
                df = df[other_columns]
                df["Region"] = [region_cols] * nrows
                df["VALUE"] = data
                df = df.explode(["Region", "VALUE"], ignore_index=True)
                unknown_columns = [
                    col_name
                    for col_name in df.columns.values
                    if col_name not in known_columns | {"Region", "VALUE"}
                ]
                df.drop(columns=unknown_columns, inplace=True)

            def has_no_wildcards(list):
                return all(
                    list.apply(
                        lambda x: x is not None
                        and x[0] != "-"
                        and "*" not in x
                        and "," not in x
                        and "?" not in x
                    )
                )

            if (
                table.tag == Tag.tfm_ins_ts
                and set(df.columns) & query_columns == {"CSet_CN"}
                and has_no_wildcards(df["CSet_CN"])
            ):
                df["Comm-OUT"] = df["CSet_CN"]
                df["Comm-IN"] = df["CSet_CN"]
                df.drop(columns=["CSet_CN"], inplace=True)
                result.append(replace(table, dataframe=df, tag=Tag.fi_t))
            elif (
                table.tag == Tag.tfm_ins_ts
                and set(df.columns) & query_columns == {"PSet_PN"}
                and has_no_wildcards(df["PSet_PN"])
            ):
                df.rename(columns={"PSet_PN": "TechName"}, inplace=True)
                result.append(replace(table, dataframe=df, tag=Tag.fi_t))
            else:
                # wildcard expansion will happen later
                if table.tag == Tag.tfm_ins_ts:
                    # ~TFM_INS-TS: Regions should be specified in a column with header=Region and columns in data area are YEARS
                    data_columns = [
                        colname
                        for colname in df.columns.values
                        if colname not in known_columns | {"Region", "TS_Filter"}
                    ]
                    df, years = explode(df, data_columns)
                    df["Year"] = years
                for standard_col in known_columns:
                    if standard_col not in df.columns:
                        df[standard_col] = [None] * len(df)
                result.append(replace(table, dataframe=df))

        elif table.tag == Tag.tfm_dins:
            df = table.dataframe.copy()
            nrows = df.shape[0]

            # Find all columns with -, first part is region and sum over second part
            pairs = [(col.split("-")[0], col) for col in df.columns if "-" in col]
            for region, tup in groupby(
                sorted(pairs, key=lambda p: p[0]), lambda p: p[0]
            ):
                cols = [t[1] for t in tup]
                df[region] = df.loc[:, cols].sum(axis=1)
                df[region] = df[region].apply(lambda x: round_sig(x, 15))
                df.drop(columns=cols, inplace=True)

            # Transpose region columns to new DEMAND column and add corresponding regions in new Region column
            region_cols = [
                col_name for col_name in df.columns.values if col_name in regions
            ]
            other_columns = [
                col_name for col_name in df.columns.values if col_name not in regions
            ]
            data = df[region_cols].values.tolist()
            df = df[other_columns]
            df["Region"] = [region_cols] * nrows
            df["DEMAND"] = data
            df = df.explode(["Region", "DEMAND"], ignore_index=True)

            df.rename(columns={"Cset_CN": "Comm-IN"}, inplace=True)

            result.append(replace(table, dataframe=df, tag=Tag.fi_t))

        else:
            dropped.append(table)

    if len(dropped) > 0:
        # TODO handle
        by_tag = [
            (key, list(group))
            for key, group in groupby(
                sorted(dropped, key=lambda t: t.tag), lambda t: t.tag
            )
        ]
        for (key, group) in by_tag:
            print(
                f"WARNING: Dropped {len(group)} transform insert tables ({key})"
                f" rather than processing them"
            )

    return result


def process_transform_availability(
    tables: List[EmbeddedXlTable],
) -> List[EmbeddedXlTable]:
    result = []
    dropped = []
    for table in tables:
        if table.tag != Tag.tfm_ava:
            result.append(table)
        else:
            dropped.append(table)

    if len(dropped) > 0:
        # TODO handle
        by_tag = [
            (key, list(group))
            for key, group in groupby(
                sorted(dropped, key=lambda t: t.tag), lambda t: t.tag
            )
        ]
        for (key, group) in by_tag:
            print(
                f"WARNING: Dropped {len(group)} transform availability tables ({key})"
                f" rather than processing them"
            )

    return result


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
    pattern = pattern.replace("*", ".*").replace("?", ".")
    # Do not match substrings
    pattern = "|".join(["^" + word + "$" for word in pattern.split(",")])
    return re.compile(pattern)


def create_negative_regexp(pattern):
    pattern = remove_positive_patterns(pattern)
    if len(pattern) == 0:
        pattern = "^$"  # matches nothing
    return create_regexp(pattern)


def process_wildcards(tables: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    # We need to be able to fetch processes based on any combination of name, description, set, comm-in, or comm-out
    # So we construct tables whose indices are names, etc. and use pd.filter
    processes = tables[Tag.fi_process]
    duplicated_processes = processes[["TechName"]].duplicated()
    if any(duplicated_processes):
        duplicated_process_names = processes["TechName"][duplicated_processes]
        print(
            f"WARNING: {len(duplicated_process_names)} duplicated processes: {duplicated_process_names.values[1:3]}"
        )
        processes.drop_duplicates(subset="TechName", inplace=True)
    processes_by_name = (
        processes[["TechName"]]
        .dropna()
        .set_index("TechName", drop=False)
        .rename_axis("index")
    )
    processes_by_desc = (
        processes[["TechName", "TechDesc"]].dropna().set_index("TechDesc")
    )
    processes_by_sets = processes[["TechName", "Sets"]].dropna().set_index("Sets")
    processes_and_commodities = tables[Tag.fi_t]
    processes_by_comm_in = (
        processes_and_commodities[["TechName", "Comm-IN"]]
        .dropna()
        .drop_duplicates()
        .set_index("Comm-IN")
    )
    processes_by_comm_out = (
        processes_and_commodities[["TechName", "Comm-OUT"]]
        .dropna()
        .drop_duplicates()
        .set_index("Comm-OUT")
    )
    commodities = tables[Tag.fi_comm]
    commodities_by_name = (
        commodities[["CommName"]]
        .dropna()
        .set_index("CommName", drop=False)
        .rename_axis("index")
    )
    commodities_by_desc = (
        commodities[["CommName", "CommDesc"]].dropna().set_index("CommDesc")
    )
    commodities_by_sets = commodities[["CommName", "Csets"]].dropna().set_index("Csets")

    def filter_by_pattern(df, pattern):
        # Duplicates can be created when a process has multiple commodities that match the pattern
        df = df.filter(regex=create_regexp(pattern), axis="index").drop_duplicates()
        exclude = df.filter(regex=create_negative_regexp(pattern), axis="index").index
        return df.drop(exclude)

    def intersect(acc, df):
        if acc is None:
            return df
        return acc.merge(df)

    def get_matching_processes(row):
        matching_processes = None
        if row.PSet_PN is not None:
            matching_processes = intersect(
                matching_processes, filter_by_pattern(processes_by_name, row.PSet_PN)
            )
        if row.PSet_PD is not None:
            matching_processes = intersect(
                matching_processes, filter_by_pattern(processes_by_desc, row.PSet_PD)
            )
        if row.PSet_Set is not None:
            matching_processes = intersect(
                matching_processes, filter_by_pattern(processes_by_sets, row.PSet_Set)
            )
        if row.PSet_CI is not None:
            matching_processes = intersect(
                matching_processes, filter_by_pattern(processes_by_comm_in, row.PSet_CI)
            )
        if row.PSet_CO is not None:
            matching_processes = intersect(
                matching_processes,
                filter_by_pattern(processes_by_comm_out, row.PSet_CO),
            )
        if matching_processes is not None and any(matching_processes.duplicated()):
            raise ValueError("duplicated")
        return matching_processes

    def get_matching_commodities(row):
        matching_commodities = None
        if row.CSet_CN is not None:
            matching_commodities = intersect(
                matching_commodities,
                filter_by_pattern(commodities_by_name, row.CSet_CN),
            )
        if row.CSet_CD is not None:
            matching_commodities = intersect(
                matching_commodities,
                filter_by_pattern(commodities_by_desc, row.CSet_CD),
            )
        if row.CSet_Set is not None:
            matching_commodities = intersect(
                matching_commodities,
                filter_by_pattern(commodities_by_sets, row.CSet_Set),
            )
        return matching_commodities

    for tag in {Tag.tfm_ins, Tag.tfm_ins_ts, Tag.tfm_upd}:
        if tag in tables:
            start_time = time.time()
            upd = tables[tag]
            for i in range(0, len(upd)):
                row = upd.iloc[i]
                debug = False
                if debug:
                    print(row)
                matching_processes = get_matching_processes(row)
                if matching_processes is not None and len(matching_processes) == 0:
                    print(f"WARNING: {tag} row matched no processes")
                    continue
                matching_commodities = get_matching_commodities(row)
                if matching_commodities is not None and len(matching_commodities) == 0:
                    print(f"WARNING: {tag} row matched no commodities")
                    continue
                df = tables[Tag.fi_t]
                if any(df.index.duplicated()):
                    raise ValueError("~FI_T table has duplicated indices")
                if tag == Tag.tfm_upd:
                    # construct query into ~FI_T to get indices of matching rows
                    df = df.reset_index()
                    if matching_processes is not None:
                        df = df.merge(matching_processes, on="TechName")
                    if debug:
                        print(f"{len(df)} rows after processes")
                        if any(df["index"].duplicated()):
                            raise ValueError("~FI_T table has duplicated indices")
                    if matching_commodities is not None:
                        df = df.merge(
                            matching_commodities.rename(columns={"CommName": "Comm-IN"})
                        )
                    if debug:
                        print(f"{len(df)} rows after commodities")
                        if any(df["index"].duplicated()):
                            raise ValueError("~FI_T table has duplicated indices")
                    attribute = row.Attribute
                    if attribute is not None:
                        df = df.query("Attribute == @attribute")
                    if debug:
                        print(f"{len(df)} rows after Attribute")
                        if any(df["index"].duplicated()):
                            raise ValueError("~FI_T table has duplicated indices")
                    region = row.Region
                    if region is not None:
                        df = df.query("Region == @region")
                    if debug:
                        print(f"{len(df)} rows after Region")
                        if any(df["index"].duplicated()):
                            raise ValueError("~FI_T table has duplicated indices")
                    df = df.set_index("index")
                    if debug:
                        if any(df.index.duplicated()):
                            raise ValueError("~FI_T table has duplicated indices")
                    if isinstance(row.VALUE, str) and row.VALUE[0] in {
                        "*",
                        "+",
                        "-",
                        "/",
                    }:
                        df = df.astype({"VALUE": float}).eval("VALUE=VALUE" + row.VALUE)
                    else:
                        df["VALUE"] = [row.VALUE] * len(df)
                    if len(df) == 0:
                        print(f"WARNING: {tag} row matched nothing")
                    tables[Tag.fi_t].update(df)
                else:
                    # Construct 1-row data frame for data
                    # Cross merge with processes and commodities (if they exist)
                    row = row.filter(df.columns)
                    row = pd.DataFrame([row])
                    if matching_processes is not None:
                        row = matching_processes.merge(row, how="cross")
                    if matching_commodities is not None:
                        matching_commodities.rename(
                            columns={"CommName": "Comm-IN"}, inplace=True
                        )
                        matching_commodities["Comm-OUT"] = matching_commodities[
                            "Comm-IN"
                        ]
                        row = matching_commodities.merge(row, how="cross")
                    tables[Tag.fi_t] = pd.concat([df, row], ignore_index=True)
        print(
            f"  process_wildcards: {tag} took {time.time()-start_time:.2f} seconds for {len(upd)} rows"
        )

    return tables


def process_time_slices(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    def timeslices_table(
        table: EmbeddedXlTable, regions: str, result: List[EmbeddedXlTable]
    ):
        # TODO will need to handle columns with multiple values
        timeslices = [(col, values[0]) for col, values in table.dataframe.items()]

        # No idea why casing of Weekly is special
        timeslices = [
            (col.upper(), val) if col != "Weekly" else (col, val)
            for col, val in timeslices
        ]

        # Accumulate values from previous entries
        ts_map = []
        for i in range(1, len(timeslices)):
            col, val = timeslices[i]
            timeslices[i] = (col, timeslices[i - 1][1] + val)
            for j in range(0, i):
                ts_map.append((timeslices[j][1], timeslices[i][1]))

        ts_maps = {
            "Region": regions,
            "Parent": [t[0] for t in ts_map],
            "TimesliceMap": [t[1] for t in ts_map],
        }
        result.append(replace(table, tag="TimeSliceMap", dataframe=DataFrame(ts_maps)))

        timeslices.insert(0, ("ANNUAL", "ANNUAL"))

        ts_groups = {
            "Region": regions,
            "TSLVL": [t[0] for t in timeslices],
            "TS_GROUP": [t[1] for t in timeslices],
        }
        result.append(
            replace(table, tag="TimeSlicesGroup", dataframe=DataFrame(ts_groups))
        )

    result = []
    regions = ",".join(single_column(tables, Tag.book_regions_map, "Region"))

    for table in tables:
        if table.tag != Tag.time_slices:
            result.append(table)
        else:
            timeslices_table(table, regions, result)

    return result


def convert_to_string(input: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    output = {}
    for key, value in input.items():
        output[key] = value.applymap(
            lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
        )
    return output


def produce_times_tables(
    input: Dict[str, DataFrame], mappings: List[TimesXlMap]
) -> Dict[str, DataFrame]:
    print(
        f"produce_times_tables: {len(input)} tables incoming,"
        f" {sum(len(value) for (_, value) in input.items())} rows"
    )
    result = {}
    used_tables = set()
    for mapping in mappings:
        if not mapping.xl_name in input:
            print(
                f"WARNING: Cannot produce table {mapping.times_name} because input table"
                f" {mapping.xl_name} does not exist"
            )
        else:
            used_tables.add(mapping.xl_name)
            df = input[mapping.xl_name].copy()
            if mapping.filter_rows:
                # Select just the rows where the attribute value matches the last mapping column or output table name
                if not "Attribute" in df.columns:
                    print(
                        f"WARNING: Cannot produce table {mapping.times_name} because input"
                        f" table {mapping.xl_name} does not contain an Attribute column"
                    )
                else:
                    colname = mapping.xl_cols[-1]
                    filter = set(x.lower() for x in {colname, mapping.times_name})
                    i = df["Attribute"].str.lower().isin(filter)
                    df = df.loc[i, :]
                    if colname not in df.columns:
                        df = df.rename(columns={"VALUE": colname})
            # TODO find the correct tech group
            if "TechGroup" in mapping.xl_cols:
                df["TechGroup"] = df["TechName"]
            if not all(c in df.columns for c in mapping.xl_cols):
                missing = set(mapping.xl_cols) - set(df.columns)
                print(
                    f"WARNING: Cannot produce table {mapping.times_name} because input"
                    f" table {mapping.xl_name} does not contain the required columns"
                    f" - {', '.join(missing)}"
                )
            else:
                cols_to_drop = [x for x in df.columns if not x in mapping.xl_cols]
                df.drop(columns=cols_to_drop, inplace=True)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.rename(columns=mapping.col_map, inplace=True)
                i = df[mapping.times_cols[-1]].notna()
                df = df.loc[i, mapping.times_cols]
                result[mapping.times_name] = df

    unused_tables = set(input.keys()) - used_tables
    if len(unused_tables) > 0:
        print(
            f"WARNING: {len(unused_tables)} unused tables: {', '.join(sorted(unused_tables))}"
        )

    return result


def dump_tables(tables: List, filename: str) -> List:
    os.makedirs("output", exist_ok=True)
    with open(rf"output/{filename}", "w") as text_file:
        for t in tables if isinstance(tables, List) else tables.items():
            if isinstance(t, EmbeddedXlTable):
                tag = t.tag
                text_file.write(f"sheetname: {t.sheetname}\n")
                text_file.write(f"range: {t.range}\n")
                text_file.write(f"filename: {t.filename}\n")
                if t.uc_sets:
                    text_file.write(f"uc_sets: {t.uc_sets}\n")
                df = t.dataframe
            else:
                tag = t[0]
                df = t[1]
            text_file.write(f"tag: {tag}\n")
            types = ", ".join([f"{i} ({v})" for i, v in df.dtypes.items()])
            text_file.write(f"types: {types}\n")
            text_file.write(df.to_csv(index=False, lineterminator="\n"))
            text_file.write("\n" * 2)

    return tables


def convert_xl_to_times(
    dir: str, input_files: List[str], mappings: List[TimesXlMap]
) -> Dict[str, DataFrame]:
    pickle_file = "raw_tables.pkl"
    if os.path.isfile(pickle_file):
        raw_tables = pickle.load(open(pickle_file, "rb"))
        print(f"WARNING: Using pickled data not xlsx")
    else:
        raw_tables = []
        filenames = [os.path.join(dir, filename) for filename in input_files]

        use_pool = True
        if use_pool:
            with ProcessPoolExecutor() as executor:
                for result in executor.map(extract_tables, filenames):
                    raw_tables.extend(result)
        else:
            for f in filenames:
                result = extract_tables(f)
                raw_tables.extend(result)
        pickle.dump(raw_tables, open(pickle_file, "wb"))

    print(
        f"Extracted {len(raw_tables)} tables,"
        f" {sum(table.dataframe.shape[0] for table in raw_tables)} rows"
    )

    transforms = [
        lambda tables: dump_tables(tables, "raw_tables.txt"),
        normalize_tags_columns_attrs,
        remove_fill_tables,
        lambda tables: [remove_comment_rows(t) for t in tables],
        lambda tables: [remove_comment_cols(t) for t in tables],
        remove_tables_with_formulas,  # slow
        process_transform_insert,
        process_flexible_import_tables,  # slow
        process_user_constraint_tables,
        process_commodity_emissions,
        process_commodities,
        process_processes,
        process_transform_availability,
        fill_in_missing_values,
        process_time_slices,
        expand_rows_parallel,  # slow
        remove_invalid_values,
        process_time_periods,
        process_currencies,
        apply_fixups,
        extract_commodity_groups,
        merge_tables,
        process_years,
        process_wildcards,
        convert_to_string,
        lambda tables: dump_tables(tables, "merged_tables.txt"),
        lambda tables: produce_times_tables(tables, mappings),
    ]

    results = []
    input = raw_tables
    for transform in transforms:
        start_time = time.time()
        output = transform(input)
        end_time = time.time()
        print(
            f"transform {transform.__code__.co_name} took {end_time-start_time:.2f} seconds"
        )
        results.append(output)
        input = output

    print(
        f"Conversion complete, {len(output)} tables produced,"
        f" {sum(df.shape[0] for tablename, df in output.items())} rows"
    )

    return output


def expand_rows_parallel(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(expand_rows, tables))


def write_csv_tables(tables: Dict[str, DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".csv"):
            os.remove(os.path.join(output_dir, item))
    for tablename, df in tables.items():
        df.to_csv(os.path.join(output_dir, tablename + "_output.csv"), index=False)


def read_csv_tables(input_dir: str) -> Dict[str, DataFrame]:
    result = {}
    for filename in os.listdir(input_dir):
        result[filename.split(".")[0]] = pd.read_csv(
            os.path.join(input_dir, filename), dtype=str
        )
    return result


def compare(data: Dict[str, DataFrame], ground_truth: Dict[str, DataFrame]):
    print(
        f"Ground truth contains {len(ground_truth)} tables,"
        f" {sum(df.shape[0] for tablename, df in ground_truth.items())} rows"
    )

    missing = set(ground_truth.keys()) - set(data.keys())
    missing_str = ", ".join(
        [f"{x} ({ground_truth[x].shape[0]})" for x in sorted(missing)]
    )
    if len(missing) > 0:
        print(f"WARNING: Missing {len(missing)} tables: {missing_str}")

    total_gt_rows = 0
    total_correct_rows = 0
    for table_name, gt_table in sorted(
        ground_truth.items(), reverse=True, key=lambda t: len(t[1].values)
    ):
        total_gt_rows += len(gt_table.values)
        if table_name in data:
            data_table = data[table_name]

            # Remove .integer suffix added to duplicate column names by CSV reader (mangle_dupe_cols=False not supported)
            transformed_gt_cols = [col.split(".")[0] for col in gt_table.columns]

            if transformed_gt_cols != list(data[table_name].columns):
                print(
                    f"WARNING: Table {table_name} header incorrect, was"
                    f" {data_table.columns.values}, should be {transformed_gt_cols}"
                )
            else:
                # both are in string form so can be compared without any issues
                gt_rows = set(tuple(row) for row in gt_table.values.tolist())
                data_rows = set(tuple(row) for row in data_table.values.tolist())
                total_correct_rows += len(gt_rows.intersection(data_rows))
                additional = data_rows - gt_rows
                missing = gt_rows - data_rows
                if len(additional) != 0 or len(missing) != 0:
                    print(
                        f"WARNING: Table {table_name} ({data_table.shape[0]} rows,"
                        f" {gt_table.shape[0]} GT rows) contains {len(additional)}"
                        f" additional rows and is missing {len(missing)} rows"
                    )
                    DataFrame(additional).to_csv(
                        os.path.join("output", table_name + "_additional.csv"),
                        index=False,
                    )
                    DataFrame(missing).to_csv(
                        os.path.join("output", table_name + "_missing.csv"), index=False
                    )

    print(
        f"{total_correct_rows / total_gt_rows :.1%} of ground truth rows present"
        f" in output ({total_correct_rows}/{total_gt_rows})"
    )


def round_sig(x, sig_figs):
    if x == 0.0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + sig_figs - 1)


if __name__ == "__main__":
    mappings = read_mappings("times_mapping.txt")

    uk = False
    if uk:
        xl_files_dir = "input"
        input_files = [
            "SysSettings.xlsx",
            "VT_UK_RES.xlsx",
            "VT_UK_ELC.xlsx",
            "VT_UK_IND.xlsx",
            "VT_UK_AGR.xlsx",
        ]
    else:
        # Make sure you set A3 in SysSettings.xlsx#Regions to Single-region if comparing with times-ireland-model_gams
        xl_files_dir = os.path.join("..", "times-ireland-model")
        input_files = [
            str(path)
            for path in Path(xl_files_dir).rglob("*.xlsx")
            if not path.name.startswith("~")
        ]

    tables = convert_xl_to_times(xl_files_dir, input_files, mappings)

    write_csv_tables(tables, "output")

    ground_truth = read_csv_tables("ground_truth")
    compare(tables, ground_truth)
