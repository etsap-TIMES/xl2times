from openpyxl import load_workbook
from openpyxl.worksheet.cell_range import CellRange
from pandas.core.frame import DataFrame
import pandas as pd
from dataclasses import dataclass, replace
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
                    match = re.match("~UC_SETS:(.*)", value, re.IGNORECASE)
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
    return value is None or (isinstance(value, numpy.float64) and numpy.isnan(value))


def remove_comment_rows(table: EmbeddedXlTable) -> EmbeddedXlTable:
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
            result[key] = pd.concat([table.dataframe for table in group])
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
    df = df.assign(VALUE=data)
    nrows = df.shape[0]
    df = df.explode(value_column)

    names = pd.Series(data_columns * nrows, index=df.index, dtype=str)
    # Remove rows with no VALUE
    filter = df[value_column].notna()
    df = df[filter]
    names = names[filter]
    return df, names


def timeslices(tables: List[EmbeddedXlTable]):
    # TODO merge with other timeslice code

    # No idea why casing of Weekly is special
    cols = single_table(tables, "~TimeSlices").dataframe.columns
    timeslices = [col if col == "Weekly" else col.upper() for col in cols]
    timeslices.insert(0, "ANNUAL")
    return timeslices


def process_flexible_import_tables(
    tables: List[EmbeddedXlTable],
) -> List[EmbeddedXlTable]:
    legal_values = {
        "LimType": {"LO", "UP", "FX"},
        "TimeSlice": timeslices(tables),
        "Comm-OUT": set(merge_columns(tables, "~FI_Comm", "CommName")),
        "Region": single_column(tables, "~BookRegions_Map", "Region"),
        "Curr": single_column(tables, "~Currencies", "Currency"),
        "Other_Indexes": {"Input", "Output"},
    }

    def get_colname(value):
        if value.isdigit():
            return "Year", int(value)
        for name, values in legal_values.items():
            if value in values:
                return name, value
        return None, value

    def process_flexible_import_table(table: EmbeddedXlTable) -> EmbeddedXlTable:
        # See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16

        if not table.tag.startswith("~FI_T"):
            return table
        df = table.dataframe
        mapping = {"YEAR": "Year", "CommName": "Comm-IN"}
        df = df.rename(columns=mapping)

        nrows = df.shape[0]
        if ("Comm-IN" in df.columns) and ("Comm-OUT" in df.columns):
            kwargs = {"TOP-IN": ["IN"] * nrows, "TOP-OUT": ["OUT"] * nrows}
            df = df.assign(**kwargs)

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
        df, attribute_suffix = explode(df, data_columns)

        # Append the data column name to the Attribute column
        attribute = "Attribute"
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
        df.reset_index(drop=True, inplace=True)

        # Should have all index_columns and VALUE
        if len(df.columns) != (len(index_columns) + 1):
            raise ValueError(f"len(df.columns) = {len(df.columns)}")

        # Note the logic in produce_times_tables that allows mappings to filter by the attribute column

        return replace(table, dataframe=df)

    return [process_flexible_import_table(t) for t in tables]


def merge_columns(tables: List[EmbeddedXlTable], tag: str, colname: str):
    columns = [table.dataframe[colname].values for table in tables if table.tag == tag]
    return numpy.concatenate(columns)


def single_table(tables: List[EmbeddedXlTable], tag: str):
    return one(table for table in tables if table.tag == tag)


def single_column(tables: List[EmbeddedXlTable], tag: str, colname: str):
    return single_table(tables, tag).dataframe[colname].values


def fill_in_missing_values(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    regions = single_column(tables, "~BookRegions_Map", "Region")
    start_year = one(single_column(tables, "~StartYear", "VALUE"))
    # TODO there are multiple currencies
    currency = single_column(tables, "~Currencies", "Currency")[0]

    for table in tables:
        df = table.dataframe.copy()
        for colname in df.columns:
            # TODO make this more declarative
            if colname == "Csets" or colname == "TechName":
                missing_value_inherit(df, colname)
            elif colname == "LimType" and table.tag == "~FI_Comm" and False:
                isna = df[colname].isna()
                ismat = df["Csets"] == "MAT"
                df.loc[isna & ismat, colname] = "FX"
                df.loc[isna & ~ismat, colname] = "LO"
            elif colname == "LimType" and table.tag == "~FI_T":
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
            return s.split(",")
        else:
            return s

    df = table.dataframe.copy()
    c = df.applymap(has_comma)
    columns_with_commas = [colname for colname in c.columns.values if c[colname].any()]
    if len(columns_with_commas) > 0:
        # Transform comma-separated strings into lists
        df = df.applymap(split_by_commas)
        for colname in columns_with_commas:
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html#pandas.DataFrame.explode
            df = df.explode(colname)
    return replace(table, dataframe=df)


def remove_invalid_values(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    # TODO pull this out
    regions = single_column(tables, "~BookRegions_Map", "Region")
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
            col_map = {}
            for index, value in enumerate(xl_cols):
                col_map[value] = times_cols[index]
            entry = TimesXlMap(
                times_name=times_name,
                times_cols=times_cols,
                xl_name=xl_name,
                xl_cols=xl_cols,
                col_map=col_map,
            )

            # TODO remove: Filter out mappings that are not yet finished
            if entry.xl_name != "~TODO" and not any(
                c.startswith("TODO") for c in entry.xl_cols
            ):
                mappings.append(entry)
            else:
                dropped.append(line)

    if len(dropped) > 0:
        print(f"WARNING: Dropping {len(dropped)} mappings that are not yet complete")
    return mappings


def process_time_periods(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    start_year = get_scalar("~StartYear", tables)
    active_pdef = get_scalar("~ActivePDef", tables)

    def process_time_periods_table(table: EmbeddedXlTable):
        if table.tag != "~TimePeriods":
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
        df["Curr"] = df["Curr"].apply(lambda x: x.replace("MEUR20", "EUR"))

        return replace(table, dataframe=df)

    return [process_currencies_table(table) for table in tables]


def apply_fixups(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    def apply_fixups_table(table: EmbeddedXlTable):
        if not table.tag.startswith("~FI_T"):
            return table

        df = table.dataframe.copy()

        i = df["Attribute"].str.upper() == "FLO_SHAR"
        df.loc[i, "Comm-IN"] = df["Comm-OUT"]

        # Append _NRGI (energy input) to some cells in FLO_SHAR
        i = (df["Attribute"].str.lower() == "share-i") & (
            (df["LimType"] == "UP") | (df["LimType"] == "LO")
        )
        df.loc[i, "Other_Indexes"] = df["TechName"].astype(str) + "_NRGI"

        # TODO allow multiple columns in mapping
        df["Attribute"] = df["Attribute"].str.replace(
            "Share-I", "FLO_SHAR", case=False, regex=False
        )

        return replace(table, dataframe=df)

    return [apply_fixups_table(table) for table in tables]


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
        if table.tag != "~TFM_FILL" and not table.tag.startswith("~TFM_Fill-R"):
            result.append(table)
    return result


def process_commodity_emissions(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    regions = single_column(tables, "~BookRegions_Map", "Region")

    result = []
    for table in tables:
        if table.tag != "~COMEMI":
            result.append(table)
        else:
            df = table.dataframe.copy()
            index_columns = ["Region", "Year", "CommName"]
            data_columns = [
                colname for colname in df.columns.values if colname not in index_columns
            ]
            df, names = explode(df, data_columns)
            df.rename(columns={"VALUE": "EMCB"}, inplace=True)
            df = df.assign(Other_Indexes=names)

            if "Region" in df.columns.values:
                df = df.astype({"Region": "string"})
                df["Region"] = df["Region"].map(lambda s: s.split(","))
                df = df.explode("Region")
                df = df[df["Region"].isin(regions)]

            nrows = df.shape[0]
            for colname in index_columns:
                if colname not in df.columns:
                    df[colname] = [None] * nrows

            result.append(replace(table, dataframe=df))

    return result


def process_commodities(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    regions = ",".join(single_column(tables, "~BookRegions_Map", "Region"))

    result = []
    for table in tables:
        if table.tag != "~FI_Comm" and table.tag != "~FI_COMM":
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
            result.append(replace(table, dataframe=df, tag="~FI_Comm"))

    return result


def process_processes(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    for table in tables:
        if table.tag != "~FI_Process":
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
    regions = single_column(tables, "~BookRegions_Map", "Region")

    tech_commodity_pairs = set()
    for table in tables:
        if (
            table.tag.startswith("~FI_T")
            and "TechName" in table.dataframe.columns
            and "Comm-IN" in table.dataframe.columns
        ):
            pairs = (
                table.dataframe[["TechName", "Comm-IN"]]
                .apply(tuple, axis=1)
                .values.tolist()
            )
            tech_commodity_pairs = tech_commodity_pairs.union(pairs)

    result = []
    dropped = []
    for table in tables:
        if (
            not table.tag.startswith("~TFM_INS")
            and not table.tag.startswith("~TFM_DINS")
            and not table.tag.startswith("~TFM_TOPINS")
            and not table.tag.startswith("~TFM_UPD")
            and not table.tag.startswith("~TFM_COMGRP")
        ):
            result.append(table)

        # TODO ~TFM_INS-TS: Regions should be specified in a column with header=Region and columns in data area are YEARS
        elif (
            table.tag == "~TFM_INS"
            or table.tag == "~TFM_INS-TS"
            or table.tag == "~TFM_UPD"
            or table.tag == "~TFM_COMGRP"
        ):
            put_into_table = table.tag if table.tag == "~TFM_COMGRP" else "~FI_T"

            df = table.dataframe.copy()
            nrows = df.shape[0]

            if "TimeSlice" not in table.dataframe.columns.values:
                df.insert(0, "TimeSlice", [None] * nrows)
            if "LimType" not in table.dataframe.columns.values:
                df.insert(1, "LimType", [None] * nrows)

            if "AllRegions" in table.dataframe.columns:
                for region in regions:
                    df[region] = df["AllRegions"]
                df.drop(columns=["AllRegions"], inplace=True)

            if table.tag == "~TFM_INS-TS":
                # TODO what to do if there is already a region column?
                df = df.assign(Region=[regions] * nrows)
                df = df.explode(["Region"])
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
                df = df.assign(Region=[region_cols] * nrows)
                df = df.assign(VALUE=data)
                df = df.explode(["Region", "VALUE"])

            if "Cset_CN" in df.columns:
                df["Comm-OUT"] = df["Cset_CN"]
                df.drop(columns=["Cset_CN"], inplace=True)

            # TODO what to do about Other_indexes?

            if "PSET_PN" in df.columns:
                df.rename(columns={"PSET_PN": "Pset_PN"}, inplace=True)

            if "Pset_PN" in df.columns and "Pset_CI" in df.columns:
                df["TechName"] = [None] * nrows
                df["Comm-IN"] = [None] * nrows
                for index, row in df.iterrows():
                    tech_name_wildcard = row["Pset_PN"]
                    # TODO: these can be comma separated
                    comm = row["Pset_CI"]
                    regexp = (
                        None
                        if tech_name_wildcard == None
                        else re.compile(tech_name_wildcard.replace("*", ".*"))
                    )
                    matched_tech_commodity = [
                        t
                        for t in tech_commodity_pairs
                        if (
                            t[0] == None
                            if regexp == None
                            else (t[0] != None and regexp.match(t[0]))
                        )
                        and comm == t[1]
                    ]
                    df.at[index, "TechName"] = [t[0] for t in matched_tech_commodity]
                    df.at[index, "Comm-IN"] = [t[1] for t in matched_tech_commodity]
                df = df.explode(["TechName", "Comm-IN"])
                df.drop(columns=["Pset_PN", "Pset_CI"], inplace=True)
                result.append(replace(table, dataframe=df, tag=put_into_table))

            elif "Pset_PN" in df.columns:
                df["TechName"] = [None] * nrows
                for index, row in df.iterrows():
                    tech_name_wildcard = row["Pset_PN"]
                    regexp = (
                        None
                        if tech_name_wildcard == None
                        else re.compile(tech_name_wildcard.replace("*", ".*"))
                    )
                    matched_tech = [
                        t[0]
                        for t in tech_commodity_pairs
                        if (
                            t[0] == None
                            if regexp == None
                            else (t[0] != None and regexp.match(t[0]))
                        )
                    ]
                    df.at[index, "TechName"] = list(set(matched_tech))
                df = df.explode(["TechName"])
                df.drop(columns=["Pset_PN"], inplace=True)
                result.append(replace(table, dataframe=df, tag=put_into_table))

            elif "Pset_CI" in df.columns:
                df["Comm-IN"] = [None] * nrows
                for index, row in df.iterrows():
                    # TODO: these can be comma separated
                    comm = row["Pset_CI"]
                    matched_commodity = [
                        t[1] for t in tech_commodity_pairs if comm == t[1]
                    ]
                    df.at[index, "Comm-IN"] = list(set(matched_commodity))
                df = df.explode(["Comm-IN"])
                df.drop(columns=["Pset_CI"], inplace=True)
                result.append(replace(table, dataframe=df, tag=put_into_table))

            else:
                # No filters
                result.append(replace(table, dataframe=df, tag=put_into_table))

        elif table.tag == "~TFM_DINS":
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
            df = df.assign(Region=[region_cols] * nrows)
            df = df.assign(DEMAND=data)
            df = df.explode(["Region", "DEMAND"])

            df.rename(columns={"Cset_CN": "Comm-IN"}, inplace=True)

            result.append(replace(table, dataframe=df, tag="~FI_T"))

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
        if table.tag != "~TFM_AVA":
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


def process_transform_update(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    dropped = []
    for table in tables:
        if table.tag != "~TFM_UPD":
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
                f"WARNING: Dropped {len(group)} transform update tables ({key})"
                f" rather than processing them"
            )

    return result


def process_user_constraints(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    dropped = []
    for table in tables:
        if not table.tag.startswith("~UC_"):
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
                f"WARNING: Dropped {len(group)} user constraint tables ({key})"
                f" rather than processing them"
            )

    return result


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
    regions = ",".join(single_column(tables, "~BookRegions_Map", "Region"))

    for table in tables:
        if table.tag != "~TimeSlices":
            result.append(table)
        else:
            timeslices_table(table, regions, result)

    return result


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
            if "Attribute" in df.columns:
                # Select just the rows where the attribute value matches the last mapping column or output table name
                colname = mapping.xl_cols[-1]
                filter = set(x.lower() for x in {colname, mapping.times_name})
                i = df["Attribute"].str.lower().isin(filter)
                df = df.loc[i, :]
                if colname not in df.columns:
                    df = df.rename(columns={"VALUE": colname})
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
                df = t.dataframe
            else:
                tag = t[0]
                df = t[1]
            text_file.write(f"tag: {tag}\n")
            types = ", ".join([f"{i} ({v})" for i, v in df.dtypes.items()])
            text_file.write(f"types: {types}\n")
            text_file.write(df.to_csv(index=False, line_terminator="\n"))
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
        remove_fill_tables,
        lambda tables: [remove_comment_rows(t) for t in tables],
        lambda tables: [remove_comment_cols(t) for t in tables],
        remove_tables_with_formulas,  # slow
        process_transform_insert,
        process_flexible_import_tables,  # slow
        process_commodity_emissions,
        process_commodities,
        process_processes,
        process_transform_availability,
        process_transform_update,
        process_user_constraints,
        fill_in_missing_values,
        process_time_slices,
        lambda tables: [expand_rows(t) for t in tables],  # slow
        remove_invalid_values,
        process_time_periods,
        process_currencies,
        apply_fixups,
        merge_tables,
        lambda tables: dump_tables(tables, "merged_tables.txt"),
        lambda tables: produce_times_tables(tables, mappings),
    ]

    results = []
    input = raw_tables
    for transform in transforms:
        start_time = time.time()
        output = transform(input)
        end_time = time.time()
        print(f"transform took {end_time-start_time:.2f} seconds")
        results.append(output)
        input = output

    print(
        f"Conversion complete, {len(output)} tables produced,"
        f" {sum(df.shape[0] for tablename, df in output.items())} rows"
    )

    return output


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
        result[filename.split(".")[0]] = pd.read_csv(os.path.join(input_dir, filename))
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
                gt_rows = set(tuple(i) for i in gt_table.values.tolist())
                data_rows = set(tuple(i) for i in data_table.values.tolist())
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
