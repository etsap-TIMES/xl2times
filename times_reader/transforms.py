from pandas.core.frame import DataFrame
import pandas as pd
from dataclasses import replace
from typing import Dict, List
from more_itertools import locate, one
from itertools import groupby
import os
from concurrent.futures import ProcessPoolExecutor
import time
from functools import reduce
from . import datatypes
from . import utils


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


def remove_comment_rows(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
    """
    Return a modified copy of 'table' where rows with cells containig '*'
    or '\I:' in their first or third columns have been deleted. These characters
    are defined in https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
    as comment identifiers (pag 15).
    TODO: we believe the deletion of the third column is a bug. We tried deleting that part
    of the code but we failed to parse a row as a consequence. We need to investigate why,
    fix that parsing and remove the deletion of the third column.

    :param table:       Table object in EmbeddedXlTable format.
    :return:            Table object in EmbeddedXlTable format without comment rows.
    """
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

    # TODO: the deletion of this third column is a bug. Removing it causes the
    # program to fail parse all rows. We need to fix the parsing so it can read
    # all rows and remove this code block.
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


def remove_comment_cols(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
    """
    Return a modified copy of 'table' where columns with labels starting with '*'
    have been deleted.

    :param table:       Table object in EmbeddedXlTable format.
    :return:            Table object in EmbeddedXlTable format without comment columns.
    """
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


def remove_tables_with_formulas(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Return a modified copy of 'tables' where tables with formulas (as identified by an
    initial '=') have deleted from the list.

    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format without any formulas.
    """

    def is_formula(s):
        return isinstance(s, str) and len(s) > 0 and s[0] == "="

    def has_formulas(table):
        has = table.dataframe.applymap(is_formula).any(axis=None)
        if has:
            print(f"WARNING: Excluding table {table.tag} because it has formulas")
        return has

    return [table for table in tables if not has_formulas(table)]


def normalize_tags_columns_attrs(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Normalize (uppercase) tags, column names, and values in attribute columns.


    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with normalzed values.
    """
    # TODO Uppercase column names and attribute values in mapping.txt when reading it
    # TODO Check all string literals left in file
    def normalize(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
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


def merge_tables(tables: List[datatypes.EmbeddedXlTable]) -> Dict[str, DataFrame]:
    """
    Merge all tables in 'tables' with the same table tag, as long as they share the same
    column field values. Print a warning for those that don't share the same column values.
    Return a dictionary linking each table tag with its merged table.

    :param tables:      List of tables in datatypes.EmbeddedXlTable format.
    :return:            Dictionary associating a given table tag with its merged table.
    """
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
                df["Year"] = pd.to_numeric(df["Year"]).astype("Int64")
            result[key] = df
    return result


def process_flexible_import_tables(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Attempt to process all flexible import tables in 'tables'. The processing includes:
    - Checking that the table is indeed a flexible import table. If not, return it unmodified.
    - Removing, adding and renaming columns as needed.
    - Populating index columns.
    - Handing Attribute column and Other Indexes.
    See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16.


    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with all FI_T processed.
    """
    # Get a list of allowed values for each category.
    legal_values = {
        "LimType": {"LO", "UP", "FX"},
        "TimeSlice": utils.timeslices(tables),
        "Comm-OUT": set(utils.merge_columns(tables, datatypes.Tag.fi_comm, "CommName")),
        "Region": utils.single_column(tables, datatypes.Tag.book_regions_map, "Region"),
        "Curr": utils.single_column(tables, datatypes.Tag.currencies, "Currency"),
        "Other_Indexes": {"Input", "Output"},
    }

    def get_colname(value):
        # Return the value in the desired format along with the associated category (if any)
        # TODO make sure to do case-insensitive comparisons when parsing composite column names
        if value.isdigit():
            return "Year", int(value)
        for name, values in legal_values.items():
            if value in values:
                return name, value
        return None, value

    def process_flexible_import_table(
        table: datatypes.EmbeddedXlTable,
    ) -> datatypes.EmbeddedXlTable:
        # Make sure it's a flexible import table, and return the table untouched if not
        if not table.tag.startswith(datatypes.Tag.fi_t) and table.tag not in {
            datatypes.Tag.tfm_upd,
        }:
            return table

        # Rename, add and remove specific columns if the circumstances are right
        df = table.dataframe
        mapping = {"YEAR": "Year", "CommName": "Comm-IN"}
        df = df.rename(columns=mapping)

        if "CURR" in df.columns.values:
            df.rename(columns={"CURR": "Curr"}, inplace=True)

        nrows = df.shape[0]
        if (
            ("Comm-IN" in df.columns)
            and ("Comm-OUT" in df.columns)
            and (table.tag != datatypes.Tag.tfm_upd)
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

        # datatypes.Tag column no longer used to identify data columns
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

        table = utils.apply_composite_tag(table)
        df = table.dataframe

        attribute = "Attribute"
        if table.tag != datatypes.Tag.tfm_upd:
            df, attribute_suffix = utils.explode(df, data_columns)

            # Append the data column name to the Attribute column values
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
        if table.tag == datatypes.Tag.fi_t and len(df.columns) != (
            len(index_columns) + 1
        ):
            raise ValueError(f"len(df.columns) = {len(df.columns)}")

        # TODO: should we have a global list of column name -> type?
        df["Year"] = df["Year"].astype("Int64")

        return replace(table, dataframe=df)

    return [process_flexible_import_table(t) for t in tables]


def process_user_constraint_tables(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Attempt to process all tables in 'tables' as user constraint tables. The processing includes:
    - Checking that the table is indeed a user constraint table. If not, return it unmodified.
    - Removing, adding and renaming columns as needed.
    - Populating index columns.
    - Handing Attribute column and wildcards.
    See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16.


    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with all FI_T processed.
    """
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
        "Region": utils.single_column(tables, datatypes.Tag.book_regions_map, "Region"),
        "LimType": {"LO", "UP"},
        "Side": {"LHS", "RHS"},
    }

    commodity_names = set(
        utils.merge_columns(tables, datatypes.Tag.fi_comm, "CommName")
    )
    process_names = set(
        utils.merge_columns(tables, datatypes.Tag.fi_process, "TechName")
    )

    def get_colname(value):
        # TODO make sure to do case-insensitive comparisons when parsing composite column names
        if value.isdigit():
            return "Year", int(value)
        for name, values in legal_values.items():
            if value in values:
                return name, value
        return None, value

    def process_user_constraint_table(
        table: datatypes.EmbeddedXlTable,
    ) -> datatypes.EmbeddedXlTable:
        # See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16

        if not table.tag.startswith(datatypes.Tag.uc_t):
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

        table = utils.apply_composite_tag(table)
        df = table.dataframe
        df, attribute_suffix = utils.explode(df, data_columns)

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

        utils.apply_wildcards(df, commodity_names, "Cset_CN", "CommName")
        df.drop("Cset_CN", axis=1, inplace=True)
        df = df.explode(["CommName"], ignore_index=True)

        utils.apply_wildcards(df, process_names, "Pset_PN", "TechName")
        df.drop("Pset_PN", axis=1, inplace=True)
        df = df.explode(["TechName"], ignore_index=True)

        # TODO: handle other wildcard columns

        # TODO: should we have a global list of column name -> type?
        df["Year"] = df["Year"].astype("Int64")

        return replace(table, dataframe=df)

    return [process_user_constraint_table(t) for t in tables]


def fill_in_missing_values(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Attempt to fill in missing values for all tables except update tables (as these contain
    wildcards). How the value is filled in depends on the name of the column the empty values
    belong to.

    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with empty values filled in.
    """
    result = []
    regions = utils.single_column(tables, datatypes.Tag.book_regions_map, "Region")
    start_year = one(utils.single_column(tables, datatypes.Tag.start_year, "VALUE"))
    # TODO there are multiple currencies
    currency = utils.single_column(tables, datatypes.Tag.currencies, "Currency")[0]

    def fill_in_missing_values_inplace(df):
        for colname in df.columns:
            # TODO make this more declarative
            if colname == "Csets" or colname == "TechName":
                utils.missing_value_inherit(df, colname)
            elif colname == "LimType" and table.tag == datatypes.Tag.fi_comm and False:
                isna = df[colname].isna()
                ismat = df["Csets"] == "MAT"
                df.loc[isna & ismat, colname] = "FX"
                df.loc[isna & ~ismat, colname] = "LO"
            elif colname == "LimType" and (
                table.tag == datatypes.Tag.fi_t or table.tag.startswith("~TFM")
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
        if table.tag == datatypes.Tag.tfm_upd:
            # Missing values in update tables are wildcards and should not be filled in
            result.append(table)
        else:
            df = table.dataframe.copy()
            fill_in_missing_values_inplace(df)
            result.append(replace(table, dataframe=df))
    return result


def expand_rows(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
    """
    Expand entries with commas into separate entries in the same column. Do this
    for all tables except transformation update tables.

    :param table:       Table in EmbeddedXlTable format.
    :return:            Table in EmbeddedXlTable format with expanded comma entries.
    """

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


def remove_invalid_values(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Remove all entries of any dataframes that are considered invalid. The rules for
    allowing an entry can be seen in the 'constraints' dictionary below.

    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with disallowed entries removed.
    """
    # TODO pull this out
    regions = utils.single_column(tables, datatypes.Tag.book_regions_map, "Region")
    # TODO pull this out
    # Rules for allowing entries. Each entry of the dictionary designates a rule for a
    # a given column, and the values that are allowed for that column.
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


def process_time_periods(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    start_year = utils.get_scalar(datatypes.Tag.start_year, tables)
    active_pdef = utils.get_scalar(datatypes.Tag.active_p_def, tables)

    def process_time_periods_table(table: datatypes.EmbeddedXlTable):
        if table.tag != datatypes.Tag.time_periods:
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


def process_currencies(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    def process_currencies_table(table: datatypes.EmbeddedXlTable):
        if "Curr" not in table.dataframe.columns:
            return table

        df = table.dataframe.copy()

        # TODO: work out how to implement this correctly, EUR18 etc. do not appear in raw tables
        df["Curr"] = df["Curr"].apply(
            lambda x: None if x is None else x.replace("MEUR20", "EUR")
        )

        return replace(table, dataframe=df)

    return [process_currencies_table(table) for table in tables]


def apply_fixups(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    def apply_fixups_table(table: datatypes.EmbeddedXlTable):
        if not table.tag.startswith(datatypes.Tag.fi_t) or table.dataframe.size == 0:
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


def extract_commodity_groups(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    fit_tables = [t for t in tables if t.tag == datatypes.Tag.fi_t]
    process_tables = [t for t in tables if t.tag == datatypes.Tag.fi_process]
    commodity_tables = [t for t in tables if t.tag == datatypes.Tag.fi_comm]

    demand_processes = set()
    for process_table in process_tables:
        df = process_table.dataframe
        demand_processes = demand_processes.union(
            set(df[df["Sets"] == "DMD"]["TechName"])
        )

    demand_commodities = set()
    environment_commodities = set()
    for commodity_table in commodity_tables:
        df = commodity_table.dataframe
        demand_commodities = demand_commodities.union(
            set(df[df["Csets"] == "DEM"]["CommName"])
        )
        environment_commodities = environment_commodities.union(
            set(df[df["Csets"] == "ENV"]["CommName"])
        )

    gmap_tables = []
    for fit_table in fit_tables:
        # Energy input
        inputs = fit_table.dataframe[["Region", "TechName", "Comm-IN"]].copy()
        inputs = inputs[~inputs["TechName"].isnull()]
        inputs = inputs[~inputs["Comm-IN"].isnull()]
        # TODO find where ACT is coming from and fix
        inputs = inputs[inputs["Comm-IN"] != "ACT"]
        inputs["TechName"] = inputs["TechName"].astype(str) + "_NRGI"
        inputs.rename(columns={"Comm-IN": "Comm"}, inplace=True)

        # Energy output
        outputs = fit_table.dataframe[["Region", "TechName", "Comm-OUT"]].copy()
        outputs = outputs[~outputs["TechName"].isnull()]
        outputs = outputs[~outputs["Comm-OUT"].isnull()]
        outputs = outputs[~outputs["TechName"].isin(demand_processes)]
        outputs = outputs[~outputs["Comm-OUT"].isin(demand_commodities)]
        outputs = outputs[~outputs["Comm-OUT"].isin(environment_commodities)]
        # TODO this removes two rows where tech is in CHP set
        outputs = outputs[outputs["Comm-OUT"] != "ELCC"]
        outputs = outputs[~outputs["Comm-OUT"].str.startswith("BIO")]
        outputs = outputs[~outputs["Comm-OUT"].str.startswith("PWR")]
        outputs["TechName"] = outputs["TechName"].astype(str) + "_NRGO"
        outputs.rename(columns={"Comm-OUT": "Comm"}, inplace=True)

        # Demand output
        demo = fit_table.dataframe[["Region", "TechName", "Comm-OUT"]].copy()
        demo = demo[~demo["Comm-OUT"].isnull()]
        demo = demo[demo["TechName"].isin(demand_processes)]
        demo["TechName"] = demo["TechName"].astype(str) + "_DEMO"
        demo.rename(columns={"Comm-OUT": "Comm"}, inplace=True)

        gmap_tables += [inputs, outputs, demo]

        # Additional:
        #   NRGO: 411
        #     R-SH_Det_BDL_X0_NRGO,RSDSH_Det, process PRE, com NRG
        #       RSDSH_Det, 21 in gt for NRGO, R-SW_Det_GAS_N2, process PRE, com NRG
        #   NRGI: 619
        #     SH2LDEL_01, not in gt, PRE set, in SUPH2LC, out SUPH2LD, there is one _NRGI,SUPH2LC in gt FT-INDH2L
        #   DEMO: 40
        # Missing: 7
        #   IE,S-DCE-CS_NRGI,SRVELC-DC-C
        #   IE,IMPNRGZ_NRGO,SRVELC-DC-C
        #   IE,FT-RSDAHT_NRGO,RSDAHT2
        #   IE,IMPNRGZ_NRGO,RSDAHT2
        #   IE,FT-SRVELC_NRGO,SRVELC-DC-C
        #   IE,P-TH-OCGT-GAS00-SK4_NRGO,ELCC (CHP)
        #   IE,P-TH-OCGT-GAS00-SK3_NRGO,ELCC (CHP)

    merged = pd.concat(gmap_tables, ignore_index=True, sort=False)

    # TODO apply renamings from ~TFM_TOPINS e.g. RSDAHT to RSDAHT2

    tables.append(
        datatypes.EmbeddedXlTable(
            sheetname="",
            range="",
            filename="",
            uc_sets="",
            tag="COM_GMAP",
            dataframe=merged,
        )
    )
    return tables


def remove_fill_tables(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    # These tables collect data from elsewhere and update the table itself or a region below
    # The collected data is then presumably consumed via Excel references or vlookups
    # TODO for the moment, assume VEDA has updated these tables but we will need a tool to do this
    result = []
    for table in tables:
        if table.tag != datatypes.Tag.tfm_fill and not table.tag.startswith(
            datatypes.Tag.tfm_fill_r
        ):
            result.append(table)
    return result


def process_commodity_emissions(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    regions = utils.single_column(tables, datatypes.Tag.book_regions_map, "Region")

    result = []
    for table in tables:
        if table.tag != datatypes.Tag.comemi:
            result.append(table)
        else:
            df = table.dataframe.copy()
            index_columns = ["Region", "Year", "CommName"]
            data_columns = [
                colname for colname in df.columns.values if colname not in index_columns
            ]
            df, names = utils.explode(df, data_columns)
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


def process_commodities(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    regions = ",".join(
        utils.single_column(tables, datatypes.Tag.book_regions_map, "Region")
    )

    result = []
    for table in tables:
        if table.tag != datatypes.Tag.fi_comm and table.tag != datatypes.Tag.fi_comm:
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
            result.append(replace(table, dataframe=df, tag=datatypes.Tag.fi_comm))

    return result


def process_years(tables: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    # Datayears is the set of all years in ~FI_T's Year column
    # We ignore values < 1000 because those signify interpolation/extrapolation rules
    # (see Table 8 of Part IV of the Times Documentation)
    datayears = tables[datatypes.Tag.fi_t]["Year"].where(lambda x: x >= 1000).dropna()
    datayears = datayears.drop_duplicates().sort_values()
    tables["DataYear"] = pd.DataFrame({"Year": datayears})

    # Pastyears is the set of all years before ~StartYear
    start_year = tables[datatypes.Tag.start_year]["VALUE"][0]
    pastyears = datayears.where(lambda x: x <= start_year).dropna()
    tables["PastYear"] = pd.DataFrame({"Year": pastyears})

    # Modelyears is the union of pastyears and the representative years of the model (middleyears)
    modelyears = (
        pastyears.append(tables[datatypes.Tag.time_periods]["M"], ignore_index=True)
        .drop_duplicates()
        .sort_values()
    )
    tables["ModelYear"] = pd.DataFrame({"Year": modelyears})

    return tables


def process_processes(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    result = []
    for table in tables:
        if table.tag != datatypes.Tag.fi_process:
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


def process_transform_insert(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    regions = utils.single_column(tables, datatypes.Tag.book_regions_map, "Region")
    tfm_tags = [
        datatypes.Tag.tfm_ins,
        datatypes.Tag.tfm_dins,
        datatypes.Tag.tfm_topins,
        datatypes.Tag.tfm_upd,
        datatypes.Tag.tfm_comgrp,
    ]

    result = []
    dropped = []
    for table in tables:
        if not any(table.tag.startswith(t) for t in tfm_tags):
            result.append(table)

        elif table.tag in [
            datatypes.Tag.tfm_ins,
            datatypes.Tag.tfm_ins_ts,
            datatypes.Tag.tfm_upd,
            datatypes.Tag.tfm_comgrp,
        ]:
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

            if table.tag == datatypes.Tag.tfm_ins_ts:
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
                table.tag == datatypes.Tag.tfm_ins_ts
                and set(df.columns) & query_columns == {"CSet_CN"}
                and has_no_wildcards(df["CSet_CN"])
            ):
                df["Comm-OUT"] = df["CSet_CN"]
                df["Comm-IN"] = df["CSet_CN"]
                df.drop(columns=["CSet_CN"], inplace=True)
                result.append(replace(table, dataframe=df, tag=datatypes.Tag.fi_t))
            elif (
                table.tag == datatypes.Tag.tfm_ins_ts
                and set(df.columns) & query_columns == {"PSet_PN"}
                and has_no_wildcards(df["PSet_PN"])
            ):
                df.rename(columns={"PSet_PN": "TechName"}, inplace=True)
                result.append(replace(table, dataframe=df, tag=datatypes.Tag.fi_t))
            else:
                # wildcard expansion will happen later
                if table.tag == datatypes.Tag.tfm_ins_ts:
                    # ~TFM_INS-TS: Regions should be specified in a column with header=Region and columns in data area are YEARS
                    data_columns = [
                        colname
                        for colname in df.columns.values
                        if colname not in known_columns | {"Region", "TS_Filter"}
                    ]
                    df, years = utils.explode(df, data_columns)
                    df["Year"] = years
                for standard_col in known_columns:
                    if standard_col not in df.columns:
                        df[standard_col] = [None] * len(df)
                result.append(replace(table, dataframe=df))

        elif table.tag == datatypes.Tag.tfm_dins:
            df = table.dataframe.copy()
            nrows = df.shape[0]

            # Find all columns with -, first part is region and sum over second part
            pairs = [(col.split("-")[0], col) for col in df.columns if "-" in col]
            for region, tup in groupby(
                sorted(pairs, key=lambda p: p[0]), lambda p: p[0]
            ):
                cols = [t[1] for t in tup]
                df[region] = df.loc[:, cols].sum(axis=1)
                df[region] = df[region].apply(lambda x: utils.round_sig(x, 15))
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

            result.append(replace(table, dataframe=df, tag=datatypes.Tag.fi_t))

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
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    result = []
    dropped = []
    for table in tables:
        if table.tag != datatypes.Tag.tfm_ava:
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


def process_wildcards(tables: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    # We need to be able to fetch processes based on any combination of name, description, set, comm-in, or comm-out
    # So we construct tables whose indices are names, etc. and use pd.filter
    processes = tables[datatypes.Tag.fi_process]
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
    processes_and_commodities = tables[datatypes.Tag.fi_t]
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
    commodities = tables[datatypes.Tag.fi_comm]
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
        df = df.filter(
            regex=utils.create_regexp(pattern), axis="index"
        ).drop_duplicates()
        exclude = df.filter(
            regex=utils.create_negative_regexp(pattern), axis="index"
        ).index
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

    for tag in {datatypes.Tag.tfm_ins, datatypes.Tag.tfm_ins_ts, datatypes.Tag.tfm_upd}:
        if tag in tables:
            start_time = time.time()
            upd = tables[tag]
            new_rows = []
            # reset index to make sure there are no duplicates
            tables[datatypes.Tag.fi_t].reset_index(drop=True, inplace=True)
            if tag == datatypes.Tag.tfm_upd:
                # copy old index to new column 'index'
                tables[datatypes.Tag.fi_t].reset_index(inplace=True)
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
                df = tables[datatypes.Tag.fi_t]
                if any(df.index.duplicated()):
                    raise ValueError("~FI_T table has duplicated indices")
                if tag == datatypes.Tag.tfm_upd:
                    # construct query into ~FI_T to get indices of matching rows
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
                    # so that we can update the original table, copy original index back that was lost when merging
                    df = df.set_index("index")
                    # for speed, extract just the VALUE column as that is the only one being updated
                    df = df[["VALUE"]]
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
                    tables[datatypes.Tag.fi_t].update(df)
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
                    new_rows.append(row)

            if tag != datatypes.Tag.tfm_upd:
                new_rows.append(df)
                tables[datatypes.Tag.fi_t] = pd.concat(new_rows, ignore_index=True)

            print(
                f"  process_wildcards: {tag} took {time.time()-start_time:.2f} seconds for {len(upd)} rows"
            )

    return tables


def process_time_slices(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    def timeslices_table(
        table: datatypes.EmbeddedXlTable,
        regions: str,
        result: List[datatypes.EmbeddedXlTable],
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
            previousVal = timeslices[i - 1][1]
            timeslices[i] = (col, previousVal if val is None else previousVal + val)
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
    regions = ",".join(
        utils.single_column(tables, datatypes.Tag.book_regions_map, "Region")
    )

    for table in tables:
        if table.tag != datatypes.Tag.time_slices:
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
    input: Dict[str, DataFrame], mappings: List[datatypes.TimesXlMap]
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
            if isinstance(t, datatypes.EmbeddedXlTable):
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


def expand_rows_parallel(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(expand_rows, tables))
