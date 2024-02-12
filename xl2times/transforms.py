from collections import defaultdict
from pandas.core.frame import DataFrame
from pathlib import Path
import pandas as pd
from dataclasses import replace
from typing import Dict, List
from more_itertools import locate, one
from itertools import groupby
import re
from concurrent.futures import ProcessPoolExecutor
import time
from functools import reduce
from . import datatypes
from . import utils

query_columns = {
    "pset_set",
    "pset_pn",
    "pset_pd",
    "pset_ci",
    "pset_co",
    "cset_set",
    "cset_cn",
    "cset_cd",
}

csets_ordered_for_pcg = ["DEM", "MAT", "NRG", "ENV", "FIN"]
default_pcg_suffixes = [
    cset + io for cset in csets_ordered_for_pcg for io in ["I", "O"]
]

attr_prop = {
    "COM_LIM": "limtype",
    "COM_TSL": "ctslvl",
    "COM_TYPE": "ctype",
    "PRC_PCG": "primarycg",
    "PRC_TSL": "tslvl",
    "PRC_VINT": "vintage",
}


def remove_comment_rows(
    config: datatypes.Config,
    table: datatypes.EmbeddedXlTable,
    model: datatypes.TimesModel,
) -> datatypes.EmbeddedXlTable:
    """
    Return a modified copy of 'table' where rows with cells starting with symbols
    indicating a comment row in any column have been deleted. Comment row symbols
    are column name dependant and are specified in the config.

    :param table:       Table object in EmbeddedXlTable format.
    :return:            Table object in EmbeddedXlTable format without comment rows.
    """
    if table.dataframe.size == 0:
        return table

    df = table.dataframe.copy()

    tag = table.tag.split(":")[0]

    if tag in config.row_comment_chars:
        chars_by_colname = config.row_comment_chars[tag]
    else:
        return table

    comment_rows = set()

    for colname in df.columns:
        if colname in chars_by_colname.keys():
            comment_rows.update(
                list(
                    locate(
                        df[colname],
                        lambda cell: isinstance(cell, str)
                        and (cell.startswith(tuple(chars_by_colname[colname]))),
                    )
                )
            )

    df.drop(index=list(comment_rows), inplace=True)
    df.reset_index(drop=True, inplace=True)

    return replace(table, dataframe=df)


def remove_comment_cols(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
    """
    Return a modified copy of 'table' where columns with labels starting with '*'
    have been deleted. Assumes that any leading spaces in the original input table
    have been removed.

    :param table:       Table object in EmbeddedXlTable format.
    :return:            Table object in EmbeddedXlTable format without comment columns.
    """
    if table.dataframe.size == 0:
        return table

    comment_cols = [
        colname
        for colname in table.dataframe.columns
        if isinstance(colname, str) and colname.startswith("*")
    ]

    df = table.dataframe.drop(comment_cols, axis=1)
    df.reset_index(drop=True, inplace=True)
    return replace(table, dataframe=df)


def remove_tables_with_formulas(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
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
        has = table.dataframe.map(is_formula).any(axis=None)
        if has:
            print(f"WARNING: Excluding table {table.tag} because it has formulas")
        return has

    return [table for table in tables if not has_formulas(table)]


def validate_input_tables(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Perform some basic validation (tag names are valid, no duplicate column labels), and
    remove empty tables (for recognized tags).
    """

    def discard(table):
        if table.tag in config.discard_if_empty:
            return not table.dataframe.shape[0]
        elif table.tag == datatypes.Tag.unitconversion:
            print("Dropping ~UNITCONVERSION table")
            return True
        else:
            return False

    result = []
    for table in tables:
        if not datatypes.Tag.has_tag(table.tag.split(":")[0]):
            print(f"WARNING: Dropping table with unrecognized tag {table.tag}")
            continue
        if discard(table):
            continue
        # Check for duplicate columns:
        seen = set()
        dupes = [x for x in table.dataframe.columns if x in seen or seen.add(x)]
        if len(dupes) > 0:
            print(
                f"WARNING: Duplicate columns in {table.range}, {table.sheetname},"
                f" {table.filename}: {','.join(dupes)}"
            )
        result.append(table)
    return result


def normalize_tags_columns(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Normalize (uppercase) tags and (lowercase) column names.


    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with normalzed values.
    """

    def normalize(table: datatypes.EmbeddedXlTable) -> datatypes.EmbeddedXlTable:
        # Only uppercase upto ':', the rest can be non-uppercase values like regions
        parts = table.tag.split(":")
        # assert len(parts) <= 2
        parts[0] = parts[0].upper()
        newtag = ":".join(parts)

        df = table.dataframe
        # Strip leading and trailing whitespaces from column names
        df.columns = df.columns.str.strip()

        col_name_map = {x: x.lower() for x in df.columns}
        df = df.rename(columns=col_name_map)

        return replace(table, tag=newtag, dataframe=df)

    return [normalize(table) for table in tables]


def normalize_column_aliases(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    for table in tables:
        tag = table.tag.split(":")[0]
        if tag in config.column_aliases:
            table.dataframe = table.dataframe.rename(
                columns=config.column_aliases[tag], errors="ignore"
            )
        else:
            print(f"WARNING: could not find {table.tag} in config.column_aliases")
        if len(set(table.dataframe.columns)) > len(table.dataframe.columns):
            raise ValueError(
                f"Table has duplicate column names (after normalization): {table}"
            )
    return tables


def include_tables_source(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Add a column specifying source filename to every table
    """

    def include_table_source(table: datatypes.EmbeddedXlTable):
        df = table.dataframe.copy()
        df["source_filename"] = table.filename
        return replace(table, dataframe=df)

    return [include_table_source(table) for table in tables]


def merge_tables(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    """
    Merge all tables in 'tables' with the same table tag, as long as they share the same
    column field values. Print a warning for those that don't share the same column values.
    Return a dictionary linking each table tag with its merged table or populate TimesModel class.

    :param tables:      List of tables in datatypes.EmbeddedXlTable format.
    :return:            Dictionary associating a given table tag with its merged table.
    """
    result = {}

    for key, value in groupby(sorted(tables, key=lambda t: t.tag), lambda t: t.tag):
        group = list(value)
        if not all(
            set(t.dataframe.columns) == set(group[0].dataframe.columns) for t in group
        ):
            cols = [(",".join(g.dataframe.columns), g) for g in group]
            print(
                f"WARNING: Cannot merge tables with tag {key} as their columns are not identical"
            )
            for c, table in cols:
                print(f"  {c} from {table.range}, {table.sheetname}, {table.filename}")
        else:
            df = pd.concat([table.dataframe for table in group], ignore_index=True)

            match key:
                case datatypes.Tag.fi_comm:
                    model.commodities = df
                case datatypes.Tag.fi_process:
                    model.processes = df
                case _:
                    result[key] = df
    return result


def process_flexible_import_tables(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
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
    # TODO: update this dictionary
    legal_values = {
        "limtype": {"LO", "UP", "FX"},
        # TODO: check what the values for the below should be
        "timeslice": set(model.ts_tslvl["tslvl"]),
        "commodity-out": set(
            utils.merge_columns(tables, datatypes.Tag.fi_comm, "commodity")
        ),
        "region": model.internal_regions,
        "currency": utils.single_column(tables, datatypes.Tag.currencies, "currency"),
        "other_indexes": {"INPUT", "OUTPUT", "DEMO", "DEMI"},
    }

    def get_colname(value):
        # Return the value in the desired format along with the associated category (if any)
        # TODO make sure to do case-insensitive comparisons when parsing composite column names
        if value.isdigit():
            return "year", int(value)
        for name, values in legal_values.items():
            if value.upper() in values:
                return name, value
        return None, value

    # TODO decide whether VedaProcessSets should become a new Enum type or part of TimesModelData type
    veda_process_sets = utils.single_table(tables, "VedaProcessSets").dataframe

    def process_flexible_import_table(
        table: datatypes.EmbeddedXlTable, veda_process_sets: DataFrame
    ) -> datatypes.EmbeddedXlTable:
        # Make sure it's a flexible import table, and return the table untouched if not
        if not table.tag.startswith(datatypes.Tag.fi_t) and table.tag not in {
            datatypes.Tag.tfm_upd,
        }:
            return table

        # Rename, add and remove specific columns if the circumstances are right
        # TODO: We should do a full scale normalisation here, incl. renaming of aliases
        df = table.dataframe

        nrows = df.shape[0]

        # datatypes.Tag column no longer used to identify data columns
        # https://veda-documentation.readthedocs.io/en/latest/pages/introduction.html#veda2-0-enhanced-features
        # TODO: Include other valid column headers
        known_columns = config.known_columns[datatypes.Tag.fi_t]
        data_columns = [x for x in df.columns if x not in known_columns]

        # Populate index columns
        index_columns = [
            "region",
            "process",
            "commodity",
            "commodity-in",
            "commodity-in-aux",
            "commodity-out",
            "commodity-out-aux",
            "attribute",
            "year",
            "timeslice",
            "limtype",
            "currency",
            "other_indexes",
        ]
        for colname in index_columns:
            if colname not in df.columns:
                df[colname] = [None] * nrows
        table = replace(table, dataframe=df)

        table = utils.apply_composite_tag(table)
        df = table.dataframe

        attribute = "attribute"
        if table.tag != datatypes.Tag.tfm_upd:
            df, attribute_suffix = utils.explode(df, data_columns)

            # Append the data column name to the Attribute column values
            if nrows > 0:
                i = df[attribute].notna()
                df.loc[i, attribute] = df.loc[i, attribute] + "~" + attribute_suffix[i]
                i = df[attribute].isna()
                df.loc[i, attribute] = attribute_suffix[i]

        # Capitalise all attributes, unless column type float
        if df[attribute].dtype != float:
            df[attribute] = df[attribute].str.upper()

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
        other = "other_indexes"
        for attr in df[attribute].unique():
            if attr == "END":
                i = df[attribute] == attr
                df.loc[i, "year"] = df.loc[i, "value"].astype("int") + 1
                df.loc[i, other] = "EOH"
                df.loc[i, attribute] = "PRC_NOFF"

        df = df.reset_index(drop=True)

        # Fill other_indexes for COST
        cost_mapping = {"MIN": "IMP", "EXP": "EXP", "IMP": "IMP"}
        i = (df[attribute] == "COST") & df["process"]
        for process in df[i]["process"].unique():
            veda_process_set = (
                veda_process_sets["sets"]
                .loc[veda_process_sets["process"] == process]
                .unique()
            )
            if veda_process_set.shape[0]:
                df.loc[i & (df["process"] == process), other] = cost_mapping[
                    veda_process_set[0]
                ]
            else:
                print(
                    f"WARNING: COST won't be processed as IRE_PRICE for {process}, because it is not in IMP/EXP/MIN"
                )

        # Use CommName to store the active commodity for EXP / IMP
        i = df[attribute].isin({"COST", "IRE_PRICE"})
        i_exp = i & (df[other] == "EXP")
        df.loc[i_exp, "commodity"] = df.loc[i_exp, "commodity-in"]
        i_imp = i & (df[other] == "IMP")
        df.loc[i_imp, "commodity"] = df.loc[i_imp, "commodity-out"]

        # Should have all index_columns and VALUE
        if table.tag == datatypes.Tag.fi_t and len(df.columns) != (
            len(index_columns) + 1
        ):
            raise ValueError(f"len(df.columns) = {len(df.columns)}")

        df["year2"] = df.apply(
            lambda row: int(row["year"].split("-")[1])
            if "-" in str(row["year"])
            else "EOH",
            axis=1,
        )

        df["year"] = df.apply(
            lambda row: int(row["year"].split("-")[0])
            if "-" in str(row["year"])
            else (row["year"] if row["year"] != "" else "BOH"),
            axis=1,
        )

        return replace(table, dataframe=df)

    return [process_flexible_import_table(t, veda_process_sets) for t in tables]


def process_user_constraint_tables(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Process all user constraint tables in 'tables'. The processing includes:
    - Removing, adding and renaming columns as needed.
    - Populating index columns.
    - Handing Attribute column and wildcards.
    See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16.


    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with all FI_T processed.
    """
    legal_values = {
        # TODO: load these from times-info.json
        "attribute": {
            "UC_ACT",
            "UC_ATTR",
            "UC_CAP",
            "UC_COMNET",
            "UC_COMPRD",
            "UC_FLO",
            "UC_NCAP",
            "UC_RHSRT",
            "UC_RHSRTS",
            "UC_RHSTS",
            "UC_R_EACH",
            "UC_R_SUM",
        },
        "region": model.internal_regions,
        "limtype": {"FX", "LO", "UP"},
        "side": {"LHS", "RHS"},
    }

    def get_colname(value):
        # TODO make sure to do case-insensitive comparisons when parsing composite column names
        if value.isdigit():
            return "year", int(value)
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
        df["uc_n"] = df["uc_n"].ffill()

        data_columns = [
            x for x in df.columns if x not in config.known_columns[datatypes.Tag.uc_t]
        ]

        # Populate columns
        nrows = df.shape[0]
        for colname in config.known_columns[datatypes.Tag.uc_t]:
            if colname not in df.columns:
                df[colname] = [None] * nrows
        table = replace(table, dataframe=df)

        # Fill missing regions using defaults (if specified)
        regions_lists = [x for x in table.uc_sets.keys() if x.upper().startswith("R")]
        if regions_lists and table.uc_sets[regions_lists[-1]] != "":
            regions = table.uc_sets[regions_lists[-1]]
            if regions.lower() != "allregions":
                df["region"] = df["region"].fillna(regions)

        # TODO: detect RHS correctly
        i = df["side"].isna()
        df.loc[i, "side"] = "LHS"

        table = utils.apply_composite_tag(table)
        df = table.dataframe
        df, attribute_suffix = utils.explode(df, data_columns)

        # Append the data column name to the Attribute column
        if nrows > 0:
            i = df["attribute"].notna()
            df.loc[i, "attribute"] = df.loc[i, "attribute"] + "~" + attribute_suffix[i]
            i = df["attribute"].isna()
            df.loc[i, "attribute"] = attribute_suffix[i]

        # Capitalise all attributes, unless column type float
        if df["attribute"].dtype != float:
            df["attribute"] = df["attribute"].str.upper()

        # Handle Attribute containing tilde, such as 'STOCK~2030'
        for attr in df["attribute"].unique():
            if "~" in attr:
                i = df["attribute"] == attr
                parts = attr.split("~")
                for value in parts:
                    colname, typed_value = get_colname(value)
                    if colname is None:
                        df.loc[i, "attribute"] = typed_value
                    else:
                        df.loc[i, colname] = typed_value

        return replace(table, dataframe=df)

    return [process_user_constraint_table(t) for t in tables]


def fill_in_missing_values(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Attempt to fill in missing values for all tables except update tables (as these contain
    wildcards). How the value is filled in depends on the name of the column the empty values
    belong to.

    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with empty values filled in.
    """
    result = []
    start_year = one(utils.single_column(tables, datatypes.Tag.start_year, "value"))
    # TODO there are multiple currencies
    currency = utils.single_column(tables, datatypes.Tag.currencies, "currency")[0]
    # The default regions for VT_* files is given by ~BookRegions_Map:
    vt_regions = defaultdict(list)
    brm = utils.single_table(tables, datatypes.Tag.book_regions_map).dataframe
    utils.missing_value_inherit(brm, "bookname")
    for _, row in brm.iterrows():
        if row["region"] in model.internal_regions:
            vt_regions[row["bookname"]].append(row["region"])

    ele_default_tslvl = (
        "DAYNITE" if "DAYNITE" in model.ts_tslvl["tslvl"].unique() else "ANNUAL"
    )

    def fill_in_missing_values_table(table):
        df = table.dataframe.copy()
        for colname in df.columns:
            # TODO make this more declarative
            if colname in ["sets", "csets", "process"]:
                utils.missing_value_inherit(df, colname)
            elif colname == "limtype" and table.tag == datatypes.Tag.fi_comm and False:
                isna = df[colname].isna()
                ismat = df["csets"] == "MAT"
                df.loc[isna & ismat, colname] = "FX"
                df.loc[isna & ~ismat, colname] = "LO"
            elif (
                colname == "limtype"
                and (table.tag == datatypes.Tag.fi_t or table.tag.startswith("~TFM"))
                and len(df) > 0
            ):
                isna = df[colname].isna()
                for lim in config.veda_attr_defaults["limtype"].keys():
                    df.loc[
                        isna
                        & df["attribute"]
                        .str.upper()
                        .isin(config.veda_attr_defaults["limtype"][lim]),
                        colname,
                    ] = lim
            elif colname == "timeslice" and len(df) > 0 and "attribute" in df.columns:
                isna = df[colname].isna()
                for timeslice in config.veda_attr_defaults["tslvl"].keys():
                    df.loc[
                        isna
                        & df["attribute"]
                        .str.upper()
                        .isin(config.veda_attr_defaults["tslvl"][timeslice]),
                        colname,
                    ] = timeslice
            elif (
                colname == "tslvl" and table.tag == datatypes.Tag.fi_process
            ):  # or colname == "CTSLvl" or colname == "PeakTS":
                isna = df[colname].isna()
                isele = df["sets"] == "ELE"
                df.loc[isna & isele, colname] = ele_default_tslvl
                df.loc[isna & ~isele, colname] = "ANNUAL"
            elif colname == "region":
                # Use BookRegions_Map to fill VT_* files, and all regions for other files
                matches = re.search(r"VT_([A-Za-z0-9]+)_", Path(table.filename).name)
                if matches is not None:
                    book = matches.group(1)
                    if book in vt_regions:
                        df.fillna({colname: ",".join(vt_regions[book])}, inplace=True)
                    else:
                        print(f"WARNING: book name {book} not in BookRegions_Map")
                else:
                    df.fillna({colname: ",".join(model.internal_regions)}, inplace=True)
            elif colname == "year":
                df.fillna({colname: start_year}, inplace=True)
            elif colname == "currency":
                df.fillna({colname: currency}, inplace=True)

        return replace(table, dataframe=df)

    for table in tables:
        if table.tag == datatypes.Tag.tfm_upd:
            # Missing values in update tables are wildcards and should not be filled in
            result.append(table)
        else:
            result.append(fill_in_missing_values_table(table))
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
    c = df.map(has_comma)
    columns_with_commas = [
        colname
        for colname in c.columns
        if colname not in query_columns and c[colname].any()
    ]
    if len(columns_with_commas) > 0:
        # Transform comma-separated strings into lists
        df[columns_with_commas] = df[columns_with_commas].map(split_by_commas)
        for colname in columns_with_commas:
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html#pandas.DataFrame.explode
            df = df.explode(colname, ignore_index=True)
    return replace(table, dataframe=df)


def remove_invalid_values(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Remove all entries of any dataframes that are considered invalid. The rules for
    allowing an entry can be seen in the 'constraints' dictionary below.

    :param tables:      List of tables in EmbeddedXlTable format.
    :return:            List of tables in EmbeddedXlTable format with disallowed entries removed.
    """
    # TODO: This should be table type specific
    # TODO pull this out
    # TODO: This should take into account whether a specific dimension is required
    # Rules for allowing entries. Each entry of the dictionary designates a rule for a
    # a given column, and the values that are allowed for that column.
    constraints = {
        "csets": csets_ordered_for_pcg,
        "region": model.all_regions,
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


def process_units(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    units_map = {
        "activity": model.processes["tact"].unique(),
        "capacity": model.processes["tcap"].unique(),
        "commodity": model.commodities["unit"].unique(),
        "currency": tables[datatypes.Tag.currencies]["currency"].unique(),
    }

    model.units = pd.concat(
        [pd.DataFrame({"unit": v, "type": k}) for k, v in units_map.items()]
    )

    return tables


def process_time_periods(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    model.start_year = utils.get_scalar(datatypes.Tag.start_year, tables)
    active_pdef = utils.get_scalar(datatypes.Tag.active_p_def, tables)
    df = utils.single_table(tables, datatypes.Tag.time_periods).dataframe.copy()

    active_series = df[active_pdef.lower()]
    # Remove empty rows
    active_series.dropna(inplace=True)

    df = pd.DataFrame({"d": active_series})
    # Start years = start year, then cumulative sum of period durations
    df["b"] = (active_series.cumsum() + model.start_year).shift(
        1, fill_value=model.start_year
    )
    df["e"] = df.b + df.d - 1
    df["m"] = df.b + ((df.d - 1) // 2)
    df["year"] = df.m

    model.time_periods = df.astype(int)

    return tables


def process_regions(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Include IMPEXP and MINRNW together with the user-defined regions in the AllRegions set.
    IMPEXP and MINRNW are external regions that are defined by default by Veda.
    """

    model.all_regions.update((["IMPEXP", "MINRNW"]))
    model.internal_regions.update(
        utils.single_column(tables, datatypes.Tag.book_regions_map, "region")
    )
    model.all_regions.update(model.internal_regions)

    # Apply regions filter
    if config.filter_regions:
        keep_regions = model.internal_regions.intersection(config.filter_regions)
        if keep_regions:
            model.internal_regions = keep_regions
        else:
            print("WARNING: Regions filter not applied; no valid entries found. ")

    return tables


def complete_dictionary(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    for k, v in [
        ("AllRegions", model.all_regions),
        ("Regions", model.internal_regions),
        ("DataYears", model.data_years),
        ("PastYears", model.past_years),
        ("ModelYears", model.model_years),
    ]:
        if "region" in k.lower():
            column_list = ["region"]
        else:
            column_list = ["year"]

        tables[k] = pd.DataFrame(sorted(v), columns=column_list)

    # Dataframes
    for k, v in {
        "Attributes": model.attributes,
        "Commodities": model.commodities,
        "CommodityGroups": model.commodity_groups,
        "CommodityGroupMap": model.com_gmap,
        "Processes": model.processes,
        "Topology": model.topology,
        "Trade": model.trade,
        "TimePeriods": model.time_periods,
        "TimeSlices": model.ts_tslvl,
        "TimeSliceMap": model.ts_map,
        "UserConstraints": model.user_constraints,
        "Units": model.units,
    }.items():
        tables[k] = v

    return tables


def capitalise_some_values(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Ensure that all attributes and units are uppercase
    """

    # TODO: This should include other dimensions
    # TODO: This should be part of normalisation

    colnames = ["attribute", "tact", "tcap", "unit"]

    def capitalise_attributes_table(table: datatypes.EmbeddedXlTable):
        df = table.dataframe.copy()
        seen_cols = [colname for colname in colnames if colname in df.columns]
        if len(df) > 0:
            for seen_col in seen_cols:
                df[seen_col] = df[seen_col].str.upper()
            return replace(table, dataframe=df)
        else:
            return table

    return [capitalise_attributes_table(table) for table in tables]


def apply_fixups(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    reg_com_flows = utils.single_table(tables, "ProcessTopology").dataframe.copy()
    reg_com_flows.drop(columns="io", inplace=True)

    def apply_fixups_table(table: datatypes.EmbeddedXlTable):
        if not table.tag.startswith(datatypes.Tag.fi_t) or table.dataframe.size == 0:
            return table

        df = table.dataframe.copy()

        # TODO: should we have a global list of column name -> type?
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Populate CommName based on defaults
        i = (
            df["attribute"]
            .str.upper()
            .isin(config.veda_attr_defaults["commodity"].keys())
            & df["commodity"].isna()
        )
        if len(df[i]) > 0:
            for attr in df[i]["attribute"].unique():
                for com_in_out in config.veda_attr_defaults["commodity"][attr.upper()]:
                    index = i & (df["attribute"] == attr) & (df["commodity"].isna())
                    if len(df[index]) > 0:
                        df.loc[index, ["commodity"]] = df[index][com_in_out]

        # Fill other indexes for some attributes
        # FLO_SHAR
        i = df["attribute"] == "SHARE-I"
        df.loc[i, "other_indexes"] = "NRGI"
        i = df["attribute"] == "SHARE-O"
        df.loc[i, "other_indexes"] = "NRGO"
        # ACT_EFF
        i = df["attribute"].isin({"CEFF", "CEFFICIENCY", "CEFF-I", "CEFF-O"})
        df.loc[i, "other_indexes"] = df[i]["commodity"]
        i = df["attribute"].isin({"EFF", "EFFICIENCY"})
        df.loc[i, "other_indexes"] = "ACT"
        # FLO_EMIS
        i = df["attribute"].isin({"ENV_ACT", "ENVACT"})
        df.loc[i, "other_indexes"] = "ACT"

        # Fill CommName for COST (alias of IRE_PRICE) if missing
        if "attribute" in df.columns and "COST" in df["attribute"].unique():
            i = (df["attribute"] == "COST") & df["commodity"].isna()
            if any(i):
                df.loc[i, "commodity"] = df[i].apply(
                    lambda row: ",".join(
                        reg_com_flows.loc[
                            (reg_com_flows["region"] == row["region"])
                            & (reg_com_flows["process"] == row["process"]),
                            "commodity",
                        ].unique()
                    ),
                    axis=1,
                )
                # TODO: Expand rows if multiple comma-separated commodities are included

        return replace(table, dataframe=df)

    return [apply_fixups_table(table) for table in tables]


def generate_commodity_groups(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    process_tables = [t for t in tables if t.tag == datatypes.Tag.fi_process]
    commodity_tables = [t for t in tables if t.tag == datatypes.Tag.fi_comm]

    # Veda determines default PCG based on predetermined order and presence of OUT/IN commodity

    columns = ["region", "process", "primarycg"]
    reg_prc_pcg = pd.DataFrame(columns=columns)
    for process_table in process_tables:
        df = process_table.dataframe[columns]
        reg_prc_pcg = pd.concat([reg_prc_pcg, df])
    reg_prc_pcg.drop_duplicates(keep="first", inplace=True)

    # DataFrame with Veda PCGs specified in the process declaration tables
    reg_prc_veda_pcg = reg_prc_pcg.loc[
        reg_prc_pcg["primarycg"].isin(default_pcg_suffixes)
    ]

    # Extract commodities and their sets by region
    columns = ["region", "csets", "commodity"]
    comm_set = pd.DataFrame(columns=columns)
    for commodity_table in commodity_tables:
        df = commodity_table.dataframe[columns]
        comm_set = pd.concat([comm_set, df])
    comm_set.drop_duplicates(keep="first", inplace=True)

    prc_top = utils.single_table(tables, "ProcessTopology").dataframe

    # Commodity groups by process, region and commodity
    comm_groups = pd.merge(prc_top, comm_set, on=["region", "commodity"])
    comm_groups["commoditygroup"] = 0
    # Store the number of IN/OUT commodities of the same type per Region and Process in CommodityGroup
    for region in comm_groups["region"].unique():
        i_reg = comm_groups["region"] == region
        for process in comm_groups[i_reg]["process"].unique():
            i_reg_prc = i_reg & (comm_groups["process"] == process)
            for cset in comm_groups[i_reg_prc]["csets"].unique():
                i_reg_prc_cset = i_reg_prc & (comm_groups["csets"] == cset)
                for io in ["IN", "OUT"]:
                    i_reg_prc_cset_io = i_reg_prc_cset & (comm_groups["io"] == io)
                    comm_groups.loc[i_reg_prc_cset_io, "commoditygroup"] = sum(
                        i_reg_prc_cset_io
                    )

    def name_comm_group(df):
        """
        Return the name of a commodity group based on the member count
        """

        if df["commoditygroup"] > 1:
            return df["process"] + "_" + df["csets"] + df["io"][:1]
        elif df["commoditygroup"] == 1:
            return df["commodity"]
        else:
            return None

    # Replace commodity group member count with the name
    comm_groups["commoditygroup"] = comm_groups.apply(name_comm_group, axis=1)

    # Determine default PCG according to Veda
    comm_groups["DefaultVedaPCG"] = None
    for region in comm_groups["region"].unique():
        i_reg = comm_groups["region"] == region
        for process in comm_groups[i_reg]["process"]:
            i_reg_prc = i_reg & (comm_groups["process"] == process)
            default_set = False
            for io in ["OUT", "IN"]:
                if default_set:
                    break
                i_reg_prc_io = i_reg_prc & (comm_groups["io"] == io)
                for cset in csets_ordered_for_pcg:
                    i_reg_prc_io_cset = i_reg_prc_io & (comm_groups["csets"] == cset)
                    df = comm_groups[i_reg_prc_io_cset]
                    if not df.empty:
                        comm_groups.loc[i_reg_prc_io_cset, "DefaultVedaPCG"] = True
                        default_set = True
                        break

    # Add standard Veda PCGS named contrary to name_comm_group
    if reg_prc_veda_pcg.shape[0]:
        io_map = {"I": "IN", "O": "OUT"}
        suffix_to_cset = {suffix: suffix[:3] for suffix in default_pcg_suffixes}
        suffix_to_io = {suffix: io_map[suffix[3]] for suffix in default_pcg_suffixes}
        df = reg_prc_veda_pcg.copy()
        df["csets"] = df["primarycg"].replace(suffix_to_cset)
        df["io"] = df["primarycg"].replace(suffix_to_io)
        df["commoditygroup"] = df["process"] + "_" + df["primarycg"]
        columns = ["region", "process", "io", "csets"]
        df = pd.merge(
            df[columns + ["commoditygroup"]],
            comm_groups[columns + ["commodity"]],
            on=columns,
        )
        comm_groups = pd.concat([comm_groups, df])
        comm_groups.drop_duplicates(
            subset=["region", "process", "io", "commodity", "csets", "commoditygroup"],
            keep="first",
            inplace=True,
        )

    # TODO: Include info from ~TFM_TOPINS e.g. include RSDAHT2 in addition to RSDAHT

    i = comm_groups["commoditygroup"] != comm_groups["commodity"]

    model.topology = comm_groups
    model.com_gmap = comm_groups.loc[i, ["region", "commoditygroup", "commodity"]]

    return tables


def complete_commodity_groups(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    """
    Complete the list of commodity groups
    """

    commodities = generate_topology_dictionary(tables, model)[
        "commodities_by_name"
    ].rename(columns={"commodity": "commoditygroup"})
    cgs_in_top = model.topology["commoditygroup"].to_frame()
    commodity_groups = pd.concat([commodities, cgs_in_top])
    model.commodity_groups = commodity_groups.drop_duplicates(
        keep="first"
    ).reset_index()

    return tables


def generate_trade(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Generate inter-regional exchange topology
    """

    veda_set_ext_reg_mapping = {"IMP": "IMPEXP", "EXP": "IMPEXP", "MIN": "MINRNW"}
    dummy_process_cset = [["NRG", "IMPNRGZ"], ["MAT", "IMPMATZ"], ["DEM", "IMPDEMZ"]]
    veda_process_sets = utils.single_table(tables, "VedaProcessSets").dataframe

    ire_prc = pd.DataFrame(columns=["region", "process"])
    for table in tables:
        if table.tag == datatypes.Tag.fi_process:
            df = table.dataframe
            ire_prc = pd.concat(
                [ire_prc, df.loc[df["sets"] == "IRE", ["region", "process"]]]
            )
    ire_prc.drop_duplicates(keep="first", inplace=True)

    internal_regions = pd.DataFrame(model.internal_regions, columns=["region"])

    # Generate inter-regional exchange topology
    top_ire = pd.DataFrame(dummy_process_cset, columns=["csets", "process"])
    top_ire = pd.merge(top_ire, internal_regions, how="cross")
    top_ire = pd.merge(top_ire, model.topology[["region", "csets", "commodity"]])
    top_ire.drop(columns=["csets"], inplace=True)
    top_ire["io"] = "OUT"
    top_ire = pd.concat(
        [top_ire, model.topology[["region", "process", "commodity", "io"]]]
    )
    top_ire = pd.merge(top_ire, ire_prc)
    top_ire = pd.merge(top_ire, veda_process_sets)
    top_ire["region2"] = top_ire["sets"].replace(veda_set_ext_reg_mapping)
    top_ire[["origin", "destination", "in", "out"]] = None
    for io in ["IN", "OUT"]:
        index = top_ire["io"] == io
        top_ire.loc[index, [io.lower()]] = top_ire["commodity"].loc[index]
    na_out = top_ire["out"].isna()
    top_ire.loc[na_out, ["out"]] = top_ire["in"].loc[na_out]
    na_in = top_ire["in"].isna()
    top_ire.loc[na_in, ["in"]] = top_ire["out"].loc[na_in]
    is_imp_or_min = top_ire["sets"].isin({"IMP", "MIN"})
    is_exp = top_ire["sets"] == "EXP"
    top_ire.loc[is_imp_or_min, ["origin"]] = top_ire["region2"].loc[is_imp_or_min]
    top_ire.loc[is_imp_or_min, ["destination"]] = top_ire["region"].loc[is_imp_or_min]
    top_ire.loc[is_exp, ["origin"]] = top_ire["region"].loc[is_exp]
    top_ire.loc[is_exp, ["destination"]] = top_ire["region2"].loc[is_exp]
    top_ire.drop(columns=["region", "region2", "sets", "io"], inplace=True)
    top_ire.drop_duplicates(keep="first", inplace=True, ignore_index=True)

    cols_list = ["origin", "in", "destination", "out", "process"]
    # Include trade between internal regions
    for table in tables:
        if table.tag == datatypes.Tag.tradelinks_dins:
            df = table.dataframe
            f_links = df.rename(
                columns={
                    "reg1": "origin",
                    "comm1": "in",
                    "reg2": "destination",
                    "comm2": "out",
                }
            ).copy()
            top_ire = pd.concat([top_ire, f_links[cols_list]])
            # Check if any of the links are bi-directional
            if "b" in df["tradelink"].str.lower().unique():
                b_links = (
                    df[df["tradelink"].str.lower() == "b"]
                    .rename(
                        columns={
                            "reg1": "destination",
                            "comm1": "out",
                            "reg2": "origin",
                            "comm2": "in",
                        }
                    )
                    .copy()
                )
                top_ire = pd.concat([top_ire, b_links[cols_list]])

    filter_regions = model.internal_regions.union({"IMPEXP", "MINRNW"})
    i = top_ire["origin"].isin(filter_regions) & top_ire["destination"].isin(
        filter_regions
    )

    model.trade = top_ire[i].reset_index()

    return tables


def fill_in_missing_pcgs(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Fill in missing primary commodity groups in FI_Process tables.
    Expand primary commodity groups specified in FI_Process tables by a suffix.
    """

    def expand_pcg_from_suffix(df):
        """
        Return the name of a default primary commodity group based on suffix and process name
        """

        if df["primarycg"] in default_pcg_suffixes:
            return df["process"] + "_" + df["primarycg"]
        else:
            return df["primarycg"]

    result = []

    for table in tables:
        if table.tag != datatypes.Tag.fi_process:
            result.append(table)
        else:
            df = table.dataframe.copy()
            df["primarycg"] = df.apply(expand_pcg_from_suffix, axis=1)
            default_pcgs = model.topology.copy()
            default_pcgs = default_pcgs.loc[
                default_pcgs["DefaultVedaPCG"] == 1,
                ["region", "process", "commoditygroup"],
            ]
            default_pcgs.rename(columns={"commoditygroup": "primarycg"}, inplace=True)
            default_pcgs = pd.merge(
                default_pcgs,
                df.loc[df["primarycg"].isna(), df.columns != "primarycg"],
                how="right",
            )
            df = pd.concat([df, default_pcgs])
            df.drop_duplicates(
                subset=[
                    "sets",
                    "region",
                    "process",
                    "description",
                    "tact",
                    "tcap",
                    "tslvl",
                    "vintage",
                ],
                keep="last",
                inplace=True,
            )

            result.append(replace(table, dataframe=df))

    return result


def remove_fill_tables(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    # These tables collect data from elsewhere and update the table itself or a region below
    # The collected data is then presumably consumed via Excel references or vlookups
    # TODO: For the moment, assume that these tables are up-to-date. We will need a tool to do this.
    result = []
    for table in tables:
        if table.tag != datatypes.Tag.tfm_fill and not table.tag.startswith(
            datatypes.Tag.tfm_fill_r
        ):
            result.append(table)
    return result


def process_commodity_emissions(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    result = []
    for table in tables:
        if table.tag != datatypes.Tag.comemi:
            result.append(table)
        else:
            df = table.dataframe.copy()
            index_columns = ["region", "year", "commodity"]
            data_columns = [
                colname for colname in df.columns if colname not in index_columns
            ]
            df, names = utils.explode(df, data_columns)
            df.rename(columns={"value": "emcb"}, inplace=True)
            df["other_indexes"] = names
            df["other_indexes"] = df["other_indexes"].str.upper()

            if "region" in df.columns:
                df = df.astype({"region": "string"})
                df["region"] = df["region"].map(
                    lambda s: s.split(",") if isinstance(s, str) else s
                )
                df = df.explode("region", ignore_index=True)
                df = df[df["region"].isin(model.internal_regions)]

            nrows = df.shape[0]
            for colname in index_columns:
                if colname not in df.columns:
                    df[colname] = [None] * nrows

            result.append(replace(table, dataframe=df))

    return result


def process_commodities(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    regions = ",".join(model.internal_regions)

    result = []
    for table in tables:
        if table.tag != datatypes.Tag.fi_comm:
            result.append(table)
        else:
            df = table.dataframe.copy()
            nrows = df.shape[0]
            if "region" not in table.dataframe.columns:
                df.insert(1, "region", [regions] * nrows)
            if "limtype" not in table.dataframe.columns:
                df["limtype"] = [None] * nrows
            result.append(replace(table, dataframe=df, tag=datatypes.Tag.fi_comm))

    return result


def process_years(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    # Datayears is the set of all years in ~FI_T's Year column
    # We ignore values < 1000 because those signify interpolation/extrapolation rules
    # (see Table 8 of Part IV of the Times Documentation)

    datayears = (
        tables[datatypes.Tag.fi_t]["year"]
        .apply(lambda x: x if (x is not str) and x >= 1000 else None)
        .dropna()
    )
    model.data_years = datayears.drop_duplicates().sort_values()

    # Pastyears is the set of all years before ~StartYear
    model.past_years = datayears.where(lambda x: x < model.start_year).dropna()

    # Modelyears is the union of pastyears and the representative years of the model (middleyears)
    model.model_years = (
        pd.concat(
            [model.past_years, model.time_periods["m"]],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values()
    )

    return tables


def process_processes(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    result = []
    veda_sets_to_times = {"IMP": "IRE", "EXP": "IRE", "MIN": "IRE"}

    processes_and_sets = pd.DataFrame({"sets": [], "process": []})

    for table in tables:
        if table.tag != datatypes.Tag.fi_process:
            result.append(table)
        else:
            df = table.dataframe.copy()
            processes_and_sets = pd.concat(
                [processes_and_sets, df[["sets", "process"]].ffill()]
            )
            df.replace({"sets": veda_sets_to_times}, inplace=True)
            nrows = df.shape[0]
            # TODO: Use info from config instead. Introduce required columns in the meta file?
            add_columns = [
                (1, "region"),
                (6, "tslvl"),
                (7, "primarycg"),
                (8, "vintage"),
            ]
            for column in add_columns:
                if column[1] not in table.dataframe.columns:
                    df.insert(column[0], column[1], [None] * nrows)
            result.append(replace(table, dataframe=df))

    veda_process_sets = datatypes.EmbeddedXlTable(
        tag="VedaProcessSets",
        uc_sets={},
        sheetname="",
        range="",
        filename="",
        dataframe=processes_and_sets.loc[
            processes_and_sets["sets"].isin(veda_sets_to_times.keys())
        ],
    )

    result.append(veda_process_sets)

    return result


def process_topology(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Create topology
    """

    fit_tables = [t for t in tables if t.tag.startswith(datatypes.Tag.fi_t)]

    columns = [
        "region",
        "process",
        "commodity-in",
        "commodity-in-aux",
        "commodity-out",
        "commodity-out-aux",
    ]
    topology = pd.DataFrame(columns=columns)

    for fit_table in fit_tables:
        cols = [col for col in columns if col in fit_table.dataframe.columns]
        df = fit_table.dataframe[cols]
        topology = pd.concat([topology, df])

    topology = pd.melt(
        topology,
        id_vars=["region", "process"],
        var_name="io",
        value_name="commodity",
    )

    topology["process"] = topology["process"].ffill()
    topology.replace(
        {
            "io": {
                "commodity-in": "IN",
                "commodity-in-aux": "IN-A",
                "commodity-out": "OUT",
                "commodity-out-aux": "OUT-A",
            }
        },
        inplace=True,
    )
    topology.dropna(how="any", subset=["process", "commodity"], inplace=True)
    topology.drop_duplicates(keep="first", inplace=True)

    topology_table = datatypes.EmbeddedXlTable(
        tag="ProcessTopology",
        uc_sets={},
        sheetname="",
        range="",
        filename="",
        dataframe=topology,
    )

    tables.append(topology_table)

    return tables


def generate_dummy_processes(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
    include_dummy_processes=True,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Define dummy processes and specify default cost data for them to ensure that a TIMES model
    can always be solved. This covers situations when a commodity cannot be supplied
    by other means. Significant cost is usually associated with the activity of these
    processes to ensure that they are used as a last resort
    """

    if include_dummy_processes:
        # TODO: Activity units below are arbitrary. Suggest Veda devs not to have any.
        dummy_processes = [
            ["IMP", "IMPNRGZ", "Dummy Import of NRG", "PJ", "", "NRG"],
            ["IMP", "IMPMATZ", "Dummy Import of MAT", "Mt", "", "MAT"],
            ["IMP", "IMPDEMZ", "Dummy Import of DEM", "PJ", "", "DEM"],
        ]

        process_declarations = pd.DataFrame(
            dummy_processes,
            columns=["sets", "process", "description", "tact", "tcap", "primarycg"],
        )

        tables.append(
            datatypes.EmbeddedXlTable(
                tag="~FI_PROCESS",
                uc_sets={},
                sheetname="",
                range="",
                filename="",
                dataframe=process_declarations,
            )
        )

        process_data_specs = process_declarations[["process", "description"]].copy()
        # Use this as default activity cost for dummy processes
        # TODO: Should this be included in settings instead?
        process_data_specs["ACTCOST"] = 1111

        tables.append(
            datatypes.EmbeddedXlTable(
                tag="~FI_T",
                uc_sets={},
                sheetname="",
                range="",
                filename="",
                dataframe=process_data_specs,
            )
        )

    return tables


def process_tradelinks(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """
    Transform tradelinks to tradelinks_dins
    """

    result = []
    for table in tables:
        if table.tag == datatypes.Tag.tradelinks:
            df = table.dataframe
            sheetname = table.sheetname.lower()
            comm = df.columns[0]
            destinations = [c for c in df.columns if c != comm]
            df.rename(columns={comm: "origin"}, inplace=True)
            df = pd.melt(
                df, id_vars=["origin"], value_vars=destinations, var_name="destination"
            )
            df = df[df["value"] == 1].drop(columns=["value"])
            df["destination"] = df["destination"].str.upper()
            df.drop_duplicates(keep="first", inplace=True)

            if sheetname == "uni":
                df["tradelink"] = "u"
            elif sheetname == "bi":
                df["tradelink"] = "b"
            else:
                df["tradelink"] = 1
                # Determine whether a trade link is bi- or unidirectional
                td_type = (
                    df.groupby(["regions"])["tradelink"].agg("count").reset_index()
                )
                td_type.replace({"tradelink": {1: "u", 2: "b"}}, inplace=True)
                df.drop(columns=["tradelink"], inplace=True)
                df = df.merge(td_type, how="inner", on="regions")

            # Add a column containing linked regions (directionless for bidirectional links)
            df["regions"] = df.apply(
                lambda row: tuple(sorted([row["origin"], row["destination"]]))
                if row["tradelink"] == "b"
                else tuple([row["origin"], row["destination"]]),
                axis=1,
            )

            # Drop tradelink (bidirectional) duplicates
            df.drop_duplicates(
                subset=["regions", "tradelink"], keep="last", inplace=True
            )
            df.drop(columns=["regions"], inplace=True)
            df["comm"] = comm.upper()
            df["comm1"] = df["comm"]
            df["comm2"] = df["comm"]
            df.rename(columns={"origin": "reg1", "destination": "reg2"}, inplace=True)
            # Use Veda approach to naming of trade processes
            df["process"] = df.apply(
                lambda row: "T"
                + "_".join(
                    [
                        row["tradelink"].upper(),
                        row["comm"],
                        row["reg1"],
                        row["reg2"],
                        "01",
                    ]
                ),
                axis=1,
            )
            result.append(
                replace(table, dataframe=df, tag=datatypes.Tag.tradelinks_dins)
            )
        else:
            result.append(table)

    return result


def process_transform_insert_variants(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    """Reduces variants of TFM_INS like TFM_INS-TS to TFM_INS."""

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

    def is_year(col_name):
        """A column name is a year if it is an int >= 0"""
        return col_name.isdigit() and int(col_name) >= 0

    result = []
    for table in tables:
        if table.tag == datatypes.Tag.tfm_ins_ts:
            # ~TFM_INS-TS: Gather columns whose names are years into a single "Year" column:
            df = table.dataframe
            if "year" in df.columns:
                raise ValueError(f"TFM_INS-TS table already has Year column: {table}")
            # TODO: can we remove this hacky shortcut? Or should it be also applied to the AT variant?
            if set(df.columns) & query_columns == {"cset_cn"} and has_no_wildcards(
                df["cset_cn"]
            ):
                df.rename(columns={"cset_cn": "commodity"}, inplace=True)
                result.append(replace(table, dataframe=df, tag=datatypes.Tag.fi_t))
                continue
            elif set(df.columns) & query_columns == {"pset_pn"} and has_no_wildcards(
                df["pset_pn"]
            ):
                df.rename(columns={"pset_pn": "process"}, inplace=True)
                result.append(replace(table, dataframe=df, tag=datatypes.Tag.fi_t))
                continue

            other_columns = [
                col_name for col_name in df.columns if not is_year(col_name)
            ]
            df = pd.melt(
                df,
                id_vars=other_columns,
                var_name="year",
                value_name="value",
                ignore_index=False,
            )
            # Convert the year column to integer
            df["year"] = df["year"].astype("int")
            result.append(replace(table, dataframe=df, tag=datatypes.Tag.tfm_ins))
        elif table.tag == datatypes.Tag.tfm_ins_at:
            # ~TFM_INS-AT: Gather columns with attribute names into a single "Attribute" column
            df = table.dataframe
            if "attribute" in df.columns:
                raise ValueError(
                    f"TFM_INS-AT table already has Attribute column: {table}"
                )
            other_columns = [
                col_name
                for col_name in df.columns
                if col_name not in (config.all_attributes | config.attr_aliases)
            ]
            df = pd.melt(
                df,
                id_vars=other_columns,
                var_name="attribute",
                value_name="value",
                ignore_index=False,
            )
            result.append(replace(table, dataframe=df, tag=datatypes.Tag.tfm_ins))
        else:
            result.append(table)

    return result


def process_transform_tables(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    regions = model.internal_regions
    tfm_tags = [
        datatypes.Tag.tfm_ins,
        datatypes.Tag.tfm_ins_txt,
        datatypes.Tag.tfm_dins,
        datatypes.Tag.tfm_topins,
        datatypes.Tag.tfm_upd,
        datatypes.Tag.tfm_mig,
        datatypes.Tag.tfm_comgrp,
    ]

    result = []
    dropped = []
    for table in tables:
        if not any(table.tag.startswith(t) for t in tfm_tags):
            result.append(table)

        elif table.tag in [
            datatypes.Tag.tfm_ins,
            datatypes.Tag.tfm_ins_txt,
            datatypes.Tag.tfm_upd,
            datatypes.Tag.tfm_mig,
            datatypes.Tag.tfm_comgrp,
        ]:
            df = table.dataframe.copy()

            # Standardize column names
            known_columns = config.known_columns[table.tag] | query_columns

            # Handle Regions:
            if set(df.columns).isdisjoint(
                {x.lower() for x in regions} | {"allregions"}
            ):
                if "region" not in df.columns:
                    # If there's no region information at all, this table is for all regions:
                    df["region"] = ["allregions"] * len(df)
                # Else, we only have a "region" column so handle it below
            else:
                if "region" in df.columns:
                    raise ValueError(
                        "ERROR: table has a column called region as well as columns with"
                        f" region names:\n{table}\n{df.columns}"
                    )
                # We have columns whose names are regions, so gather them into a "region" column:
                region_cols = [
                    col_name
                    for col_name in df.columns
                    if col_name in set([x.lower() for x in regions]) | {"allregions"}
                ]
                other_columns = [
                    col_name for col_name in df.columns if col_name not in region_cols
                ]
                df = pd.melt(
                    df,
                    id_vars=other_columns,
                    var_name="region",
                    value_name="value",
                    ignore_index=False,
                )
                df = df.sort_index().reset_index(drop=True)  # retain original row order

            # This expands "allregions" into one row for each region:
            df["region"] = df["region"].map(
                lambda x: regions if x == "allregions" else x
            )
            df = df.explode(["region"])
            df["region"] = df["region"].str.upper()

            # Remove unknown columns and add missing known columns:
            unknown_columns = [
                col_name
                for col_name in df.columns
                if col_name not in known_columns | {"region", "value"}
            ]
            df.drop(columns=unknown_columns, inplace=True)
            for standard_col in known_columns:
                if standard_col not in df.columns:
                    df[standard_col] = [None] * len(df)

            result.append(replace(table, dataframe=df))
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
        for key, group in by_tag:
            print(
                f"WARNING: Dropped {len(group)} transform tables ({key})"
                f" rather than processing them"
            )

    return result


def process_transform_availability(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
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
        for key, group in by_tag:
            print(
                f"WARNING: Dropped {len(group)} transform availability tables ({key})"
                f" rather than processing them"
            )

    return result


def filter_by_pattern(df, pattern):
    # Duplicates can be created when a process has multiple commodities that match the pattern
    df = df.filter(regex=utils.create_regexp(pattern), axis="index").drop_duplicates()
    exclude = df.filter(regex=utils.create_negative_regexp(pattern), axis="index").index
    return df.drop(exclude)


def intersect(acc, df):
    if acc is None:
        return df
    return acc.merge(df)


def get_matching_processes(row, dictionary):
    matching_processes = None
    for col, key in [
        ("pset_pn", "processes_by_name"),
        ("pset_pd", "processes_by_desc"),
        ("pset_set", "processes_by_sets"),
        ("pset_ci", "processes_by_comm_in"),
        ("pset_co", "processes_by_comm_out"),
    ]:
        if row[col] is not None:
            matching_processes = intersect(
                matching_processes, filter_by_pattern(dictionary[key], row[col].upper())
            )
    if matching_processes is not None and any(matching_processes.duplicated()):
        raise ValueError("duplicated")
    return matching_processes


def get_matching_commodities(row, dictionary):
    matching_commodities = None
    for col, key in [
        ("cset_cn", "commodities_by_name"),
        ("cset_cd", "commodities_by_desc"),
        ("cset_set", "commodities_by_sets"),
    ]:
        if row[col] is not None:
            matching_commodities = intersect(
                matching_commodities,
                filter_by_pattern(dictionary[key], row[col].upper()),
            )
    return matching_commodities


def df_indexed_by_col(df, col):
    # Set df index using an existing column; make index is uppercase
    df = df.dropna().drop_duplicates()
    index = df[col].str.upper()
    df = df.set_index(index).rename_axis("index")

    if len(df.columns) > 1:
        df = df.drop(columns=col)
    return df


def generate_topology_dictionary(
    tables: Dict[str, DataFrame], model: datatypes.TimesModel
) -> Dict[str, DataFrame]:
    # We need to be able to fetch processes based on any combination of name, description, set, comm-in, or comm-out
    # So we construct tables whose indices are names, etc. and use pd.filter

    dictionary = dict()
    pros = model.processes
    coms = model.commodities
    pros_and_coms = tables[datatypes.Tag.fi_t]

    dict_info = [
        {"key": "processes_by_name", "df": pros[["process"]], "col": "process"},
        {
            "key": "processes_by_desc",
            "df": pros[["process", "description"]],
            "col": "description",
        },
        {"key": "processes_by_sets", "df": pros[["process", "sets"]], "col": "sets"},
        {
            "key": "processes_by_comm_in",
            "df": pros_and_coms[["process", "commodity-in"]],
            "col": "commodity-in",
        },
        {
            "key": "processes_by_comm_out",
            "df": pros_and_coms[["process", "commodity-out"]],
            "col": "commodity-out",
        },
        {"key": "commodities_by_name", "df": coms[["commodity"]], "col": "commodity"},
        {
            "key": "commodities_by_desc",
            "df": coms[["commodity", "description"]],
            "col": "description",
        },
        {
            "key": "commodities_by_sets",
            "df": coms[["commodity", "csets"]],
            "col": "csets",
        },
    ]

    for entry in dict_info:
        dictionary[entry["key"]] = df_indexed_by_col(entry["df"], entry["col"])

    return dictionary


def process_uc_wildcards(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    tag = datatypes.Tag.uc_t

    def make_str(df):
        if df is not None and len(df) != 0:
            list_from_df = df.iloc[:, 0].unique()
            return ",".join(list_from_df)
        else:
            return None

    if tag in tables:
        start_time = time.time()
        df = tables[tag]
        dictionary = generate_topology_dictionary(tables, model)

        df["process"] = df.apply(
            lambda row: make_str(get_matching_processes(row, dictionary)), axis=1
        )
        df["commodity"] = df.apply(
            lambda row: make_str(get_matching_commodities(row, dictionary)), axis=1
        )

        cols_to_drop = [col for col in df.columns if col in query_columns]

        df = expand_rows(
            datatypes.EmbeddedXlTable(
                tag="",
                uc_sets={},
                sheetname="",
                range="",
                filename="",
                dataframe=df.drop(columns=cols_to_drop),
            )
        ).dataframe

        tables[tag] = df

        print(
            f"  process_uc_wildcards: {tag} took {time.time() - start_time:.2f} seconds for {len(df)} rows"
        )

    return tables


def process_wildcards(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    topology = generate_topology_dictionary(tables, model)

    def match_wildcards(
        row: pd.Series,
    ) -> tuple[DataFrame | None, DataFrame | None] | None:
        matching_processes = get_matching_processes(row, topology)
        matching_commodities = get_matching_commodities(row, topology)
        if (matching_processes is None or len(matching_processes) == 0) and (
            matching_commodities is None or len(matching_commodities) == 0
        ):  # TODO is this necessary? Try without?
            # TODO debug these
            print(f"WARNING: a row matched no processes or commodities")
            return None
        return matching_processes, matching_commodities

    def query(
        table: DataFrame,
        processes: DataFrame | None,
        commodities: DataFrame | None,
        attribute: str | None,
        region: str | None,
    ) -> pd.Index:
        qs = []
        if processes is not None and not processes.empty:
            qs.append(f"process in [{','.join(map(repr, processes['process']))}]")
        if commodities is not None and not commodities.empty:
            qs.append(f"commodity in [{','.join(map(repr, commodities['commodity']))}]")
        if attribute is not None:
            qs.append(f"attribute == '{attribute}'")
        if region is not None:
            qs.append(f"region == '{region}'")
        return table.query(" and ".join(qs)).index

    def eval_and_update(
        table: DataFrame, rows_to_update: pd.Index, new_value: str
    ) -> None:
        """Performs an inplace update of rows `rows_to_update` of `table` with `new_value`,
        which can be a update formula like `*2.3`."""
        if isinstance(new_value, str) and new_value[0] in {"*", "+", "-", "/"}:
            old_values = table.loc[rows_to_update, "value"]
            updated = old_values.astype(float).map(lambda x: eval("x" + new_value))
            table.loc[rows_to_update, "value"] = updated
        else:
            table.loc[rows_to_update, "value"] = new_value

    if datatypes.Tag.tfm_upd in tables:
        updates = tables[datatypes.Tag.tfm_upd]
        table = tables[datatypes.Tag.fi_t]
        new_tables = [table]
        # Reset FI_T index so that queries can determine unique rows to update
        tables[datatypes.Tag.fi_t].reset_index(inplace=True)

        # TFM_UPD: expand wildcards in each row, query FI_T to find matching rows,
        # evaluate the update formula, and add new rows to FI_T
        # TODO perf: collect all updates and go through FI_T only once?
        for _, row in updates.iterrows():
            if row["value"] is None:  # TODO is this really needed?
                continue
            match = match_wildcards(row)
            if match is None:
                continue
            processes, commodities = match
            rows_to_update = query(
                table, processes, commodities, row["attribute"], row["region"]
            )
            new_rows = table.loc[rows_to_update].copy()
            eval_and_update(new_rows, rows_to_update, row["value"])
            new_tables.append(new_rows)

        # Add new rows to table
        tables[datatypes.Tag.fi_t] = pd.concat(new_tables, ignore_index=True)

    if datatypes.Tag.tfm_ins in tables:
        updates = tables[datatypes.Tag.tfm_ins]
        table = tables[datatypes.Tag.fi_t]
        new_tables = []

        # TFM_INS: expand each row by wildcards, then add to FI_T
        for _, row in updates.iterrows():
            match = match_wildcards(row)
            # TODO perf: add matched procs/comms into column and use explode?
            new_rows = pd.DataFrame([row.filter(table.columns)])
            if match is not None:
                processes, commodities = match
                if processes is not None:
                    new_rows = processes.merge(new_rows, how="cross")
                if commodities is not None:
                    new_rows = commodities.merge(new_rows, how="cross")
            new_tables.append(new_rows)

        new_tables.append(tables[datatypes.Tag.fi_t])
        tables[datatypes.Tag.fi_t] = pd.concat(new_tables, ignore_index=True)

    if datatypes.Tag.tfm_ins_txt in tables:
        updates = tables[datatypes.Tag.tfm_ins_txt]

        # TFM_INS-TXT: expand row by wildcards, query FI_PROC/COMM for matching rows,
        # evaluate the update formula, and inplace update the rows
        for _, row in updates.iterrows():
            match = match_wildcards(row)
            if match is None:
                print(f"WARNING: TFM_INS-TXT row matched neither commodity nor process")
                continue
            processes, commodities = match
            if commodities is not None:
                table = model.commodities
            elif processes is not None:
                table = model.processes
            else:
                assert False  # All rows match either a commodity or a process

            # Query for rows with matching process/commodity and region
            rows_to_update = query(table, processes, commodities, None, row["region"])
            # Overwrite (inplace) the column given by the attribute (translated by attr_prop)
            # with the value from row
            # E.g. if row['attribute'] == 'PRC_TSL' then we overwrite 'tslvl'
            table.loc[rows_to_update, attr_prop[row["attribute"]]] = row["value"]

    if datatypes.Tag.tfm_mig in tables:
        updates = tables[datatypes.Tag.tfm_mig]
        table = tables[datatypes.Tag.fi_t]
        new_tables = []

        for _, row in updates.iterrows():
            match = match_wildcards(row)
            processes, commodities = match if match is not None else (None, None)
            # TODO should we also query on limtype?
            rows_to_update = query(
                table, processes, commodities, row["attribute"], row["region"]
            )
            new_rows = table.loc[rows_to_update].copy()
            # Modify values in all '*2' columns
            for c, v in row.items():
                if c.endswith("2") and v is not None:
                    new_rows.loc[:, c[:-1]] = v
            # Evaluate 'value' column based on existing values
            eval_and_update(new_rows, rows_to_update, row["value"])
            new_tables.append(new_rows)

        # Add new rows to table
        new_tables.append(tables[datatypes.Tag.fi_t])
        tables[datatypes.Tag.fi_t] = pd.concat(new_tables, ignore_index=True)

    return tables


def process_time_slices(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    def timeslices_table(
        table: datatypes.EmbeddedXlTable,
        regions: list,
        result: List[datatypes.EmbeddedXlTable],
    ):
        # User-specified timeslices (ordered)
        user_ts_levels = ["SEASON", "WEEKLY", "DAYNITE"]

        # Ensure that all timeslice levels are uppercase
        timeslices = {
            col.upper(): list(values.unique())
            for col, values in table.dataframe.items()
        }

        # Ensure that timeslices keys contain all user-specified levels
        for ts_level in user_ts_levels:
            if ts_level not in timeslices.keys():
                timeslices[ts_level] = list()

        # Remove ANNUAL if it is the only entry in SEASON
        if (
            len(timeslices["SEASON"]) == 1
            and timeslices["SEASON"][0].upper() == "ANNUAL"
        ):
            timeslices["SEASON"] = list()

        # Create a dataframe containing regions and timeslices
        reg_ts = pd.DataFrame({"region": regions})
        for ts_level in user_ts_levels:
            if timeslices[ts_level] != [None]:
                reg_ts = pd.merge(
                    reg_ts, pd.DataFrame({ts_level: timeslices[ts_level]}), how="cross"
                )

        # Include expanded names of timeslices in the dataframe
        ncols = len(reg_ts.columns)
        if ncols > 2:
            for i in range(2, ncols):
                reg_ts.iloc[:, i] = reg_ts.iloc[:, i - 1] + reg_ts.iloc[:, i]

        ts_groups = pd.merge(
            pd.DataFrame({"region": regions}),
            pd.DataFrame({"tslvl": ["ANNUAL"], "ts": ["ANNUAL"]}),
            how="cross",
        )

        if ncols > 1:
            ts_groups = pd.concat(
                [
                    ts_groups,
                    pd.melt(
                        reg_ts,
                        id_vars=["region"],
                        var_name="tslvl",
                        value_name="ts",
                    ),
                ]
            )

        # Generate timeslice map
        ts_maps = pd.DataFrame([], columns=["region", "parent", "timeslicemap"])
        if ncols > 2:
            ts_maps = pd.concat(
                [
                    ts_maps,
                    reg_ts.iloc[:, [0, 1, 2]].rename(
                        columns={
                            reg_ts.columns[1]: "parent",
                            reg_ts.columns[2]: "timeslicemap",
                        }
                    ),
                ]
            )

            if ncols == 4:
                ts_maps = pd.concat(
                    [
                        ts_maps,
                        reg_ts.iloc[:, [0, 1, 3]].rename(
                            columns={
                                reg_ts.columns[1]: "parent",
                                reg_ts.columns[3]: "timeslicemap",
                            }
                        ),
                    ]
                )
                ts_maps = pd.concat(
                    [
                        ts_maps,
                        reg_ts.iloc[:, [0, 2, 3]].rename(
                            columns={
                                reg_ts.columns[2]: "parent",
                                reg_ts.columns[3]: "timeslicemap",
                            }
                        ),
                    ]
                )

            ts_maps.drop_duplicates(keep="first", inplace=True)
            ts_maps.sort_values(by=list(ts_maps.columns), inplace=True)

        model.ts_map = DataFrame(ts_maps)
        model.ts_tslvl = DataFrame(ts_groups)

    result = []

    # TODO: Timeslices can differ from region to region
    regions = list(model.internal_regions)

    for table in tables:
        if table.tag != datatypes.Tag.time_slices:
            result.append(table)
        else:
            timeslices_table(table, regions, result)

    return result


def convert_to_string(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    for key, value in tables.items():
        tables[key] = value.map(
            lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
        )
    return tables


def convert_aliases(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    # Ensure TIMES names for all attributes
    replacement_dict = {}
    for k, v in config.veda_attr_defaults["aliases"].items():
        for alias in v:
            replacement_dict[alias] = k

    for table_type, df in tables.items():
        if "attribute" in df.columns:
            df.replace({"attribute": replacement_dict}, inplace=True)
        tables[table_type] = df

    # TODO: do this earlier
    model.attributes = tables[datatypes.Tag.fi_t]
    if datatypes.Tag.uc_t in tables.keys():
        model.user_constraints = tables[datatypes.Tag.uc_t]

    return tables


def rename_cgs(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    df = tables.get(datatypes.Tag.fi_t)
    if df is not None:
        i = df["other_indexes"].isin(default_pcg_suffixes)
        df.loc[i, "other_indexes"] = (
            df["process"].astype(str) + "_" + df["other_indexes"].astype(str)
        )
        tables[datatypes.Tag.fi_t] = df

    return tables


def fix_topology(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    mapping = {"IN-A": "IN", "OUT-A": "OUT"}

    model.topology.replace({"io": mapping}, inplace=True)

    return tables


def complete_processes(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    # Generate processes based on trade links

    trade_processes = pd.concat(
        [
            model.trade.loc[:, ["origin", "process", "in"]].rename(
                columns={"origin": "region", "in": "commodity"}
            ),
            model.trade.loc[:, ["destination", "process", "out"]].rename(
                columns={"destination": "region", "out": "commodity"}
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    undeclared_td = trade_processes.merge(
        model.processes.loc[:, ["region", "process"]], how="left", indicator=True
    )
    undeclared_td = undeclared_td.loc[
        (
            undeclared_td["region"].isin(model.internal_regions)
            & (undeclared_td["_merge"] == "left_only")
        ),
        ["region", "process", "commodity"],
    ]

    undeclared_td = undeclared_td.merge(
        model.commodities.loc[:, ["region", "commodity", "csets", "ctslvl", "unit"]],
        how="left",
    )
    undeclared_td.drop(columns=["commodity"], inplace=True)
    undeclared_td.rename(
        columns={"csets": "primarycg", "ctslvl": "tslvl", "unit": "tact"}, inplace=True
    )
    undeclared_td["sets"] = "IRE"
    undeclared_td.drop_duplicates(keep="last", inplace=True)

    # TODO: Handle possible duplicates
    for i in ["primarycg", "tslvl", "tact"]:
        duplicates = undeclared_td.loc[:, ["region", "process", i]].duplicated(
            keep=False
        )
        if any(duplicates):
            duplicates = undeclared_td.loc[duplicates, ["region", "process", i]]
            processes = duplicates["process"].unique()
            regions = duplicates["region"].unique()
            print(f"WARNING: Multiple possible {i} for {processes} in {regions}")

    model.processes = pd.concat([model.processes, undeclared_td], ignore_index=True)

    return tables


def apply_more_fixups(
    config: datatypes.Config,
    tables: Dict[str, DataFrame],
    model: datatypes.TimesModel,
) -> Dict[str, DataFrame]:
    # TODO: This should only be applied to processes introduced in BASE
    df = tables.get(datatypes.Tag.fi_t)
    if df is not None:
        index = df["attribute"] == "STOCK"
        # Temporary solution to include only processes defined in BASE
        i_vt = index & (df["source_filename"].str.contains("VT_", case=False))
        if any(index):
            extra_rows = []
            for region in df[index]["region"].unique():
                i_reg = index & (df["region"] == region)
                for process in df[(i_reg & i_vt)]["process"].unique():
                    i_reg_prc = i_reg & (df["process"] == process)
                    if any(i_reg_prc):
                        extra_rows.append(["NCAP_BND", region, process, "UP", 0, 2])
                    # TODO: TIMES already handles this. Drop?
                    if len(df[i_reg_prc]["year"].unique()) == 1:
                        year = df[i_reg_prc]["year"].unique()[0]
                        i_attr = (
                            df["attribute"].isin({"NCAP_TLIFE", "LIFE"})
                            & (df["region"] == region)
                            & (df["process"] == process)
                        )
                        if any(i_attr):
                            lifetime = df[i_attr]["value"].unique()[-1]
                        else:
                            lifetime = 30
                        extra_rows.append(
                            ["STOCK", region, process, "", year + lifetime, 0]
                        )
            if len(extra_rows) > 0:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            extra_rows,
                            columns=[
                                "attribute",
                                "region",
                                "process",
                                "limtype",
                                "year",
                                "value",
                            ],
                        ),
                    ]
                )
        tables[datatypes.Tag.fi_t] = df

    df = tables.get(datatypes.Tag.uc_t)
    if df is not None:
        # TODO: Handle defaults in a general way.
        # Use uc_n value if uc_desc is missing
        for uc_n in df["uc_n"].unique():
            index = df["uc_n"] == uc_n
            if all(df["uc_desc"][index].isna()):
                # Populate the first row only
                if any(index):
                    df.at[list(index).index(True), "uc_desc"] = uc_n

        tables[datatypes.Tag.uc_t] = df

    return tables


def expand_rows_parallel(
    config: datatypes.Config,
    tables: List[datatypes.EmbeddedXlTable],
    model: datatypes.TimesModel,
) -> List[datatypes.EmbeddedXlTable]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(expand_rows, tables))
