import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import reduce
from itertools import groupby
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from more_itertools import locate
from pandas.core.frame import DataFrame
from tqdm import tqdm

from . import utils
from .datatypes import Config, DataModule, EmbeddedXlTable, Tag, TimesModel
from .utils import max_workers

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

process_map = {
    "pset_pn": "processes_by_name",
    "pset_pd": "processes_by_desc",
    "pset_set": "processes_by_sets",
    "pset_ci": "processes_by_comm_in",
    "pset_co": "processes_by_comm_out",
}

commodity_map = {
    "cset_cn": "commodities_by_name",
    "cset_cd": "commodities_by_desc",
    "cset_set": "commodities_by_sets",
}

dm_cols = {"source_filename", "module_type", "submodule", "module_name"}


def remove_comment_rows(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Remove comment rows from all the tables.

    Assumes table dataframes are not empty.
    """
    result = []

    for table in tables:
        tag = Tag(table.tag)
        if tag in config.row_comment_chars:
            df = table.dataframe
            _remove_df_comment_rows(df, config.row_comment_chars[tag])
            # Keep the table if it stays not empty
            if not df.empty:
                result.append(table)
            else:
                continue
        else:
            result.append(table)

    return result


def _remove_df_comment_rows(
    df: DataFrame,
    comment_chars: dict[str, list],
) -> None:
    """Modify a dataframe in-place by deleting rows with cells starting with symbols
    indicating a comment row in any column. Comment row symbols are column name
    dependant and are passed as an additional argument.

    Parameters
    ----------
    df
        Dataframe.
    comment_chars
        Dictionary where keys are column names and values are lists of valid comment symbols
    """
    comment_rows = set()

    for colname in df.columns:
        if colname in comment_chars:
            comment_rows.update(
                list(
                    locate(
                        df[colname],
                        lambda cell: isinstance(cell, str)
                        and (cell.startswith(tuple(comment_chars[colname]))),
                    )
                )
            )

    df.drop(index=list(comment_rows), inplace=True)
    df.reset_index(drop=True, inplace=True)


def remove_comment_cols(table: EmbeddedXlTable) -> EmbeddedXlTable:
    """Return a modified copy of 'table' where columns with labels starting with '*'
    have been deleted. Assumes that any leading spaces in the original input table have
    been removed.

    Parameters
    ----------
    table
        Table object in EmbeddedXlTable format.

    Returns
    -------
    EmbeddedXlTable
        Table object in EmbeddedXlTable format without comment columns.
    """
    if table.dataframe.size == 0:
        return table

    comment_cols = [
        colname
        for colname in table.dataframe.columns
        if isinstance(colname, str) and colname.startswith("*")
    ]
    # Drop comment columns and any resulting empty rows
    df = table.dataframe.drop(comment_cols, axis=1).dropna(axis=0, how="all")
    df.reset_index(drop=True, inplace=True)
    return replace(table, dataframe=df)


def remove_exreg_cols(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Remove external region columns from all the tables except tradelinks."""
    external_regions = model.external_regions

    def remove_table_exreg_cols(
        table: EmbeddedXlTable,
    ) -> EmbeddedXlTable:
        """
        Return a modified copy of 'table' where columns that are external regions
        have been removed.
        """
        exreg_cols = [
            colname
            for colname in table.dataframe.columns
            if colname.upper() in external_regions
        ]

        if exreg_cols:
            df = table.dataframe.drop(exreg_cols, axis=1)
            return replace(table, dataframe=df)

        else:
            return table

    # Do not do anything if external_reagions is empty
    if not external_regions:
        return tables
    # Otherwise remove external region column from the relevant tables
    else:
        return [
            remove_table_exreg_cols(t) if t.tag != Tag.tradelinks else t for t in tables
        ]


def remove_tables_with_formulas(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Return a modified copy of 'tables' where tables with formulas (as identified by
    an initial '=') have deleted from the list.

    Parameters
    ----------
    config

    tables
        List of tables in EmbeddedXlTable format.
    model


    Returns
    -------
    list[EmbeddedXlTable]
        List of tables in EmbeddedXlTable format without any formulas.
    """

    def is_formula(s):
        return isinstance(s, str) and len(s) > 0 and s[0] == "="

    def has_formulas(table):
        has = table.dataframe.map(is_formula).any(axis=None)
        if has:
            logger.warning(f"Excluding table {table.tag} because it has formulas")
        return has

    return [table for table in tables if not has_formulas(table)]


def validate_input_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Perform some basic validation (tag names are valid, no duplicate column labels),
    and remove empty tables (for recognized tags).
    """

    def discard(table):
        if table.tag in config.discard_if_empty:
            return table.dataframe.empty
        elif table.tag == Tag.unitconversion:
            logger.info("Dropping ~UNITCONVERSION table")
            return True
        else:
            return False

    result = []
    for table in tables:
        if not Tag.has_tag(table.tag):
            logger.warning(f"Dropping table with unrecognized tag {table.tag}")
            continue
        if discard(table):
            continue
        # Check for duplicate columns:
        seen = set()
        dupes = [x for x in table.dataframe.columns if x in seen or seen.add(x)]
        if len(dupes) > 0:
            logger.warning(
                f"Duplicate columns in {table.range}, {table.sheetname},"
                f" {table.filename}: {','.join(dupes)}"
            )
        result.append(table)
    return result


def revalidate_input_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Perform further validation of input tables:
    - remove tables without required columns;
    - remove any row with missing values in any of the required columns;
    - add any column expected for processing downstream;
    - forward fill values in columns as specified in the config.
    """
    result = []
    for table in tables:
        tag = Tag(table.tag)
        # Replace empty strings with NA
        df = table.dataframe.replace("", pd.NA)
        # Drop columns that are all NA
        df = df.dropna(axis=1, how="all", ignore_index=True)
        required_cols = config.required_columns[tag]
        if required_cols:
            # Drop table if any column in required columns is missing
            missing_cols = required_cols.difference(df.columns)
            if missing_cols:
                logger.warning(
                    f"Dropping {tag.value} table withing range {table.range} on sheet {table.sheetname}"
                    f" in file {table.filename} due to missing required columns: {missing_cols}"
                )
                # Discard the table
                continue
            # Drop any rows with missing values in required columns
            df = df.dropna(
                subset=list(required_cols), axis=0, how="any", ignore_index=True
            )

        # Add columns in config.add_columns if missing
        add_columns = config.add_columns[tag]
        if add_columns and not add_columns.issubset(df.columns):
            cols_to_add = add_columns.difference(df.columns)
            for col in cols_to_add:
                df[col] = pd.NA
        # Forwards fill values in columns
        ff_cols = config.forward_fill_cols[tag].intersection(df.columns)
        for col in ff_cols:
            df[col] = df[col].ffill()
        # Append table to the list if reached this far
        result.append(replace(table, dataframe=df))

    return result


def normalize_tags_columns(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Normalize (uppercase) tags and (lowercase) column names.

    Parameters
    ----------
    config

    tables
        List of tables in EmbeddedXlTable format.
    model


    Returns
    -------
    list[EmbeddedXlTable]
        List of tables in EmbeddedXlTable format with normalzed values.
    """

    def normalize(table: EmbeddedXlTable) -> EmbeddedXlTable:
        # Only uppercase upto ':', the rest can be non-uppercase values like regions
        parts = table.tag.split(":")
        # assert len(parts) <= 2
        newtag = parts[0].upper()
        defaults = parts[1].strip() if len(parts) > 1 else None

        df = table.dataframe
        # Strip leading and trailing whitespaces from column names
        df.columns = df.columns.str.strip()

        col_name_map = {x: x.lower() for x in df.columns}
        df = df.rename(columns=col_name_map)

        return replace(table, tag=newtag, dataframe=df, defaults=defaults)

    return [normalize(table) for table in tables]


def normalize_column_aliases(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    for table in tables:
        tag = Tag(table.tag)
        if tag in config.column_aliases:
            table.dataframe = table.dataframe.rename(
                columns=config.column_aliases[tag], errors="ignore"
            )
        else:
            logger.warning(f"could not find {tag.value} in config.column_aliases")
        if len(set(table.dataframe.columns)) > len(table.dataframe.columns):
            raise ValueError(
                f"Table has duplicate column names (after normalization): {table}"
            )
    return tables


def include_tables_source(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Add a column specifying source filename to every table."""

    def include_table_source(table: EmbeddedXlTable):
        df = table.dataframe.copy()
        df["source_filename"] = Path(table.filename).stem
        df["module_type"] = DataModule.module_type(table.filename)
        df["submodule"] = DataModule.submodule(table.filename)
        df["module_name"] = DataModule.module_name(table.filename)
        return replace(table, dataframe=df)

    return [include_table_source(table) for table in tables]


def merge_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Merge all tables in 'tables' with the same table tag. Return a dictionary
    linking each table tag with its merged table or populate TimesModel class.

    Parameters
    ----------
    config

    tables
        List of tables in EmbeddedXlTable format.
    model


    Returns
    -------
    dict[str,DataFrame]
        Dictionary associating a given table tag with its merged table.
    """
    result = {}

    for key, value in groupby(sorted(tables, key=lambda t: t.tag), lambda t: t.tag):
        group = list(value)

        if len(group) == 0:
            continue

        df = pd.concat([table.dataframe for table in group], ignore_index=True)
        result[key] = df

        match key:
            case Tag.fi_process:
                # TODO: Find a better place for this (both info and processing)
                times_prc_sets = set(config.times_sets["PRC_GRP"])
                # Index of rows with TIMES process sets
                index = df["sets"].str.upper().isin(times_prc_sets)
                # Print a warning if non-TIMES sets are present
                if not all(index):
                    for _, row in df[~index].iterrows():
                        region, sets, process = row[["region", "sets", "process"]]
                        logger.warning(
                            f"WARNING: Unknown process set {sets} specified for process {process}"
                            f" in region {region}. The record will be dropped."
                        )
                # Exclude records with non-TIMES sets
                model.processes = df.loc[index]
            case _:
                result[key] = df

    return result


def apply_tag_specified_defaults(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:

    return [utils.apply_composite_tag(t) for t in tables]


def process_flexible_import_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Attempt to process all flexible import tables in 'tables'.

    The processing includes:

    - Checking that the table is indeed a flexible import table. If not, return it unmodified.
    - Removing, adding and renaming columns as needed.
    - Populating index columns.
    - Handing Attribute column and Other Indexes.

    See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16.

    Parameters
    ----------
    config

    tables
        List of tables in EmbeddedXlTable format.
    model


    Returns
    -------
    list[EmbeddedXlTable]
        List of tables in EmbeddedXlTable format with all FI_T processed.
    """
    # Get a list of allowed values for each category.
    # TODO: update this dictionary
    legal_values = {
        "limtype": set(config.times_sets["LIM"]),
        "timeslice": set(model.ts_tslvl["tslvl"]),
        "commodity": set(utils.merge_columns(tables, Tag.fi_comm, "commodity")),
        "region": model.internal_regions,
        "currency": utils.single_column(tables, Tag.currencies, "currency"),
        "other_indexes": set(config.times_sets["IN_OUT"])
        .union(config.times_sets["IMPEXP"])
        .union(default_pcg_suffixes),
    }

    def process_flexible_import_table(
        table: EmbeddedXlTable,
    ) -> EmbeddedXlTable:
        # Make sure it's a flexible import table, and return the table untouched if not
        if not table.tag == Tag.fi_t:
            return table

        # Rename, add and remove specific columns if the circumstances are right
        df = table.dataframe

        # Tag column no longer used to identify data columns
        # https://veda-documentation.readthedocs.io/en/latest/pages/introduction.html#veda2-0-enhanced-features

        known_columns = config.known_columns[Tag.fi_t]
        # TODO: Verify this list against other lists
        data_columns = [x for x in df.columns if x not in known_columns]

        # Populate index columns (same as known columns for this table type)
        index_columns = known_columns
        for colname in index_columns:
            if colname not in df.columns:
                df[colname] = None
        table = replace(table, dataframe=df)

        df = table.dataframe

        if data_columns:
            df, attribute_suffix = utils.explode(df, data_columns)
            # Append the data column name to the Attribute column values
            i = df["attribute"].notna()
            df.loc[i, "attribute"] = df.loc[i, "attribute"] + "~" + attribute_suffix[i]
            i = df["attribute"].isna()
            df.loc[i, "attribute"] = attribute_suffix[i]

        # Capitalise all attributes, unless column type float
        if df["attribute"].dtype != float:
            df["attribute"] = df["attribute"].str.upper()

        # Handle Attribute containing tilde, such as 'STOCK~2030'
        index = df["attribute"].str.contains("~")
        if any(index):
            for attr in df["attribute"][index].unique():
                i = index & (df["attribute"] == attr)
                parts = [part.strip() for part in attr.split("~")]
                for value in parts:
                    colname, typed_value = _get_colname(value, legal_values)
                    if colname is None:
                        df.loc[i, "attribute"] = typed_value
                    else:
                        df.loc[i, colname] = typed_value

        # Handle Other_Indexes
        other = "other_indexes"
        if "END" in df["attribute"]:
            i = df["attribute"] == "END"
            df.loc[i, "year"] = df.loc[i, "value"].astype("int") + 1
            df.loc[i, other] = "EOH"
            df.loc[i, "attribute"] = "PRC_NOFF"

        df = df.reset_index(drop=True)

        # Should have all index_columns and VALUE
        if len(df.columns) != (len(index_columns) + 1):
            # TODO: Should be ok to drop as long as the topology info is stored.
            if len(df.columns) == len(index_columns) and "value" not in df.columns:
                df["value"] = pd.NA
            else:
                raise ValueError(f"len(df.columns) = {len(df.columns)}")

        df["year2"] = df.apply(
            lambda row: (
                int(row["year"].split("-")[1]) if "-" in str(row["year"]) else "EOH"
            ),
            axis=1,
        )

        df["year"] = df.apply(
            lambda row: (
                int(row["year"].split("-")[0])
                if "-" in str(row["year"])
                else (row["year"] if row["year"] != "" else "BOH")
            ),
            axis=1,
        )

        return replace(table, dataframe=df)

    return [process_flexible_import_table(t) for t in tables]


def _get_colname(value, legal_values):
    """Return the value in the desired format along with the associated category (if any)."""
    if value.isdigit():
        return "year", int(value)
    for name, values in legal_values.items():
        if value.upper() in values:
            return name, value.upper()
    return None, value


def process_user_constraint_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Process all user constraint tables in 'tables'.

    The processing includes:

    - Removing, adding and renaming columns as needed.
    - Populating index columns.
    - Handing Attribute column and wildcards.

    See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16.

    Parameters
    ----------
    config

    tables
        List of tables in EmbeddedXlTable format.
    model


    Returns
    -------
    list[EmbeddedXlTable]
        List of tables in EmbeddedXlTable format with all FI_T processed.
    """
    legal_values = {
        "attribute": {attr for attr in config.all_attributes if attr.startswith("uc")},
        "region": model.internal_regions.union({"allregions".upper()}),
        "commodity": set(utils.merge_columns(tables, Tag.fi_comm, "commodity")),
        "timeslice": set(model.ts_tslvl["tslvl"]),
        "limtype": set(config.times_sets["LIM"]),
        "side": set(config.times_sets["SIDE"]),
    }

    def process_user_constraint_table(
        table: EmbeddedXlTable,
    ) -> EmbeddedXlTable:
        # See https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV_October-2016.pdf from p16

        if not table.tag == Tag.uc_t:
            return table

        df = table.dataframe

        # TODO: Handle pseudo-attributes in a more general way
        known_columns = config.known_columns[Tag.uc_t].difference(
            {"uc_attr", "allregions"}
        )

        data_columns = [x for x in df.columns if x not in known_columns]

        # Populate columns
        for colname in known_columns:
            if colname not in df.columns:
                df[colname] = None
        table = replace(table, dataframe=df)

        # TODO: detect RHS correctly
        i = df["side"].isna()
        df.loc[i, "side"] = "LHS"

        df, attribute_suffix = utils.explode(df, data_columns)

        # Append the data column name to the Attribute column
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
                parts = [part.strip() for part in attr.split("~")]
                for value in parts:
                    colname, typed_value = _get_colname(value, legal_values)
                    if colname is None:
                        df.loc[i, "attribute"] = typed_value
                    else:
                        df.loc[i, colname] = typed_value

        # Remove allregions from region column, so it can be filled in later
        i = df["region"].str.lower() == "allregions"
        df.loc[i, "region"] = pd.NA
        # Apply any general region specification if present
        # TODO: This assumes several regions lists may be present. Overwrite earlier?
        regions_lists = [x for x in table.uc_sets.keys() if x.upper().startswith("R_")]
        # Using the last regions_list
        if regions_lists and table.uc_sets[regions_lists[-1]] != "":
            regions = table.uc_sets[regions_lists[-1]]
            # Only expand regions if specified regions list is not allregions
            if regions.lower() != "allregions":
                # Only include valid model region names
                regions = model.internal_regions.intersection(
                    set(regions.upper().split(","))
                )
                regions = ",".join(regions)
                i_allregions = df["region"].isna()
                df.loc[i_allregions, "region"] = regions

        return replace(table, dataframe=df)

    return [process_user_constraint_table(t) for t in tables]


def generate_uc_properties(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Generate a dataframe containing User Constraint properties."""
    uc_tables = [table for table in tables if table.tag == Tag.uc_t]
    columns = [
        "uc_n",
        "description",
        "region",
        "region_action",
        "period_action",
        "timeslice_action",
        "uc_attr",
        "group_type",
        "side",
    ]
    user_constraints = pd.DataFrame(columns=columns)
    # Create df_list to hold DataFrames that will be concatenated later on
    df_list = list()
    for uc_table in uc_tables:
        uc_df = uc_table.dataframe
        # DataFrame with unique UC names and descriptions if they exist:
        df = (
            uc_df.loc[:, ["uc_n", "region", "description"]]
            .groupby(["uc_n", "region"], sort=False)
            .first()
        )
        df = df.reset_index()
        # Add info on how regions, periods and timeslices should be treated by the UCs
        for key in uc_table.uc_sets.keys():
            if key.startswith("R_"):
                df["region_action"] = key
            elif key.startswith("T_"):
                df["period_action"] = key
            elif key.startswith("TS_"):
                df["timeslice_action"] = key
        # Supplement with UC_ATTR if present
        index = uc_df["attribute"] == "UC_ATTR"
        if any(index):
            uc_attr_rows = uc_df.loc[index, ["uc_n", "region", "value", "side"]]
            # uc_attr is expected as column name
            uc_attr_rows = (
                uc_attr_rows.rename(columns={"value": "uc_attr"})
                .dropna()
                .drop_duplicates()
            )
            df = pd.merge(df, uc_attr_rows, on=["uc_n", "region"], how="left")
            # Remove UC_ATTR records from the original dataframe
            uc_table.dataframe = uc_df[~index].reset_index(drop=True)

        df_list.append(df)
    # Do further processing if df_list is not empty
    if df_list:
        # Create a single DataFrame with all UCs
        user_constraints = pd.concat(df_list).reset_index(drop=True)

        # Use name to populate description if it is missing
        index = user_constraints["description"].isna()
        if any(index):
            user_constraints.loc[index, ["description"]] = user_constraints["uc_n"][
                index
            ]
        # Handle uc_attr
        index = (
            user_constraints["uc_attr"].notna()
            if "uc_attr" in user_constraints.columns
            else list()
        )
        # Unpack uc_attr if not all the values in the column are na
        if any(index):
            # Handle semicolon-separated values
            i_pairs = index & user_constraints["uc_attr"].str.contains(";")
            if any(i_pairs):
                user_constraints.loc[i_pairs, "uc_attr"] = user_constraints[
                    i_pairs
                ].apply(
                    lambda row: row["uc_attr"].strip().split(";"),
                    axis=1,
                )
                user_constraints = user_constraints.explode(
                    "uc_attr", ignore_index=True
                )
                # Update index
                index = user_constraints["uc_attr"].notna()
            # Extend UC_NAME set with timeslice levels
            extended_uc_names = set(
                config.times_sets["UC_NAME"] + config.times_sets["TSLVL"]
            )
            uc_group_types = set(config.times_sets["UC_GRPTYPE"])

            def process_uc_attr(pair_str):
                items = [s.strip().upper() for s in pair_str.split(",")]
                group_types = [s for s in items if s in uc_group_types]
                uc_names = [s for s in items if s in extended_uc_names]
                if (
                    len(group_types) > 1
                    or len(uc_names) > 1
                    or len(group_types) + len(uc_names) != len(items)
                ):
                    raise ValueError(
                        f"uc_attr column value expected to be a pair (UC_GRPTYPE, UC_NAME/TSLVL) but got {pair_str}"
                    )
                return next(iter(group_types), None), next(iter(uc_names), None)

            group_types, uc_attrs = zip(
                *map(process_uc_attr, user_constraints.loc[index, "uc_attr"])
            )
            user_constraints.loc[index, "group_type"] = group_types
            user_constraints.loc[index, "uc_attr"] = uc_attrs
        # TODO: Can this (until user_constraints.explode) become a utility function?
        # Handle allregions by substituting it with a list of internal regions
        index = user_constraints["region"].str.lower() == "allregions"
        if any(index):
            user_constraints.loc[index, ["region"]] = ",".join(model.internal_regions)

        # Handle comma-separated regions
        index = user_constraints["region"].str.contains(",")
        if any(index):
            user_constraints.loc[index, "region"] = user_constraints[index].apply(
                lambda row: [
                    region
                    for region in row["region"].split(",")
                    if region in model.internal_regions
                ],
                axis=1,
            )
        # Explode regions
        user_constraints = user_constraints.explode("region", ignore_index=True)

    model.user_constraints = user_constraints.rename(
        columns={"uc_n": "name", "uc_attr": "uc_param"}
    )

    return tables


def fill_in_missing_values(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Attempt to fill in missing values for all tables except upd and mig tables (as these
    query data). How the value is filled in depends on the name of the column the
    empty values belong to.

    Parameters
    ----------
    config

    tables
        List of tables in EmbeddedXlTable format.
    model


    Returns
    -------
    list[EmbeddedXlTable]
        List of tables in EmbeddedXlTable format with empty values filled in.
    """
    result = []
    # TODO there are multiple currencies
    currency = utils.single_column(tables, Tag.currencies, "currency")[0]
    # The default regions for VT_* files is given by ~BookRegions_Map:
    vt_regions = defaultdict(list)
    brm = utils.single_table(tables, Tag.book_regions_map).dataframe
    for _, row in brm.iterrows():
        if row["region"] in model.internal_regions:
            vt_regions[row["bookname"]].append(row["region"])

    ele_default_tslvl = (
        "DAYNITE" if "DAYNITE" in set(model.ts_tslvl["tslvl"]) else "ANNUAL"
    )

    def fill_in_missing_values_table(table):
        df = table.dataframe.copy()
        default_values = config.column_default_value.get(table.tag, {})
        mapping_to_defaults = {"limtype": "limtype", "timeslice": "tslvl"}

        for colname in df.columns:
            # TODO make this more declarative
            # Apply default values to missing cells
            col_default_value = default_values.get(colname)
            if col_default_value is not None:
                df[colname] = df[colname].fillna(col_default_value)
            elif colname == "limtype" and table.tag == Tag.fi_comm and False:
                isna = df[colname].isna()
                ismat = df["csets"] == "MAT"
                df.loc[isna & ismat, colname] = "FX"
                df.loc[isna & ~ismat, colname] = "LO"
            elif colname in {"limtype", "timeslice"} and "attribute" in df.columns:
                isna = df[colname].isna()
                if any(isna):
                    key = mapping_to_defaults[colname]
                    for value in config.veda_attr_defaults[key].keys():
                        df.loc[
                            isna
                            & df["attribute"].isin(
                                config.veda_attr_defaults[key][value]
                            ),
                            colname,
                        ] = value
            elif (
                colname == "tslvl" and table.tag == Tag.fi_process
            ):  # or colname == "CTSLvl" or colname == "PeakTS":
                isna = df[colname].isna()
                isele = df["sets"] == "ELE"
                df.loc[isna & isele, colname] = ele_default_tslvl
                df.loc[isna & ~isele, colname] = "ANNUAL"
            elif colname == "region":
                # Use BookRegions_Map to fill VT_* files, and all regions for other files
                matches = re.search(r"VT_([A-Za-z0-9]+)_", Path(table.filename).stem)
                isna = df[colname].isna()
                if matches is not None:
                    book = matches.group(1)
                    if book in vt_regions:
                        df.loc[isna, [colname]] = ",".join(vt_regions[book])
                    else:
                        logger.warning(f"book name {book} not in BookRegions_Map")
                else:
                    df.loc[isna, [colname]] = ",".join(model.internal_regions)
            elif colname == "year":
                df.loc[df[colname].isna(), [colname]] = model.start_year
            elif colname == "currency":
                df.loc[df[colname].isna(), [colname]] = currency

        return replace(table, dataframe=df)

    for table in tables:
        if table.tag in [Tag.tfm_mig, Tag.tfm_upd]:
            # Missing values in these tables are wildcards and should not be filled in
            result.append(table)
        else:
            result.append(fill_in_missing_values_table(table))
    return result


def _has_comma(s) -> bool:
    return isinstance(s, str) and "," in s


def _split_by_commas(s):
    if _has_comma(s):
        return [x.strip() for x in s.split(",")]
    else:
        return _remove_empty_str(s)


def _remove_empty_str(s):
    return s if s != "" else pd.NA


def expand_rows(
    query_columns: set[str], lists_columns: set[str], table: EmbeddedXlTable
) -> EmbeddedXlTable:
    """Expand entries with commas in lists_columns into separate entries in the same column. Do this for
    all tables; keep entries in query_columns as lists.

    Parameters
    ----------
    query_columns
        List of query column names.
    lists_columns
        List of columns that may contain comma-separated lists.
    table
        Table in EmbeddedXlTable format.

    Returns
    -------
    EmbeddedXlTable
        Table in EmbeddedXlTable format with expanded comma entries.
    """
    # Exclude columns that have patterns
    exclude_cols = set(process_map.keys()).union(set(commodity_map.keys()))
    lists_columns = lists_columns.difference(exclude_cols)
    df = table.dataframe.copy()
    c = df.map(_has_comma)
    cols_to_make_lists = [
        colname
        for colname in c.columns
        if colname in lists_columns and c[colname].any()
    ]
    cols_to_explode = [
        colname for colname in cols_to_make_lists if colname not in query_columns
    ]

    if len(cols_to_make_lists) > 0:
        # Transform comma-separated strings into lists
        df[cols_to_make_lists] = df[cols_to_make_lists].map(_split_by_commas)
        if len(cols_to_explode) > 0:
            for colname in cols_to_explode:
                # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html#pandas.DataFrame.explode
                df = df.explode(colname, ignore_index=True)
    return replace(table, dataframe=df)


def remove_invalid_values(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Remove all entries of any dataframes that are considered invalid. The rules for
    allowing an entry can be seen in the 'constraints' dictionary below.

    Parameters
    ----------
    config

    tables
        List of tables in EmbeddedXlTable format.
    model


    Returns
    -------
    list[EmbeddedXlTable]
        List of tables in EmbeddedXlTable format with disallowed entries removed.
    """
    # TODO: This should be table type specific
    # TODO pull this out
    # Rules for allowing entries. Each entry of the dictionary designates a rule for a
    # a given column, and the values that are allowed for that column.
    constraints = {
        "csets": csets_ordered_for_pcg,
        "region": model.internal_regions,
    }

    # TODO: FI_T and UC_T should take into account whether a specific dimension is required
    skip_tags = {
        Tag.tfm_ava,
        Tag.tfm_mig,
        Tag.tfm_upd,
        Tag.uc_t,
    }

    def remove_table_invalid_values(
        table: EmbeddedXlTable,
    ) -> EmbeddedXlTable:
        """Remove invalid entries in a table dataframe."""
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
        table = replace(table, dataframe=df)
        return table

    return [
        remove_table_invalid_values(t) if t.tag not in skip_tags else t for t in tables
    ]


def process_units(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    units_map = {
        "activity": model.processes["tact"].unique(),
        "capacity": model.processes["tcap"].unique(),
        "commodity": model.commodities["unit"].unique(),
        "currency": tables[Tag.currencies]["currency"].unique(),
    }

    model.units = pd.concat(
        [pd.DataFrame({"unit": v, "type": k}) for k, v in units_map.items()]
    )

    return tables


def process_time_periods(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    model.start_year = utils.get_scalar(Tag.start_year, tables)
    active_pdef = utils.get_scalar(Tag.active_p_def, tables)
    df = utils.single_table(tables, Tag.time_periods).dataframe.copy()

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
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Read model regions and update model.internal_regions and model.all_regions.

    Include IMPEXP and MINRNW in model.all_regions (defined by default by Veda).
    """
    model.all_regions.update(["IMPEXP", "MINRNW"])
    # Read region settings
    region_def = utils.single_table(tables, Tag.book_regions_map).dataframe
    # Harmonise the dataframe
    region_def = (
        region_def.dropna(how="any")
        .apply(lambda x: x.str.upper())
        .drop_duplicates(ignore_index=True)
    )
    # Update model.all_regions
    model.all_regions.update(region_def["region"])
    # Determine model.internal_regions
    booknames = set(region_def["bookname"])
    valid_booknames = {
        b
        for b in booknames
        if any(re.match(rf"^VT_{b}_", file, re.IGNORECASE) for file in model.files)
    }
    model.internal_regions.update(
        region_def["region"][region_def["bookname"].isin(valid_booknames)]
    )

    # Print a warning for any region treated as external
    for bookname in booknames.difference(valid_booknames):
        external = region_def["region"][region_def["bookname"] == bookname].to_list()
        logger.warning(
            f"VT_{bookname}_* is not in model files. Treated {external} as external regions."
        )

    # Apply regions filter
    if config.filter_regions:
        keep_regions = model.internal_regions.intersection(config.filter_regions)
        if keep_regions:
            model.internal_regions = keep_regions
        else:
            logger.warning("Regions filter not applied; no valid entries found.")

    return tables


def complete_dictionary(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
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
        "Processes": model.processes,
        "Topology": model.topology,
        "Trade": model.trade,
        "TimePeriods": model.time_periods,
        "TimeSlices": model.ts_tslvl,
        "TimeSliceMap": model.ts_map,
        "UserConstraints": model.user_constraints,
        "UCAttributes": model.uc_attributes,
        "Units": model.units,
    }.items():
        if not v.empty:
            tables[k] = v

    return tables


def capitalise_table_values(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Ensure that all table entries are uppercase. Strip leading and trailing whitespace."""

    def capitalise_table_entries(table: EmbeddedXlTable):
        df = table.dataframe
        # Capitalise all entries if column type string
        colnames = df.select_dtypes(include="object").columns
        seen_cols = [colname for colname in colnames if colname in df.columns]
        if len(df) > 0:
            for seen_col in seen_cols:
                # Index of rows with string entries
                i = df[seen_col].apply(lambda x: isinstance(x, str))
                if any(i):
                    df.loc[i, seen_col] = df[seen_col][i].str.upper().str.strip()
            return replace(table, dataframe=df)
        else:
            return table

    return [capitalise_table_entries(table) for table in tables]


def _populate_defaults(
    tag: Tag,
    dataframe: DataFrame,
    col_name: str,
    config: Config,
    attr_col_name: str = "attribute",
):
    """Fill in some of the missing values based on defaults in place."""
    i_relevant_attributes = dataframe[attr_col_name].isin(
        config.veda_attr_defaults[col_name].keys()
    )
    if col_name in dataframe.columns:
        starting_na = i_relevant_attributes & dataframe[col_name].isna()
    else:
        # Add the column if it does not exist, but is needed
        if any(i_relevant_attributes):
            starting_na = i_relevant_attributes
            dataframe[col_name] = pd.NA
        else:
            starting_na = pd.Series(False, index=dataframe.index)

    if any(starting_na):
        attributes = dataframe[starting_na][attr_col_name].unique()
        for attr in attributes:
            i_attr = dataframe[attr_col_name] == attr
            default_values = config.veda_attr_defaults[col_name][attr.upper()]
            for default_value in default_values:
                # Ensure that previously filled values are not overwritten
                current_na = dataframe[col_name].isna()
                remaining_na = starting_na & i_attr & current_na
                if any(remaining_na):
                    if default_value not in config.known_columns[tag]:
                        dataframe.loc[remaining_na, [col_name]] = default_value
                    elif default_value in dataframe.columns:
                        dataframe.loc[remaining_na, [col_name]] = dataframe[
                            remaining_na
                        ][default_value]


def _populate_calculated_defaults(df: DataFrame, model: TimesModel):
    """Determine values of and fill in some indexes."""
    if any(df["cg"] == "veda_cg"):
        i = df["cg"] == "veda_cg"
        df.loc[i, "cg"] = df[i].apply(
            lambda x: model.veda_cgs.get((x["region"], x["process"], x["commodity"])),
            axis=1,
        )


def apply_fixups(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    def apply_fixups_table(table: EmbeddedXlTable):
        tags = [Tag.fi_t, Tag.uc_t]
        if table.tag not in tags:
            return table

        df = table.dataframe.copy()

        # TODO: should we have a global list of column name -> type?
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Populate commodity and other_indexes based on defaults
        for col in ("commodity", "other_indexes", "cg"):
            _populate_defaults(table.tag, df, col, config)

        return replace(table, dataframe=df)

    return [apply_fixups_table(table) for table in tables]


def generate_commodity_groups(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Generate commodity groups."""
    process_tables = [t for t in tables if t.tag == Tag.fi_process]

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
    comm_set = model.commodities[columns].copy()
    comm_set.drop_duplicates(keep="first", inplace=True)

    # Commodity groups by process, region and commodity
    comm_groups = pd.merge(model.topology, comm_set, on=["region", "commodity"])
    comm_groups.drop_duplicates(keep="last", inplace=True)

    # Add columns for the number of IN/OUT commodities of each type
    _count_comm_group_vectorised(comm_groups)

    def name_comm_group(df: pd.Series) -> str | None:
        """Generate the name of a commodity group based on the member count."""
        if df["commoditygroup"] > 1:
            return df["process"] + "_" + df["csets"] + df["io"][:1]
        elif df["commoditygroup"] == 1:
            return df["commodity"]
        else:
            return None

    # Replace commodity group member count with the name
    comm_groups["commoditygroup"] = comm_groups.apply(name_comm_group, axis=1)

    # Determine default PCG according to Veda's logic
    comm_groups = _process_comm_groups_vectorised(comm_groups, csets_ordered_for_pcg)

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
            keep="last",
            inplace=True,
            ignore_index=True,
        )

    # TODO: Include info from ~TFM_TOPINS e.g. include RSDAHT2 in addition to RSDAHT

    model.topology = comm_groups

    return tables


def _count_comm_group_vectorised(comm_groups: DataFrame) -> None:
    """Store the number of IN/OUT commodities of the same type per Region and Process in
    CommodityGroup. `comm_groups` is modified in-place.

    Parameters
    ----------
    comm_groups
        'Process' DataFrame with additional columns "commoditygroup"
    """
    cols_to_groupby = comm_groups.columns.difference(["commodity"])
    comm_groups["commoditygroup"] = 0

    comm_groups["commoditygroup"] = (
        comm_groups.groupby(list(cols_to_groupby)).transform("count")
    )["commoditygroup"]
    # set commodity group to 0 for io rows that aren't IN or OUT
    comm_groups.loc[~comm_groups["io"].isin(["IN", "OUT"]), "commoditygroup"] = 0


def _process_comm_groups_vectorised(
    comm_groups: DataFrame, csets_ordered_for_pcg: list[str]
) -> DataFrame:
    """Sets the first commodity group in the list of csets_ordered_for_pcg as the
    default pcg for each region/process/io combination, but setting the io="OUT" subset
    as default before "IN".

    See:
        Section 3.7.2.2, pg 80. of `TIMES Documentation PART IV` for details.

    Parameters
    ----------
    comm_groups
        'Process' DataFrame with columns `["region", "process", "io", "csets", "commoditygroup"]`
    csets_ordered_for_pcg
        List of csets in the order they should be considered for default pcg


    Returns
    -------
        Processed DataFrame with a new column "DefaultVedaPCG" set to True for the default pcg in
        each region/process/io combination.
    """

    def _set_default_veda_pcg(group):
        """For a given [region, process] group, default group is set as the first cset
        in the `csets_ordered_for_pcg` list, which is an output, if one exists,
        otherwise the first input.
        """
        if not group["csets"].isin(csets_ordered_for_pcg).all():
            return group

        for io in ["OUT", "IN"]:
            for cset in csets_ordered_for_pcg:
                group.loc[
                    (group["io"] == io) & (group["csets"] == cset), "DefaultVedaPCG"
                ] = True
                if group["DefaultVedaPCG"].any():
                    break
        return group

    comm_groups["DefaultVedaPCG"] = None
    comm_groups_subset = comm_groups.groupby(
        ["region", "process"], sort=False, as_index=False
    ).apply(_set_default_veda_pcg)
    comm_groups_subset = comm_groups_subset.reset_index(
        level=0, drop=True
    ).sort_index()  # back to the original index and row order
    return comm_groups_subset


def complete_commodity_groups(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Complete the list of commodity groups."""
    # Single member CGs i.e., CG and commodity are the same
    single_cgs = model.commodities[["region", "commodity"]].drop_duplicates(
        ignore_index=True
    )
    single_cgs["commoditygroup"] = single_cgs["commodity"]
    # Commodity groups from topology
    top_cgs = model.topology[["region", "commodity", "commoditygroup"]].drop_duplicates(
        ignore_index=True
    )
    cgs = pd.concat([single_cgs, top_cgs], ignore_index=True)
    cgs["gmap"] = cgs["commoditygroup"] != cgs["commodity"]
    model.commodity_groups = cgs.dropna().drop_duplicates(ignore_index=True)

    return tables


def generate_trade(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Generate inter-regional exchange topology."""
    veda_set_ext_reg_mapping = {"IMP": "IMPEXP", "EXP": "IMPEXP", "MIN": "MINRNW"}
    veda_ire_sets = model.custom_sets

    ire_prc = pd.DataFrame(columns=["region", "process"])
    for table in tables:
        if table.tag == Tag.fi_process:
            df = table.dataframe
            ire_prc = pd.concat(
                [ire_prc, df.loc[df["sets"] == "IRE", ["region", "process"]]]
            )
    ire_prc.drop_duplicates(keep="first", inplace=True)

    # Generate inter-regional exchange topology
    top_ire = model.topology[
        ["region", "process", "commodity", "io"] + list(dm_cols)
    ].copy()
    if config.include_dummy_imports:
        dummy_process_cset = [
            ["NRG", "IMPNRGZ"],
            ["MAT", "IMPMATZ"],
            ["DEM", "IMPDEMZ"],
        ]
        dummy_ire = pd.DataFrame(dummy_process_cset, columns=["csets", "process"])
        dummy_ire = dummy_ire.merge(
            pd.DataFrame(model.internal_regions, columns=["region"]), how="cross"
        )
        dummy_ire = dummy_ire.merge(model.topology[["region", "csets", "commodity"]])
        dummy_ire.drop(columns=["csets"], inplace=True)
        dummy_ire["io"] = "OUT"
        top_ire = pd.concat([top_ire, dummy_ire])

    top_ire = top_ire.merge(ire_prc)
    top_ire = top_ire.merge(veda_ire_sets)
    top_ire["region2"] = top_ire["sets"].replace(veda_set_ext_reg_mapping)
    top_ire[["origin", "destination", "in", "out"]] = None
    for io in ("IN", "OUT"):
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
    cols_list.extend(dm_cols)
    # Include trade between internal regions
    for table in tables:
        if table.tag == Tag.tradelinks_dins:
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
    # Discard tradelinks if none of the regions is internal
    i = top_ire["origin"].isin(model.internal_regions) | top_ire["destination"].isin(
        model.internal_regions
    )
    model.trade = top_ire[i].reset_index(drop=True)

    return tables


def fill_in_missing_pcgs(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Fill in missing primary commodity groups in FI_Process tables.

    Expand primary commodity groups specified in FI_Process tables by a suffix.
    """
    result = []

    for table in tables:
        if table.tag != Tag.fi_process:
            result.append(table)
        else:
            df = table.dataframe.copy()
            # Expand primary commodity groups specified in primarycg column by a suffix
            i = df["primarycg"].isin(default_pcg_suffixes) & df["process"].notna()
            if any(i):
                # Specify primary commodity group based on suffix and the process name.
                df.loc[i, "primarycg"] = df["process"][i] + "_" + df["primarycg"][i]
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
            # Keep last if a row appears more than once (disregard primarycg)
            df.drop_duplicates(
                subset=[c for c in df.columns if c != "primarycg"],
                keep="last",
                inplace=True,
            )

            result.append(replace(table, dataframe=df))

    return result


def remove_fill_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    # These tables collect data from elsewhere and update the table itself or a region below
    # The collected data is then presumably consumed via Excel references or vlookups
    # TODO: For the moment, assume that these tables are up-to-date. We will need a tool to do this.
    result = []
    for table in tables:
        if table.tag != Tag.tfm_fill and not table.tag == Tag.tfm_fill_r:
            result.append(table)
    return result


def convert_com_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Transform comemi and comagg tables to fi_t."""
    convert_tags = {
        Tag.comemi: {
            "attribute": "vda_emcb",
            "index_column": "commodity",
            "other_column": "other_indexes",
        },
        Tag.comagg: {
            "attribute": "com_agg",
            "index_column": "other_indexes",
            "other_column": "commodity",
        },
    }
    result = []
    for table in tables:
        if table.tag not in convert_tags:
            result.append(table)
        else:
            info = convert_tags[table.tag]
            index_column = info["index_column"]
            other_column = info["other_column"]
            df = table.dataframe.copy()
            # Remove columns that are not allowed
            # TODO: Base this on the config file instead
            remove_cols = ["region", "year"]
            df.drop(columns=remove_cols, errors="ignore", inplace=True)
            data_columns = [
                colname for colname in df.columns if colname != index_column
            ]
            df, names = utils.explode(df, data_columns)
            df.rename(columns={"value": info["attribute"]}, inplace=True)
            df[other_column] = names
            df[other_column] = df[other_column].str.upper()

            df = df.reset_index(drop=True)
            result.append(replace(table, dataframe=df, tag=Tag.fi_t))

    return result


def internalise_commodities(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Populate model.commodities."""
    result = []
    comm_dfs = []
    for table in tables:
        if table.tag != Tag.fi_comm:
            result.append(table)
        else:
            comm_dfs.append(table.dataframe)

    model.commodities = pd.concat(comm_dfs, ignore_index=True)

    return result


def process_processes(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Process processes."""
    result = []
    veda_sets_to_times = {"IMP": "IRE", "EXP": "IRE", "MIN": "IRE"}
    original_dfs = []

    for table in tables:
        if table.tag != Tag.fi_process:
            result.append(table)
        else:
            original_dfs.append(table.dataframe)
            df = table.dataframe.copy()
            df.replace({"sets": veda_sets_to_times}, inplace=True)
            result.append(replace(table, dataframe=df))

    merged_tables = pd.concat(original_dfs, ignore_index=True)
    i = merged_tables["sets"].isin(veda_sets_to_times.keys())
    model.custom_sets = merged_tables[["sets", "process"]][i].drop_duplicates(
        ignore_index=True
    )

    return result


def process_topology(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Create model.topology. Drop rows with missing values in fi_t tables."""
    fit_tables = [t for t in tables if t.tag == Tag.fi_t]

    columns = {
        "region",
        "process",
        "commodity-in",
        "commodity-in-aux",
        "commodity-out",
        "commodity-out-aux",
    }.union(dm_cols)
    topology = pd.DataFrame(columns=list(columns))

    top_info = []

    for fit_table in fit_tables:
        df = fit_table.dataframe
        cols = [col for col in columns.intersection(df.columns)]
        top_info.append(df[cols].copy())
        # Rows with missing values in fi_t tables can now safely be dropped.
        df.dropna(subset=["value"], axis=0, ignore_index=True, inplace=True)

    topology = pd.concat(top_info, ignore_index=True)

    topology = pd.melt(
        topology,
        id_vars=["region", "process"] + list(dm_cols),
        var_name="io",
        value_name="commodity",
    )

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
    # Populate region column if missing
    topology.fillna({"region": ",".join(model.internal_regions)}, inplace=True)
    # Check if dataframe contains entries with commas
    df = topology.map(_has_comma)
    cols_with_comma = [col for col in df.columns if df[col].any()]
    if cols_with_comma:
        # Convert entries with commas to lists
        topology = topology.map(_split_by_commas)
        # Explode lists to rows
        for col in cols_with_comma:
            topology = topology.explode(col, ignore_index=True)
    model.topology = topology

    return tables


def generate_dummy_processes(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Define dummy processes and specify default cost data for them to ensure that a
    TIMES model can always be solved.

    This covers situations when a commodity cannot be supplied by other means.
    Significant cost is usually associated with the activity of these processes to
    ensure that they are used as a last resort
    """
    if config.include_dummy_imports:
        # TODO: Activity units below are arbitrary. Suggest Veda devs not to have any.
        dummy_processes = [
            ["IMPNRGZ", "Dummy Import of NRG", "PJ", "NRG"],
            ["IMPMATZ", "Dummy Import of MAT", "MT", "MAT"],
            ["IMPDEMZ", "Dummy Import of DEM", "PJ", "DEM"],
        ]

        process_declarations = pd.DataFrame(
            dummy_processes,
            columns=["process", "description", "tact", "primarycg"],
        )

        # Data that is the same for all dummy processes
        additional_cols = {
            "region": pd.NA,
            "sets": "IMP",
            "tcap": pd.NA,
            "tslvl": "ANNUAL",
            "vintage": pd.NA,
        }

        for col, value in additional_cols.items():
            process_declarations[col] = value

        tables.append(
            EmbeddedXlTable(
                tag="~FI_PROCESS",
                uc_sets={},
                sheetname="",
                range="",
                filename="",
                dataframe=process_declarations,
            )
        )

        process_data_specs = process_declarations[["process", "description"]].copy()
        # Provide an empty value in case an upd table is used to provide data
        process_data_specs["ACTCOST"] = ""

        tables.append(
            EmbeddedXlTable(
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
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Transform tradelinks to tradelinks_dins."""
    result = []
    for table in tables:
        if table.tag == Tag.tradelinks:
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
                lambda row: (
                    tuple(sorted([row["origin"], row["destination"]]))
                    if row["tradelink"] == "b"
                    else tuple([row["origin"], row["destination"]])
                ),
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
            result.append(replace(table, dataframe=df, tag=Tag.tradelinks_dins))
        else:
            result.append(table)

    return result


def process_transform_table_variants(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Reduces variants of TFM_INS like TFM_INS-TS to TFM_INS."""

    def has_no_wildcards(list):
        return all(
            list.apply(
                lambda x: x is not None
                and x[0] != "-"
                and "*" not in x
                and "," not in x
                and "?" not in x
                and "_" not in x
            )
        )

    def is_year(col_name):
        """A column name is a year if it is an int >= 0."""
        return col_name.isdigit() and int(col_name) >= 0

    result = []
    for table in tables:
        tag = Tag(table.tag)
        if tag in [
            Tag.tfm_dins_ts,
            Tag.tfm_ins_ts,
            Tag.tfm_upd_ts,
        ]:
            # ~TFM_INS-TS: Gather columns whose names are years into a single "Year" column:
            df = table.dataframe

            if "year" in df.columns:
                raise ValueError(f"TFM_INS-TS table already has Year column: {table}")

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
            result.append(
                replace(table, dataframe=df, tag=Tag(tag.value.split("-")[0]))
            )
        elif tag in [
            Tag.tfm_dins_at,
            Tag.tfm_ins_at,
            Tag.tfm_upd_at,
        ]:
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
            # Convert the attribute column to uppercase
            df["attribute"] = df["attribute"].str.upper()
            result.append(
                replace(table, dataframe=df, tag=Tag(tag.value.split("-")[0]))
            )
        else:
            result.append(table)

    return result


def process_transform_tables(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Process transform tables."""
    regions = model.internal_regions
    # TODO: Add other tfm tags?
    tfm_tags = [
        Tag.tfm_ava,
        Tag.tfm_dins,
        Tag.tfm_ins,
        Tag.tfm_ins_txt,
        Tag.tfm_topins,
        Tag.tfm_upd,
        Tag.tfm_mig,
        Tag.tfm_comgrp,
    ]

    result = []
    dropped = []
    for table in tables:
        tag = Tag(table.tag)

        if tag not in tfm_tags:
            result.append(table)

        elif tag in tfm_tags and tag != Tag.tfm_topins:
            df = table.dataframe.copy()

            # Standardize column names
            known_columns = config.known_columns[tag]

            # Handle Regions:
            # Check whether any of model regions are among columns
            if set(df.columns).isdisjoint({x.lower() for x in regions}):
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
                # In the absence of the "region" column values in the "value" column apply to all regions
                if "value" in df.columns:
                    df = df.rename(columns={"value": "allregions"})
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
            logger.warning(
                f"Dropped {len(group)} transform tables ({key}) rather than processing them"
            )

    return result


def process_transform_availability(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    """Process transform availability tables.
    Steps include:
    - Removing rows with missing values in the "value" column.
    """
    result = []
    for table in tables:
        if table.tag == Tag.tfm_ava:
            result.append(
                replace(table, dataframe=table.dataframe.dropna(subset="value"))
            )
        else:
            result.append(table)

    return result


def filter_by_pattern(df: DataFrame, pattern: str) -> set[str]:
    """Filter dataframe index by a pattern specifying which items to include and/or exclude.
    Return a set of corresponding items from the first (and only) column in the dataframe.
    """
    map = {"include": utils.create_regexp, "exclude": utils.create_negative_regexp}
    sets = dict()
    for action, regex_maker in map.items():
        sets[action] = set(
            df.filter(regex=regex_maker(pattern), axis="index").iloc[:, 0]
        )

    return sets["include"].difference(sets["exclude"])


def get_matching_items(
    row: pd.Series, topology: dict[str, DataFrame], item_map: dict[str, str]
) -> list[str] | None:
    """Return a list of items that match conditions in the given row."""
    matching_items = None
    for col, key in item_map.items():
        if col in row.index and pd.notna(row[col]):
            item_set = topology[key]
            pattern = row[col].upper()
            filtered = filter_by_pattern(item_set, pattern)
            matching_items = (
                matching_items.intersection(filtered)
                if matching_items is not None
                else filtered
            )

    return list(matching_items) if matching_items is not None else None


def df_indexed_by_col(df: DataFrame, col: str) -> DataFrame:
    """Set df index using an existing column; make index uppercase."""
    df = df.dropna().drop_duplicates()
    index = df[col].str.upper()
    df = df.set_index(index).rename_axis("index")

    if len(df.columns) > 1:
        df = df.drop(columns=col)
    return df


def generate_topology_dictionary(
    tables: dict[str, DataFrame], model: TimesModel
) -> dict[str, DataFrame]:
    # We need to be able to fetch processes based on any combination of name, description, set, comm-in, or comm-out
    # So we construct tables whose indices are names, etc. and use pd.filter

    dictionary = dict()
    pros = model.processes
    coms = model.commodities
    pros_sets = pd.concat(
        [pros[["process", "sets"]].drop_duplicates(), model.custom_sets],
        ignore_index=True,
    )
    pros_and_coms = model.topology[["process", "commodity", "io"]].drop_duplicates()
    i_comm_in = pros_and_coms["io"] == "IN"
    i_comm_out = pros_and_coms["io"] == "OUT"

    dict_info = [
        {"key": "processes_by_name", "df": pros[["process"]], "col": "process"},
        {
            "key": "processes_by_desc",
            "df": pros[["process", "description"]],
            "col": "description",
        },
        {"key": "processes_by_sets", "df": pros_sets, "col": "sets"},
        {
            "key": "processes_by_comm_in",
            "df": pros_and_coms[["process", "commodity"]][i_comm_in],
            "col": "commodity",
        },
        {
            "key": "processes_by_comm_out",
            "df": pros_and_coms[["process", "commodity"]][i_comm_out],
            "col": "commodity",
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


def process_wildcards(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Process wildcards in the tables."""
    tags = [
        Tag.tfm_ava,
        Tag.tfm_comgrp,
        Tag.tfm_ins,
        Tag.tfm_ins_txt,
        Tag.tfm_mig,
        Tag.tfm_upd,
        Tag.uc_t,
    ]
    dictionary = generate_topology_dictionary(tables, model)
    item_maps = {
        "process": process_map,
        "commodity": commodity_map,
    }

    for tag in tags:
        if tag in tqdm(tables, desc=f"Processing wildcards in {tag.value} tables"):

            start_time = time.time()
            df = tables[tag]

            for item_type in ["process", "commodity"]:
                item_map = item_maps[item_type]
                if set(df.columns).intersection(set(item_map.keys())):
                    df = _match_wildcards(
                        df,
                        item_map,
                        dictionary,
                        item_type,
                        explode=False,
                    )

            tables[tag] = df

            # TODO: Should the tool alert about the following?
            # logger.warning("a row matched no processes or commodities")
            logger.info(
                f"  process_wildcards: {tag} took {time.time() - start_time:.2f} seconds for {len(df)} rows"
            )
    return tables


def _match_wildcards(
    df: DataFrame,
    col_map: dict[str, str],
    dictionary: dict[str, DataFrame],
    result_col: str,
    explode: bool = False,
) -> DataFrame:
    """Match wildcards in the given table using the given process map and dictionary.

    Parameters
    ----------
    df
        Table to match wildcards in.
    col_map
        Mapping of column names to sets.
    dictionary
        Dictionary of process sets to match against.
    result_col
        Name of the column to store the matched results in.
    explode
        Whether to explode the  results_col ('process'/'commodities') column into a long-format table.
        (Default value = False)

    Returns
    -------
        The table with the wildcard columns removed and the results of the wildcard matches added as a
        column named `results_col`
    """
    wild_cols = list(col_map.keys())

    # drop duplicate sets of wildcard columns to save repeated (slow) regex matching.  This makes things much faster.
    unique_filters = df[wild_cols].drop_duplicates().dropna(axis=0, how="all")

    # match all the wildcards columns against the dictionary names
    unique_filters[result_col] = unique_filters.apply(
        lambda row: get_matching_items(row, dictionary, col_map), axis=1
    )

    # Finally we merge the matches back into the original table.
    # This join re-duplicates the duplicate filters dropped above for speed.
    df = (
        df.merge(unique_filters, on=wild_cols, how="left", suffixes=("_old", ""))
        .reset_index(drop=True)
        .drop(columns=wild_cols)
    )

    # Pre-existing 'process' and 'commodity' are handled during renaming.
    # The below should not be necessary, but is left just in case.
    if f"{result_col}_old" in df.columns:
        if not df[f"{result_col}_old"].isna().all():
            logger.warning(
                f"Non-empty existing '{result_col}' column will be overwritten!"
            )
        df = df.drop(columns=[f"{result_col}_old"])

    # And we explode any matches to multiple names to give a long-format table.
    if explode:
        if result_col in df.columns:
            df = df.explode(result_col, ignore_index=True)
        else:
            df[result_col] = None

    # replace NaNs in results_col with None (expected downstream)
    if df[result_col].dtype != object:
        df[result_col] = df[result_col].astype(object)

    # replace NaNs in results_col with None (expected downstream)
    df.loc[df[result_col].isna(), [result_col]] = None

    return df


def query(
    table: DataFrame,
    process: str | list[str] | None,
    commodity: str | list[str] | None,
    attribute: str | None,
    region: str | list[str] | None,
    year: int | list | None,
    limtype: str | list[str] | None,
    val: int | float | None,
    module: str | list[str] | None,
) -> pd.Index:

    query_fields = {
        "process": process,
        "commodity": commodity,
        "attribute": attribute,
        "region": region,
        "year": year,
        "limtype": limtype,
        "value": val,
        "module_name": module,
    }

    def is_missing(field):
        return pd.isna(field) if not isinstance(field, list) else False

    qs = [
        f"{k} in {v if isinstance(v, list) else [v]}"
        for k, v in query_fields.items()
        if not is_missing(v)
    ]

    query_str = " and ".join(qs)
    row_idx = table.query(query_str).index
    return row_idx


def eval_and_update(table: DataFrame, rows_to_update: pd.Index, new_value: str) -> None:
    """Performs an inplace update of rows `rows_to_update` of `table` with `new_value`,
    which can be a update formula like `*2.3`.
    """
    if isinstance(new_value, str) and new_value[0] in {"*", "+", "-", "/"}:
        # Do not perform arithmetic operations on rows with i/e options
        if "year" in table.columns:
            rows_to_update = rows_to_update.intersection(
                table.index[table["year"] != 0]
            )
        old_values = table.loc[rows_to_update, "value"]
        updated = old_values.astype(float).map(lambda x: eval("x" + new_value))
        table.loc[rows_to_update, "value"] = updated
    else:
        table.loc[rows_to_update, "value"] = new_value


def _remove_invalid_rows(
    df: DataFrame,
    valid_combinations: DataFrame,
    verify_cols: list[str],
    limit_to: dict[str, set] | None = None,
    include_na_dimensions: bool = False,
) -> DataFrame:
    """Remove rows with invalid process / region combination."""
    df = df.copy()
    # Limit verification to specific dimensions
    allow_to_verify = {"process", "commodity", "region", "module_name"}
    wont_verify = set(verify_cols).difference(allow_to_verify)
    verify_cols = list(allow_to_verify.intersection(verify_cols))
    if wont_verify:
        logger.warning(
            f"Verification of {wont_verify} is not supported. Only {verify_cols} will be verified."
        )
    # Index of rows that won't be checked
    keep = pd.RangeIndex(0)
    if limit_to:
        for col, values in limit_to.items():
            if col in df.columns:
                keep = keep.union(df[~df[col].isin(values)].index)
            else:
                logger.warning(f"Column {col} not found in the dataframe.")
    # Don't check rows with empty dimensions
    if not include_na_dimensions:
        keep = keep.union(df[df[verify_cols].isna().any(axis=1)].index)
    # Ensure that valid_combinations has a specific set of columns
    for col in allow_to_verify:
        if col not in valid_combinations.columns:
            valid_combinations[col] = pd.NA
    # Keep only valid combinations of dimensions
    for _, row in valid_combinations.iterrows():
        keep = keep.union(
            query(
                df,
                row["process"],
                row["commodity"],
                None,
                row["region"],
                None,
                None,
                None,
                row["module_name"],
            )
        )

    return df.loc[keep]


def apply_transform_tables(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Include data from transformation tables (excl. availability)."""
    # Create a dictionary of processes/commodities indexed by module name
    obj_by_module = dict()
    obj_by_module["process"] = (
        model.processes.groupby("module_name")["process"].agg(set).to_dict()
    )
    obj_by_module["commodity"] = (
        model.commodities.groupby("module_name")["commodity"].agg(set).to_dict()
    )
    # Create a dictionary of processes/commodities available in addtion to those declared in a module
    obj_suppl = dict()
    obj_suppl["process"] = set()
    obj_suppl["commodity"] = (
        obj_by_module["commodity"]
        .get("BASE", set())
        .union(obj_by_module["commodity"].get("SYSSETTINGS", set()))
    )
    # Create sets attributes that require a process/commodity index
    attr_with_obj = {
        obj: config.attr_by_type[obj].union(config.attr_by_type["flow"])
        for obj in ["process", "commodity"]
    }
    # Create a dictionary of valid region/process and region/commodity combinations
    obj_region = dict()
    obj_region["process"] = model.processes[["region", "process"]].drop_duplicates()
    obj_region["commodity"] = model.commodities[
        ["region", "commodity"]
    ].drop_duplicates()

    if Tag.tfm_comgrp in tables:
        table = model.commodity_groups
        updates = tables[Tag.tfm_comgrp].filter(table.columns, axis=1)
        commodity_groups = pd.concat([table, updates], ignore_index=True)
        commodity_groups = commodity_groups.explode("commodity", ignore_index=True)
        commodity_groups = commodity_groups.drop_duplicates()
        commodity_groups.loc[commodity_groups["gmap"].isna(), ["gmap"]] = True
        model.commodity_groups = commodity_groups.dropna()

    for data_module in model.data_modules:
        generated_records = []
        if (
            Tag.tfm_dins in tables
            and data_module in tables[Tag.tfm_dins]["module_name"].unique()
        ):
            table = tables[Tag.fi_t]
            index = tables[Tag.tfm_dins]["module_name"] == data_module
            updates = tables[Tag.tfm_dins][index].filter(table.columns, axis=1)
            generated_records.append(updates)
        if (
            Tag.tfm_ins in tables
            and data_module in tables[Tag.tfm_ins]["module_name"].unique()
        ):
            table = tables[Tag.fi_t]
            index = tables[Tag.tfm_ins]["module_name"] == data_module
            updates = tables[Tag.tfm_ins][index].filter(table.columns, axis=1)
            generated_records.append(updates)

        if (
            Tag.tfm_ins_txt in tables
            and data_module in tables[Tag.tfm_ins_txt]["module_name"].unique()
        ):
            index = tables[Tag.tfm_ins_txt]["module_name"] == data_module
            updates = tables[Tag.tfm_ins_txt][index]

            # TFM_INS-TXT: expand row by wildcards, query FI_PROC/COMM for matching rows,
            # evaluate the update formula, and inplace update the rows
            for _, row in tqdm(
                updates.iterrows(),
                total=len(updates),
                desc=f"Applying transformations from {Tag.tfm_ins_txt.value} in {data_module}",
            ):
                if row["commodity"] is not None:
                    table = model.commodities
                elif row["process"] is not None:
                    table = model.processes
                else:
                    assert False  # All rows match either a commodity or a process

                # Query for rows with matching process/commodity and region
                rows_to_update = query(
                    table,
                    row["process"],
                    row["commodity"],
                    None,
                    row["region"],
                    None,
                    None,
                    None,
                    None,
                )
                # Overwrite (inplace) the column given by the attribute (translated by attr_prop)
                # with the value from row
                # E.g. if row['attribute'] == 'PRC_TSL' then we overwrite 'tslvl'
                if row["attribute"] not in attr_prop:
                    logger.warning(
                        f"Unknown attribute {row['attribute']}, skipping update."
                    )
                else:
                    table.loc[rows_to_update, attr_prop[row["attribute"]]] = row[
                        "value"
                    ]

        if (
            Tag.tfm_upd in tables
            and data_module in tables[Tag.tfm_upd]["module_name"].unique()
        ):
            index = tables[Tag.tfm_upd]["module_name"] == data_module
            updates = tables[Tag.tfm_upd][index]
            table = tables[Tag.fi_t]
            new_tables = []

            # TFM_UPD: expand wildcards in each row, query FI_T to find matching rows,
            # evaluate the update formula, and add new rows to FI_T
            # TODO perf: collect all updates and go through FI_T only once?
            for _, row in tqdm(
                updates.iterrows(),
                total=len(updates),
                desc=f"Applying transformations from {Tag.tfm_upd.value} in {data_module}",
            ):
                if row["module_type"] == "trans":
                    source_module = row["module_name"]
                else:
                    source_module = row["sourcescen"]

                rows_to_update = query(
                    table,
                    row["process"],
                    row["commodity"],
                    row["attribute"],
                    row["region"],
                    row["year"],
                    row["limtype"],
                    row["val_cond"],
                    source_module,
                )

                if not any(rows_to_update):
                    logger.info(f"A {Tag.tfm_upd.value} row generated no records.")
                    continue

                new_rows = table.loc[rows_to_update].copy()
                eval_and_update(new_rows, rows_to_update, row["value"])
                # In case more than one data module is present in the table, select the one with the highest index.
                # TODO: The below code is commented out because it needs to be more sophisticated.
                """
                if new_rows["module_name"].nunique() > 1:
                    indices = {
                        model.data_modules.index(x)
                        for x in new_rows["module_name"].unique()
                    }
                    new_rows = new_rows[
                        new_rows["module_name"] == model.data_modules[max(indices)]
                    ]
                """
                new_rows["source_filename"] = row["source_filename"]
                new_rows["module_name"] = row["module_name"]
                new_rows["module_type"] = row["module_type"]
                new_rows["submodule"] = row["submodule"]
                new_tables.append(new_rows)

            if new_tables:
                generated_records.append(pd.concat(new_tables, ignore_index=True))

        if (
            Tag.tfm_mig in tables
            and data_module in tables[Tag.tfm_mig]["module_name"].unique()
        ):
            index = tables[Tag.tfm_mig]["module_name"] == data_module
            updates = tables[Tag.tfm_mig][index]
            table = tables[Tag.fi_t]
            new_tables = []

            for _, row in tqdm(
                updates.iterrows(),
                total=len(updates),
                desc=f"Applying transformations from {Tag.tfm_mig.value} in {data_module}",
            ):
                if row["module_type"] == "trans":
                    source_module = row["module_name"]
                else:
                    source_module = row["sourcescen"]

                rows_to_update = query(
                    table,
                    row["process"],
                    row["commodity"],
                    row["attribute"],
                    row["region"],
                    row["year"],
                    row["limtype"],
                    row["val_cond"],
                    source_module,
                )

                if not any(rows_to_update):
                    logger.warning(f"A {Tag.tfm_mig.value} row generated no records.")
                    continue

                new_rows = table.loc[rows_to_update].copy()
                # Modify values in all '*2' columns
                for c, v in row.items():
                    if str(c).endswith("2") and v is not None:
                        new_rows.loc[:, str(c)[:-1]] = v
                # Evaluate 'value' column based on existing values
                eval_and_update(new_rows, rows_to_update, row["value"])
                # In case more than one data module is present in the table, select the one with the highest index
                # TODO: The below code is commented out because it needs to be more sophisticated.
                """
                if new_rows["module_name"].nunique() > 1:
                    indices = {
                        model.data_modules.index(x)
                        for x in new_rows["module_name"].unique()
                    }
                    new_rows = new_rows[
                        new_rows["module_name"] == model.data_modules[max(indices)]
                    ]
                """
                new_rows["source_filename"] = row["source_filename"]
                new_rows["module_name"] = row["module_name"]
                new_rows["module_type"] = row["module_type"]
                new_rows["submodule"] = row["submodule"]
                new_tables.append(new_rows)

            if new_tables:
                generated_records.append(pd.concat(new_tables, ignore_index=True))

        if generated_records:
            module_data = pd.concat(generated_records, ignore_index=True)
            module_type = module_data["module_type"].iloc[0]
            # Explode process and commodity columns and remove invalid rows
            for obj in ["process", "commodity"]:
                module_data = module_data.explode(obj, ignore_index=True)
                # Index of rows with relevant attributes
                i = module_data["attribute"].isin(attr_with_obj[obj])
                # Create an index to control which rows to drop
                drop = pd.Series(False, index=module_data.index)
                # Exclude NA values, they may be populated later on
                i = i & module_data[obj].notna()
                if module_type in {"base", "subres", "trade"}:
                    valid_objs = (
                        obj_by_module[obj].get(data_module, set()).union(obj_suppl[obj])
                    )
                    # Rows with illegal process/commodity names in the module
                    drop = drop | (~module_data[obj].isin(valid_objs) & i)
                # Remove rows with invalid process/region and commodity/region combinations
                module_data = module_data.merge(
                    obj_region[obj], on=["region", obj], how="left", indicator=True
                )
                drop = drop | (i & (module_data["_merge"] == "left_only"))
                module_data = module_data[~drop].drop(columns="_merge")

            if not module_data.empty:
                tables[Tag.fi_t] = pd.concat(
                    [tables[Tag.fi_t], module_data], ignore_index=True
                )

    return tables


def explode_process_commodity_cols(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Explodes the process and commodity columns in the tables that contain them as
    lists after process_wildcards.

    We store wildcard matches for these columns as lists and explode them late here for performance
    reasons - to avoid row-wise processing that would otherwise need to iterate over very long tables.
    """
    for tag in tables:
        df = tables[tag]

        if "process" in df.columns:
            df = df.explode("process", ignore_index=True)

        if "commodity" in df.columns:
            df = df.explode("commodity", ignore_index=True)

        if "other_indexes" in df.columns:
            df = df.explode("other_indexes", ignore_index=True)

        tables[tag] = df

    return tables


def process_time_slices(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    def timeslices_table(
        table: EmbeddedXlTable,
        regions: list,
        result: list[EmbeddedXlTable],
    ):
        # User-specified timeslices (ordered)
        user_ts_levels = ["SEASON", "WEEKLY", "DAYNITE"]

        # Ensure that all timeslice levels are uppercase
        timeslices = {
            str(col).upper(): list(values.unique())
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
            if timeslices[ts_level]:
                reg_ts = reg_ts.merge(
                    pd.DataFrame({ts_level: timeslices[ts_level]}), how="cross"
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

        model.ts_map = DataFrame(ts_maps).dropna(ignore_index=True)
        model.ts_tslvl = DataFrame(ts_groups).dropna(ignore_index=True)

    result = []

    # TODO: Timeslices can differ from region to region
    regions = list(model.internal_regions)

    for table in tables:
        if table.tag != Tag.time_slices:
            result.append(table)
        else:
            timeslices_table(table, regions, result)

    return result


def convert_to_string(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    def convert(x: Any) -> str:
        if isinstance(x, float):
            if x.is_integer():
                return str(int(x))
            else:
                return f"{x:.10g}"
        return str(x)

    for key, value in tables.items():
        tables[key] = value.map(convert)
    return tables


def convert_aliases(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Ensure TIMES names for all attributes."""
    replacement_dict = {}
    for k, v in config.veda_attr_defaults["aliases"].items():
        for alias in v:
            replacement_dict[alias] = k

    for table_type, df in tables.items():
        if "attribute" in df.columns:
            df["original_attr"] = df["attribute"]
            df.replace({"attribute": replacement_dict}, inplace=True)
        tables[table_type] = df

    return tables


top_check_map = {
    "A": {"IN", "OUT"},
    "I": {"IN"},
    "O": {"OUT"},
}


def verify_uc_topology(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Verify if region / process / commodity in UC_T are present in the topology.
    Remove rows with invalid region/process or region/commodity combinations.
    """
    if Tag.uc_t not in tables:
        return tables

    df = tables[Tag.uc_t].copy()
    topology = pd.concat([model.topology, model.implied_topology], ignore_index=True)
    result = []
    items_in_region = {
        "process": model.processes.groupby(["region"], as_index=False).agg(
            {"process": list}
        )[["region", "process"]],
        "commodity": model.commodities.groupby(["region"], as_index=False).agg(
            {"commodity": list}
        )[["region", "commodity"]],
    }
    relevant_attrs = {
        "process": config.attr_by_type["process"].union(config.attr_by_type["flow"]),
        "commodity": config.attr_by_type["commodity"].union(
            config.attr_by_type["flow"]
        ),
    }
    # Explode process/commodity columns and remove any row with invalid region/process or region/commodity combination.
    for col in ["process", "commodity"]:
        df = df.explode(col, ignore_index=True)
        df = _remove_invalid_rows(
            df,
            items_in_region[col],
            ["region", col],
            limit_to={"attribute": relevant_attrs[col]},
        )
    # Proceed with topology verification
    cols = ["region", "process", "commodity"]
    requested_checks = df["top_check"].unique()
    i_verify_attrs = df["attribute"].isin(config.attr_by_type["flow"])
    checked = pd.Series(False, index=df.index)
    for check in requested_checks:
        i = (df["top_check"] == check) & i_verify_attrs
        if check in top_check_map:
            specific_topology = topology[cols][
                topology["io"].isin(top_check_map[check])
            ].drop_duplicates()
            result.append(df[i].merge(specific_topology, on=cols, how="inner"))
            checked = checked | i

    if result:
        if any(~checked):
            result.append(df[~checked])
        df = pd.concat(result).sort_index()

    tables[Tag.uc_t] = df

    return tables


def assign_model_attributes(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Assign model attributes to the model."""
    model.attributes = tables[Tag.fi_t]
    # Also ssign UC attributes if present
    if Tag.uc_t in tables.keys():
        df = tables[Tag.uc_t].copy()
        # Expand timeslice levels for UC attributes that require timeslices
        df_attrs = set(df["attribute"].unique())
        # Check if any UC attributes that require timeslices are present
        ts_uc_attrs = {
            attr.times_name
            for attr in config.times_xl_maps
            if attr.times_name in df_attrs and "TS" in attr.times_cols
        }
        if ts_uc_attrs:
            # Index of rows with UC attributes that require timeslices, but specify timeslice levels instead
            index = df["attribute"].isin(ts_uc_attrs) & ~df["timeslice"].isin(
                set(model.ts_tslvl["ts"].unique())
            )
            if any(index):
                # Create a list of timeslices for each region / timeslice level combination
                ts_list = (
                    model.ts_tslvl.groupby(["region", "tslvl"]).agg(set).reset_index()
                )
                df["tslvl"] = df["timeslice"]
                df = df.merge(ts_list, on=["region", "tslvl"], how="left")
                df.loc[index, "timeslice"] = df["ts"][index]
                df.drop(columns=["tslvl", "ts"], inplace=True)
                # Explode timeslice column
                df = df.explode("timeslice", ignore_index=True)
        model.uc_attributes = df

    return tables


def resolve_remaining_cgs(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Resolve commodity group names in model.attributes specified as commodity type.

    Supplement model.commodity_groups with resolved commodity groups.
    """
    if not model.attributes.empty:
        i = model.attributes["cg"].isin(default_pcg_suffixes)
        if any(i):
            # Store processes with unresolved commodity groups
            check_cgs = model.attributes.loc[
                i, ["region", "process", "cg"]
            ].drop_duplicates(ignore_index=True)
            # Resolve commodity group names in model.attribues
            model.attributes.loc[i, "cg"] = (
                model.attributes["process"].astype(str)
                + "_"
                + model.attributes["cg"].astype(str)
            )
            # TODO: Combine with above to avoid repetition
            check_cgs["commoditygroup"] = (
                check_cgs["process"].astype(str) + "_" + check_cgs["cg"].astype(str)
            )
            check_cgs["csets"] = check_cgs["cg"].str[:3]
            check_cgs["io"] = check_cgs["cg"].str[3:]
            check_cgs["io"] = check_cgs["io"].replace({"I": "IN", "O": "OUT"})
            check_cgs = check_cgs.drop(columns="cg")
            check_cgs = check_cgs.merge(
                model.topology[
                    ["region", "process", "commodity", "csets", "io"]
                ].drop_duplicates(),
                how="left",
            )
            check_cgs["gmap"] = True
            check_cgs = pd.concat(
                [
                    model.commodity_groups,
                    check_cgs[["region", "commodity", "commoditygroup", "gmap"]],
                ],
                ignore_index=True,
            )
            model.commodity_groups = check_cgs.drop_duplicates().dropna()

    return tables


def fix_topology(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Include information on process availability by region in model.topology,
    model.processes, and the fi_t table. Remove indication of auxillary flows from
    model.topology.
    """
    mapping = {"IN-A": "IN", "OUT-A": "OUT"}
    model.topology.replace({"io": mapping}, inplace=True)

    if Tag.tfm_ava in tables:
        modules_with_ava = set(tables[Tag.tfm_ava]["module_name"])
        df = tables[Tag.tfm_ava].explode("process", ignore_index=True)
        # Ensure valid combinations of process / module_name
        df = df.merge(
            model.processes[["process", "module_name"]].drop_duplicates(),
            how="inner",
            on=["process", "module_name"],
        )
        # Update tfm_ava
        tables[Tag.tfm_ava] = df
        # Overwrite with the last value for each process/region pair
        df = df.drop_duplicates(
            subset=[col for col in df.columns if col != "value"], keep="last"
        )
        # Remove rows with zero value
        df = df[~(df["value"] == 0)]
        # Group back processes with multiple values
        df = df.groupby(
            [col for col in df.columns if col != "process"], as_index=False
        ).agg({"process": list})[df.columns]
        # Remove invalid rows from fi_t, processes, and topology
        verify_cols = ["region", "process", "module_name"]
        tables[Tag.fi_t] = _remove_invalid_rows(
            tables[Tag.fi_t],
            df,
            verify_cols,
            limit_to={"module_name": modules_with_ava},
        ).reset_index(drop=True)
        # TODO: This should happen much earlier in the process
        model.processes = _remove_invalid_rows(
            model.processes, df, verify_cols, limit_to={"module_name": modules_with_ava}
        ).reset_index(drop=True)
        # TODO: should be unnecessary if model.processes is updated early enough
        # Remove topology rows that are not in the processes
        model.topology = pd.merge(
            model.topology,
            model.processes[["region", "process"]].drop_duplicates(),
            on=["region", "process"],
            how="inner",
        )

    return tables


def generate_implied_topology(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Generate implied topology i.e., niether part of model.topology nor model.trade."""
    # Only done for FLO_EMIS at the moment and is oversimplified.
    # TODO: Generalize for other relevant attributes and improve the logic.
    source = tables[Tag.fi_t]
    i = source["attribute"] == "FLO_EMIS"
    if any(i):
        df = source[i].copy()
        df = df[["region", "process", "commodity"]].drop_duplicates()
        # Only keep those commodities that are ENV
        i_env = model.commodities["csets"] == "ENV"
        df = df.merge(
            model.commodities.loc[i_env, ["region", "commodity"]].drop_duplicates()
        )
        df["io"] = "OUT"
        # Exclude any existing entries in model.topology
        df = df.merge(model.topology, how="left", indicator=True)
        df = df[["region", "process", "commodity", "io"]][df["_merge"] == "left_only"]
        model.implied_topology = df.reset_index(drop=True)

    return tables


def complete_processes(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:
    """Generate processes based on trade links if not defined elsewhere."""
    # Dataframe with region, process and commodity columns (no trade direction)
    cols_in = ["origin", "process", "in"]
    cols_in.extend(dm_cols)
    cols_out = ["destination", "process", "out"]
    cols_out.extend(dm_cols)
    cols = ["region", "process", "commodity"]
    cols.extend(dm_cols)
    trade_processes = pd.concat(
        [
            model.trade.loc[:, cols_in].rename(
                columns={"origin": "region", "in": "commodity"}
            ),
            model.trade.loc[:, cols_out].rename(
                columns={"destination": "region", "out": "commodity"}
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    # Determine undeclared trade process
    undeclared_td = trade_processes.merge(
        model.processes.loc[:, ["region", "process"]], how="left", indicator=True
    )
    # Keep only those undeclared processes that are in internal regions
    undeclared_td = undeclared_td.loc[
        (
            undeclared_td["region"].isin(model.internal_regions)
            & (undeclared_td["_merge"] == "left_only")
        ),
        cols,
    ]
    # Include additional info from model.commodities
    undeclared_td = undeclared_td.merge(
        model.commodities.loc[:, ["region", "commodity", "csets", "ctslvl", "unit"]],
        how="left",
    )
    # Remove unnecessary columns
    undeclared_td.drop(columns=["commodity"], inplace=True)
    # Rename to match columns in model.processes
    undeclared_td.rename(
        columns={"csets": "primarycg", "ctslvl": "tslvl", "unit": "tact"}, inplace=True
    )
    # Specify expected set
    undeclared_td["sets"] = "IRE"
    # Remove full duplicates in case generated
    undeclared_td.drop_duplicates(keep="last", inplace=True)
    # TODO: Handle possible confilicting input
    # Print warnings in case of conflicting input data
    for i in ["primarycg", "tslvl", "tact"]:
        duplicates = undeclared_td.loc[:, ["region", "process", i]].duplicated(
            keep=False
        )
        if any(duplicates):
            duplicates = undeclared_td.loc[duplicates, ["region", "process", i]]
            processes = duplicates["process"].unique()
            regions = duplicates["region"].unique()
            logger.warning(f"Multiple possible {i} for {processes} in {regions}")

    model.processes = pd.concat([model.processes, undeclared_td], ignore_index=True)

    return tables


def apply_final_fixup(
    config: Config,
    tables: dict[str, DataFrame],
    model: TimesModel,
) -> dict[str, DataFrame]:

    veda_ire_sets = model.custom_sets
    reg_com_flows = model.topology[["region", "process", "commodity"]].copy()
    reg_com_flows.drop_duplicates(inplace=True, ignore_index=True)
    df = tables[Tag.fi_t]
    for col in ["other_indexes", "cg"]:
        _populate_defaults(Tag.fi_t, df, col, config, "original_attr")
    _populate_calculated_defaults(df, model)

    # Fill other_indexes for COST
    cost_mapping = {"MIN": "IMP", "EXP": "EXP", "IMP": "IMP"}
    cost_index = (df["original_attr"] == "COST") & df["process"].notna()

    if any(cost_index):
        processes = set(df[cost_index]["process"].unique())
        # Index of IRE processes and their IRE sets specification
        sets_index = veda_ire_sets["process"].isin(processes) & veda_ire_sets[
            "sets"
        ].isin(cost_mapping.keys())

        ire_processes = set(veda_ire_sets["process"][sets_index])
        other_processes = processes.difference(ire_processes)

        if other_processes:
            logger.warning(
                f"COST won't be processed as IRE_PRICE for {other_processes}, because they are not in IMP/EXP/MIN"
            )

        if any(ire_processes):
            # Ensure only one IRE set is specified per process
            subst_df = veda_ire_sets[sets_index].drop_duplicates(
                subset="process", keep="last"
            )
            index = cost_index & df["process"].isin(ire_processes)
            df.loc[index, "other_indexes"] = df.loc[index, "process"].replace(
                subst_df.set_index("process")["sets"].replace(cost_mapping).to_dict()
            )

    # Use Commodity to store the active commodity for EXP / IMP
    index = df["original_attr"].isin({"COST", "IRE_PRICE"})
    if any(index):
        i_exp = index & (df["other_indexes"] == "EXP")
        df.loc[i_exp, "commodity"] = df.loc[i_exp, "commodity-in"]
        i_imp = index & (df["other_indexes"] == "IMP")
        df.loc[i_imp, "commodity"] = df.loc[i_imp, "commodity-out"]

    # Fill Commodity for COST (alias of IRE_PRICE) if missing
    i_com_na = (df["original_attr"] == "COST") & df["commodity"].isna()
    if any(i_com_na):
        comm_rp = reg_com_flows.groupby(["region", "process"]).agg(set)
        comm_rp["commodity"] = comm_rp["commodity"].str.join(",")
        df.set_index(["region", "process"], inplace=True)
        i_cost = df["original_attr"] == "COST"
        df.loc[i_cost, "commodity"] = df["commodity"][i_cost].fillna(
            comm_rp["commodity"].to_dict()
        )
        df.reset_index(inplace=True)

    # Handle PRC_RESID specified for a single year
    stock_index = (df["attribute"] == "PRC_RESID") & df["process"].notna()
    if any(stock_index):
        # Include only processes defined in BASE, but not in _SUP
        i_vt = (
            stock_index
            & (df["module_name"] == "BASE")
            & (~df["source_filename"].str.contains("_SUP", case=False))
        )
        # Create (region, process) index for data defined in vt
        i_rp_vt = df[i_vt].set_index(["region", "process"]).index.drop_duplicates()
        # Create (region, process) index for which NCAP_BND is specified
        i_rp_ncap_bnd = (
            df[(df["attribute"] == "NCAP_BND") & df["process"].notna()]
            .set_index(["region", "process"])
            .index.drop_duplicates()
        )
        # Exclude processes with NCAP_BND already defined
        i_rp_vt = i_rp_vt.difference(i_rp_ncap_bnd)
        # Create extra rows with NCAP_BND
        ncap_bnd_data = {
            "attribute": "NCAP_BND",
            "limtype": "UP",
            "year": 0,
            "value": 2,
        }
        ncap_bnd_rows = pd.DataFrame(ncap_bnd_data, index=i_rp_vt).reset_index()
        # Create df list to concatenate later on
        df_list = [df, ncap_bnd_rows]
        # Stock indexed by process/region
        cols = ["region", "process", "year"]
        df_rp = (
            df[stock_index]
            .drop_duplicates(subset=cols, keep="last")
            .set_index(["region", "process"])
        )
        # Index of region/process with STOCK specified only once
        i_single_stock = ~df_rp.index.duplicated(keep=False)
        i_single_stock = i_single_stock & (df_rp["value"] != 0)
        # TODO: TIMES already handles this. Drop?
        if any(i_single_stock):
            default_life = 30
            life_rp = (
                df[df["original_attr"].isin({"NCAP_TLIFE", "LIFE"})]
                .drop_duplicates(subset=["region", "process"], keep="last")
                .set_index(["region", "process"])["value"]
            )
            stock_rows = df_rp[["attribute", "year"]][i_single_stock].copy()
            stock_rows = stock_rows.merge(
                life_rp, how="left", left_index=True, right_index=True
            )
            # Use default if lifetime not specified
            stock_rows.loc[stock_rows["value"].isna(), "value"] = default_life
            # TODO: Validate that lifetime is integer and warn user if not?
            i_integer = stock_rows["value"].apply(lambda x: isinstance(x, int))
            if any(~i_integer):
                stock_rows.loc[~i_integer, "value"] = stock_rows["value"][
                    ~i_integer
                ].apply(lambda x: round(x))
            # Exclude rows with a lifetime of 1 year, since inserting a zero would not matter
            stock_rows = stock_rows[stock_rows["value"] != 1]
            # Calculate the year in which STOCK is zero
            stock_rows["year"] = stock_rows["year"] + stock_rows["value"]
            # Specify stock value zero
            stock_rows["value"] = 0
            stock_rows.reset_index(inplace=True)
            df_list.append(stock_rows)

        df = pd.concat(df_list)

    # Clean up
    # TODO: Do this comprehensively for all relevant tables
    # TODO: Duplicates should only be removed if in the same file/module
    keep_cols = {
        "attribute",
        "region",
        "process",
        "commodity",
        "other_indexes",
        "cg",
        "year",
        "year2",
        "timeslice",
        "currency",
        "limtype",
        "sow",
        "stage",
        "module_name",
        "module_type",
    }
    df.dropna(subset="value", inplace=True)
    drop_cols = [col for col in df.columns if col != "value" and col not in keep_cols]
    df.drop(columns=drop_cols, inplace=True)
    df = df.drop_duplicates(subset=list(keep_cols), keep="last")

    # Control application of i/e rules from syssettings
    if not config.ie_override_in_syssettings:
        df = df.reset_index(drop=True)
        # Remove i/e rules from syssettings if present in BASE and SubRES
        i = (df["year"] == 0) & (
            df["module_type"].isin(["base", "syssettings", "subres"])
        )
        duplicated = df[i].duplicated(
            subset=[
                col
                for col in keep_cols
                if col != "value" and col not in {"module_name", "module_type"}
            ],
            keep=False,
        )
        i = (df["module_type"] == "syssettings") & duplicated
        if any(i):
            df = df[~i]

    tables[Tag.fi_t] = df.reset_index(drop=True)

    if Tag.uc_t in tables.keys():
        df = tables[Tag.uc_t]
        keep_cols = keep_cols.difference({"year2", "cg"})
        keep_cols = keep_cols.union({"uc_n", "side"})
        df.dropna(subset="value", inplace=True)
        drop_cols = [
            col for col in df.columns if col != "value" and col not in keep_cols
        ]
        df.drop(columns=drop_cols, inplace=True)
        df = df.drop_duplicates(subset=list(keep_cols), keep="last")
        tables[Tag.uc_t] = df.reset_index(drop=True)

    return tables


def expand_rows_parallel(
    config: Config,
    tables: list[EmbeddedXlTable],
    model: TimesModel,
) -> list[EmbeddedXlTable]:
    query_columns_lists = [
        (config.query_columns[Tag(table.tag)] if Tag.has_tag(table.tag) else set())
        for table in tables
    ]
    lists_columns_lists = [
        (config.lists_columns[Tag(table.tag)] if Tag.has_tag(table.tag) else set())
        for table in tables
    ]
    with ProcessPoolExecutor(max_workers) as executor:
        return list(
            executor.map(expand_rows, query_columns_lists, lists_columns_lists, tables)
        )
