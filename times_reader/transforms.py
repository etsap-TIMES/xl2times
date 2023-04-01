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

csets_ordered_for_pcg = ["DEM", "MAT", "NRG", "ENV", "FIN"]
default_pcg_suffixes = [
    cset + io for cset in csets_ordered_for_pcg for io in ["I", "O"]
]

# Specify a list of aliases per TIMES attribute
aliases_by_attr = {
    "ACT_BND": ["ACTBND", "BNDACT"],
    "ACT_COST": ["ACTCOST", "VAROM"],
    "ACT_CUM": ["CUM"],
    "ACT_EFF": ["CEFF", "CEFFICIENCY", "CEFF-I", "CEFF-O", "EFF", "EFFICIENCY"],
    "COM_PROJ": ["DEMAND"],
    "FLO_EMIS": ["FEMIS", "FLO_EFF", "ENV_ACT", "ENVACT"],
    "FLO_DELIV": ["DELIV"],
    "FLO_SHAR": ["FLOSHAR", "SHARE", "SHARE-I", "SHARE-O"],
    "G_DRATE": ["DISCOUNT"],
    "G_YRFR": ["YRFR"],
    "IRE_PRICE": ["COST"],
    "NCAP_AF": ["AF"],
    "NCAP_AFA": ["AFA"],
    "NCAP_AFC": ["AFC"],
    "NCAP_BND": ["IBOND", "BNDINV"],
    "NCAP_CHPR": ["CHPR"],
    "NCAP_COST": ["INVCOST"],
    "NCAP_CPX": ["CPX"],
    "NCAP_FOM": ["FIXOM"],
    "NCAP_PASTI": ["PASTI"],
    "NCAP_PKCNT": ["PEAK"],
    "NCAP_START": ["START"],
    "NCAP_TLIFE": ["LIFE"],
    "PRC_ACTFLO": ["ACTFLO"],
    "PRC_CAPACT": ["CAP2ACT"],
    "PRC_RESID": ["RESID", "STOCK"],
    "STG_EFF": ["S_EFF"],
    "VDA_CEH": ["CEH"],
    "VDA_EMCB": ["EMCB"],
    "VDA_FLOP": ["FLOP"],
}

# Specify, in order of priority, what to use as CommName if CommName is empty
attr_com_def = {
    "CEFF": ["Comm-IN", "Comm-OUT"],  # this one is a Veda alias
    "CEFFICIENCY": ["Comm-IN", "Comm-OUT"],  # this one is an alias of the above
    "CEFF-I": ["Comm-IN"],
    "CEFF-O": ["Comm-OUT"],
    "FLO_COST": ["Comm-IN", "Comm-OUT"],
    "FLO_DELIV": ["Comm-IN"],
    "DELIV": ["Comm-IN"],
    "FLO_EMIS": ["Comm-OUT", "Comm-IN"],
    "FEMIS": ["Comm-OUT", "Comm-IN"],
    "FLO_EFF": ["Comm-OUT", "Comm-IN"],
    "ENV_ACT": ["Comm-OUT", "Comm-IN"],
    "ENVACT": ["Comm-OUT", "Comm-IN"],
    "FLO_MARK": ["Comm-IN", "Comm-OUT"],
    "FLO_SHAR": ["Comm-IN", "Comm-OUT"],
    "FLOSHAR": ["Comm-IN", "Comm-OUT"],
    "SHARE": ["Comm-IN", "Comm-OUT"],
    "SHARE-I": ["Comm-IN"],
    "SHARE-O": ["Comm-OUT"],
    "FLO_SUB": ["Comm-OUT", "Comm-IN"],
    "FLO_TAX": ["Comm-OUT", "Comm-IN"],
    "STGIN_BND": ["Comm-IN"],
    "STGOUT_BND": ["Comm-OUT"],
}

attr_limtype_def = {
    "FX": [
        "ACT_LOSPL",
        "FLO_SHAR",
        "FLOSHAR",
        "SHARE",
        "SHARE-I",
        "SHARE-O",
        "NCAP_CHPR",
        "CHPR",
        "REG_BDNCAP",
    ],
    "LO": ["BS_STIME", "GR_VARGEN", "RCAP_BND"],
    "UP": [
        "ACT_BND",
        "ACTBND",
        "BNDACT",
        "ACT_CSTRMP",
        "ACT_CSTSD",
        "ACT_CUM",
        "CUM",
        "ACT_LOSSD",
        "ACT_SDTIME",
        "ACT_TIME",
        "ACT_UPS",
        "BS_BNDPRS",
        "BS_SHARE",
        "CAP_BND",
        "COM_BNDNET",
        "COM_BNDPRD",
        "COM_CUMNET",
        "COM_CUMPRD",
        "FLO_BND",
        "FLO_CUM",
        "FLO_FR",
        "FLO_MARK",
        "IRE_BND",
        "IRE_XBND",
        "NCAP_AF",
        "AF",
        "NCAP_AFA",
        "AFA",
        "NCAP_AFAC",
        "NCAP_AFS",
        "NCAP_AFSX",
        "NCAP_BND",
        "PRC_MARK",
        "REG_BNDCST",
        "REG_CUMCST",
        "S_CAP_BND",
        "S_COM_CUMNET",
        "S_COM_CUMPRD",
        "S_FLO_CUM",
        "S_UC_RHS",
        "S_UC_RHSR",
        "S_UC_RHSRT",
        "S_UC_RHSRTS",
        "S_UC_RHSTS",
        "STGIN_BND",
        "STGOUT_BND",
        "UC_DYNBND",
        "UC_RHS",
        "UC_RHSR",
        "UC_RHSRT",
        "UC_RHSRTS",
        "UC_RHST",
        "UC_RHSTS",
    ],
}

attr_timeslice_def = {
    "DAYNITE": ["ACT_CSTUP"],
    "ANNUAL": [
        "ACT_BND",
        "ACTBND",
        "BNDACT",
        "ACT_EFF",
        "CEFF",
        "CEFF-O",
        "CEFF-I",
        "CEFFICIENCY",
        "EFFICIENCY",
        "EFF",
        "ACT_FLO",
        "ACT_UPS",
        "BS_BNDPRS",
        "BS_DELTA",
        "BS_DEMDET",
        "BS_MAINT",
        "BS_OMEGA",
        "BS_RMAX",
        "BS_SIGMA",
        "COM_BNDNET",
        "COM_BNDPRD",
        "COM_BPRICE",
        "COM_CSTBAL",
        "COM_CSTNET",
        "COM_CSTPRD",
        "COM_ELAST",
        "COM_IE",
        "COM_SUBNET",
        "COM_SUBPRD",
        "COM_TAXNET",
        "COM_TAXPRD",
        "FLO_BND",
        "FLO_COST",
        "FLO_DELIV",
        "DELIV",
        "FLO_EFF",
        "FLO_EMIS",
        "FEMIS",
        "ENV_ACT",
        "ENVACT",
        "FLO_FUNC",
        "FLO_SHAR",
        "FLOSHAR",
        "SHARE",
        "SHARE-I",
        "SHARE-O",
        "FLO_SUB",
        "FLO_TAX",
        "G_YRFR",
        "GR_DEMFR",
        "IRE_BND",
        "IRE_FLOSUM",
        "IRE_PRICE",
        "COST",
        "IRE_XBND",
        "NCAP_AF",
        "AF",
        "NCAP_AFC",
        "AFC",
        "NCAP_AFCS",
        "NCAP_PKCNT",
        "PEAK",
        "PRC_FOFF",
        "S_UC_RHSRTS",
        "S_UC_RHSTS",
        "STG_CHRG",
        "STG_LOSS",
        "STG_SIFT",
        "STGIN_BND",
        "STGOUT_BND",
        "TS_CYCLE",
        "UC_ACT",
        "UC_COMCON",
        "UC_COMNET",
        "UC_COMPRD",
        "UC_FLO",
        "UC_IRE",
        "UC_RHSRTS",
        "UC_RHSTS",
        "VDA_FLOP",
    ],
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


def include_tables_source(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Add a column specifying source filename to every table
    """

    def include_table_source(table: datatypes.EmbeddedXlTable):
        df = table.dataframe.copy()
        df["source_filename"] = table.filename
        return replace(table, dataframe=df)

    return [include_table_source(table) for table in tables]


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
            for c, table in cols:
                print(f"  {c} from {table.range}, {table.sheetname}, {table.filename}")
        else:
            df = pd.concat([table.dataframe for table in group], ignore_index=True)
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
        mapping = {"YEAR": "Year", "Timeslices": "TimeSlice"}
        df = df.rename(columns=mapping)

        if "CURR" in df.columns.values:
            df.rename(columns={"CURR": "Curr"}, inplace=True)

        nrows = df.shape[0]

        # TODO: this should only be removed if it is a comment column
        # Remove any TechDesc column
        if "TechDesc" in df.columns:
            df.drop("TechDesc", axis=1, inplace=True)

        # TODO: Review this. CommGrp is an alias for Other_Indexes
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
            "CommName",
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
            "CommName",
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
            if attr == "OUTPUT":
                i = df[attribute] == attr
                df.loc[i, "Comm-IN"] = df.loc[i, "Comm-OUT-A"]
                df.loc[i, attribute] = "CEFF"
            elif attr == "END":
                i = df[attribute] == attr
                df.loc[i, "Year"] = df.loc[i, "VALUE"].astype("int") + 1
                df.loc[i, other] = "EOH"
                df.loc[i, attribute] = "PRC_NOFF"

        df = df.reset_index(drop=True)

        # Fill other_indexes for COST
        cost_mapping = {"MIN": "IMP", "EXP": "EXP", "IMP": "IMP"}
        i = df[attribute] == "COST"
        for process in df[i]["TechName"].unique():
            veda_process_set = (
                veda_process_sets["Sets"]
                .loc[veda_process_sets["TechName"] == process]
                .unique()
            )
            df.loc[i & (df["TechName"] == process), other] = cost_mapping[
                veda_process_set[0]
            ]

        # Use CommName to store the active commodity for EXP / IMP
        i = df[attribute].isin(["COST", "IRE_PRICE"])
        i_exp = i & (df[other] == "EXP")
        df.loc[i_exp, "CommName"] = df.loc[i_exp, "Comm-IN"]
        i_imp = i & (df[other] == "IMP")
        df.loc[i_imp, "CommName"] = df.loc[i_imp, "Comm-OUT"]

        # Should have all index_columns and VALUE
        if table.tag == datatypes.Tag.fi_t and len(df.columns) != (
            len(index_columns) + 1
        ):
            raise ValueError(f"len(df.columns) = {len(df.columns)}")

        df["Year2"] = df.apply(
            lambda row: int(row["Year"].split("-")[1])
            if "-" in str(row["Year"])
            else "EOH",
            axis=1,
        )

        df["Year"] = df.apply(
            lambda row: int(row["Year"].split("-")[0])
            if "-" in str(row["Year"])
            else (row["Year"] if row["Year"] != "" else "BOH"),
            axis=1,
        )

        return replace(table, dataframe=df)

    return [process_flexible_import_table(t, veda_process_sets) for t in tables]


def process_user_constraint_tables(
    tables: List[datatypes.EmbeddedXlTable],
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
            "UC_Desc",  # Why is this in the index columns?
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
            if colname in ["Sets", "Csets", "TechName"]:
                utils.missing_value_inherit(df, colname)
            elif colname == "LimType" and table.tag == datatypes.Tag.fi_comm and False:
                isna = df[colname].isna()
                ismat = df["Csets"] == "MAT"
                df.loc[isna & ismat, colname] = "FX"
                df.loc[isna & ~ismat, colname] = "LO"
            elif (
                colname == "LimType"
                and (table.tag == datatypes.Tag.fi_t or table.tag.startswith("~TFM"))
                and len(df) > 0
            ):
                isna = df[colname].isna()
                for lim in attr_limtype_def.keys():
                    df.loc[
                        isna & df["Attribute"].str.upper().isin(attr_limtype_def[lim]),
                        colname,
                    ] = lim
            elif colname == "TimeSlice" and len(df) > 0 and "Attribute" in df.columns:
                isna = df[colname].isna()
                for timeslice in attr_timeslice_def.keys():
                    df.loc[
                        isna
                        & df["Attribute"]
                        .str.upper()
                        .isin(attr_timeslice_def[timeslice]),
                        colname,
                    ] = timeslice
            elif colname == "Tslvl":  # or colname == "CTSLvl" or colname == "PeakTS":
                df[colname].fillna("ANNUAL", inplace=True)
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
        "Csets": csets_ordered_for_pcg,
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


def process_units(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    commodity_units = set()
    process_act_units = set()
    process_cap_units = set()
    currencies = set()

    for table in tables:
        if table.tag == datatypes.Tag.fi_comm:
            commodity_units.update(table.dataframe["Unit"].unique())

        if table.tag == datatypes.Tag.fi_process:
            process_act_units.update(table.dataframe["Tact"].unique())
            process_cap_units.update(
                [
                    s.upper()
                    for s in table.dataframe["Tcap"].unique()
                    if s != None and s != ""
                ]
            )

        if table.tag == datatypes.Tag.currencies:
            currencies.update(table.dataframe["Currency"].unique())

    tables.append(
        datatypes.EmbeddedXlTable(
            tag="~UNITS_ACT",
            uc_sets={},
            sheetname="",
            range="",
            filename="",
            dataframe=DataFrame({"UNITS": sorted(process_act_units)}),
        )
    )

    tables.append(
        datatypes.EmbeddedXlTable(
            tag="~UNITS_CAP",
            uc_sets={},
            sheetname="",
            range="",
            filename="",
            dataframe=DataFrame({"UNITS": sorted(process_cap_units)}),
        )
    )

    tables.append(
        datatypes.EmbeddedXlTable(
            tag="~UNITS_COM",
            uc_sets={},
            sheetname="",
            range="",
            filename="",
            dataframe=DataFrame({"UNITS": sorted(commodity_units)}),
        )
    )

    tables.append(
        datatypes.EmbeddedXlTable(
            tag="~ALL_UNITS",
            uc_sets={},
            sheetname="",
            range="",
            filename="",
            dataframe=DataFrame(
                {
                    "UNITS": sorted(
                        commodity_units.union(process_act_units).union(
                            process_cap_units.union(currencies)
                        )
                    )
                }
            ),
        )
    )

    return tables


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


def generate_all_regions(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Include IMPEXP and MINRNW together with the user-defined regions in the AllRegions set
    IMPEXP and MINRNW are external regions that are defined by default by Veda
    """

    external_regions = ["IMPEXP", "MINRNW"]
    df = pd.DataFrame(external_regions, columns=["Region"])

    for table in tables:
        if table.tag == datatypes.Tag.book_regions_map:
            df = pd.concat([df, table.dataframe])

    tables.append(
        datatypes.EmbeddedXlTable(
            tag="AllRegions",
            uc_sets={},
            sheetname="",
            range="",
            filename="",
            dataframe=df,
        )
    )

    return tables


def capitalise_attributes(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Ensure that all attributes are uppercase
    """

    # TODO: This should be part of normalisation
    def capitalise_attributes_table(table: datatypes.EmbeddedXlTable):
        df = table.dataframe.copy()
        if "Attribute" in df.columns and len(df) > 0:
            df["Attribute"] = df["Attribute"].str.upper()
            return replace(table, dataframe=df)
        else:
            return table

    return [capitalise_attributes_table(table) for table in tables]


def apply_fixups(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    reg_com_flows = utils.single_table(tables, "ProcessTopology").dataframe.copy()
    reg_com_flows.drop(columns="IO", inplace=True)

    def apply_fixups_table(table: datatypes.EmbeddedXlTable):
        if not table.tag.startswith(datatypes.Tag.fi_t) or table.dataframe.size == 0:
            return table

        df = table.dataframe.copy()

        # Populate CommName based on defaults
        i = (
            df["Attribute"].str.upper().isin(attr_com_def.keys())
            & df["CommName"].isna()
        )
        if len(df[i]) > 0:
            for attr in df[i]["Attribute"].unique():
                for com_in_out in attr_com_def[attr.upper()]:
                    index = i & (df["Attribute"] == attr) & (df["CommName"].isna())
                    if len(df[index]) > 0:
                        df.loc[index, ["CommName"]] = df[index][com_in_out]

        # Fill other indexes for some attributes
        # FLO_SHAR
        i = df["Attribute"] == "SHARE-I"
        df.loc[i, "Other_Indexes"] = "NRGI"
        i = df["Attribute"] == "SHARE-O"
        df.loc[i, "Other_Indexes"] = "NRGO"
        # ACT_EFF
        i = df["Attribute"].isin(["CEFF", "CEFFICIENCY", "CEFF-I", "CEFF-O"])
        df.loc[i, "Other_Indexes"] = df[i]["CommName"]
        i = df["Attribute"].isin(["EFF", "EFFICIENCY"])
        df.loc[i, "Other_Indexes"] = "ACT"
        # FLO_EMIS
        i = df["Attribute"].isin(["ENV_ACT", "ENVACT"])
        df.loc[i, "Other_Indexes"] = "ACT"

        # Fill CommName for COST (alias of IRE_PRICE) if missing
        if "Attribute" in df.columns and "COST" in df["Attribute"].unique():
            i = (df["Attribute"] == "COST") & df["CommName"].isna()
            if any(i):
                df.loc[i, "CommName"] = df[i].apply(
                    lambda row: ",".join(
                        reg_com_flows.loc[
                            (reg_com_flows["Region"] == row["Region"])
                            & (reg_com_flows["TechName"] == row["TechName"]),
                            "CommName",
                        ].unique()
                    ),
                    axis=1,
                )
                # TODO: Expand rows if multiple comma-separated commodities are included

        return replace(table, dataframe=df)

    return [apply_fixups_table(table) for table in tables]


def extract_commodity_groups(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    process_tables = [t for t in tables if t.tag == datatypes.Tag.fi_process]
    commodity_tables = [t for t in tables if t.tag == datatypes.Tag.fi_comm]

    # Veda determines default PCG based on predetermined order and presence of OUT/IN commodity

    columns = ["Region", "TechName", "PrimaryCG"]
    reg_prc_pcg = pd.DataFrame(columns=columns)
    for process_table in process_tables:
        df = process_table.dataframe[columns]
        reg_prc_pcg = pd.concat([reg_prc_pcg, df])
    reg_prc_pcg.drop_duplicates(keep="first", inplace=True)

    # DataFrame with Veda PCGs specified in the process declaration tables
    reg_prc_veda_pcg = reg_prc_pcg.loc[
        reg_prc_pcg["PrimaryCG"].isin(default_pcg_suffixes)
    ]

    # Extract commodities and their sets by region
    columns = ["Region", "Csets", "CommName"]
    comm_set = pd.DataFrame(columns=columns)
    for commodity_table in commodity_tables:
        df = commodity_table.dataframe[columns]
        comm_set = pd.concat([comm_set, df])
    comm_set.drop_duplicates(keep="first", inplace=True)

    prc_top = utils.single_table(tables, "ProcessTopology").dataframe

    # Commodity groups by process, region and commodity
    comm_groups = pd.merge(prc_top, comm_set, on=["Region", "CommName"])
    comm_groups["CommodityGroup"] = None
    # Store the number of IN/OUT commodities of the same type per Region and Process in CommodityGroup
    for region in comm_groups["Region"].unique():
        i_reg = comm_groups["Region"] == region
        for process in comm_groups[i_reg]["TechName"].unique():
            i_reg_prc = i_reg & (comm_groups["TechName"] == process)
            for cset in comm_groups[i_reg_prc]["Csets"].unique():
                i_reg_prc_cset = i_reg_prc & (comm_groups["Csets"] == cset)
                for io in comm_groups[i_reg_prc_cset]["IO"].unique():
                    i_reg_prc_cset_io = i_reg_prc_cset & (comm_groups["IO"] == io)
                    comm_groups.loc[i_reg_prc_cset_io, "CommodityGroup"] = sum(
                        i_reg_prc_cset_io
                    )

    def name_comm_group(df):
        """
        Return the name of a commodity group based on the member count
        """

        if df["CommodityGroup"] > 1:
            return df["TechName"] + "_" + df["Csets"] + df["IO"][:1]
        else:
            return df["CommName"]

    # Replace commodity group member count with the name
    comm_groups["CommodityGroup"] = comm_groups.apply(name_comm_group, axis=1)

    # Determine default PCG according to Veda
    comm_groups["DefaultVedaPCG"] = None
    for region in comm_groups["Region"].unique():
        i_reg = comm_groups["Region"] == region
        for process in comm_groups[i_reg]["TechName"]:
            i_reg_prc = i_reg & (comm_groups["TechName"] == process)
            default_set = False
            for io in ["OUT", "IN"]:
                if default_set:
                    break
                i_reg_prc_io = i_reg_prc & (comm_groups["IO"] == io)
                for cset in csets_ordered_for_pcg:
                    i_reg_prc_io_cset = i_reg_prc_io & (comm_groups["Csets"] == cset)
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
        df["Csets"] = df["PrimaryCG"].replace(suffix_to_cset)
        df["IO"] = df["PrimaryCG"].replace(suffix_to_io)
        df["CommodityGroup"] = df["TechName"] + "_" + df["PrimaryCG"]
        columns = ["Region", "TechName", "IO", "Csets"]
        df = pd.merge(
            df[columns + ["CommodityGroup"]],
            comm_groups[columns + ["CommName"]],
            on=columns,
        )
        comm_groups = pd.concat([comm_groups, df])
        comm_groups.drop_duplicates(
            subset=["Region", "TechName", "IO", "CommName", "Csets", "CommodityGroup"],
            keep="first",
            inplace=True,
        )

    # TODO apply renamings from ~TFM_TOPINS e.g. RSDAHT to RSDAHT2

    tables.append(
        datatypes.EmbeddedXlTable(
            sheetname="",
            range="",
            filename="",
            uc_sets="",
            tag="COMM_GROUPS",
            dataframe=comm_groups,
        )
    )

    i = comm_groups["CommodityGroup"] != comm_groups["CommName"]

    tables.append(
        datatypes.EmbeddedXlTable(
            sheetname="",
            range="",
            filename="",
            uc_sets="",
            tag="COM_GMAP",
            dataframe=comm_groups.loc[i, ["Region", "CommodityGroup", "CommName"]],
        )
    )

    return tables


def generate_top_ire(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Generate inter-regional exchange topology
    """

    veda_set_ext_reg_mapping = {"IMP": "IMPEXP", "EXP": "IMPEXP", "MIN": "MINRNW"}
    dummy_process_cset = [["NRG", "IMPNRGZ"], ["MAT", "IMPMATZ"], ["DEM", "IMPDEMZ"]]
    veda_process_sets = utils.single_table(tables, "VedaProcessSets").dataframe
    com_map = utils.single_table(tables, "COMM_GROUPS").dataframe

    ire_prc = pd.DataFrame(columns=["Region", "TechName"])
    for table in tables:
        if table.tag == datatypes.Tag.fi_process:
            df = table.dataframe
            ire_prc = pd.concat(
                [ire_prc, df.loc[df["Sets"] == "IRE", ["Region", "TechName"]]]
            )
    ire_prc.drop_duplicates(keep="first", inplace=True)

    internal_regions = pd.DataFrame([], columns=["Region"])
    for table in tables:
        if table.tag == datatypes.Tag.book_regions_map:
            internal_regions = pd.concat(
                [internal_regions, table.dataframe.loc[:, ["Region"]]]
            )

    # Generate inter-regional exchange topology
    top_ire = pd.DataFrame(dummy_process_cset, columns=["Csets", "TechName"])
    top_ire = pd.merge(top_ire, internal_regions, how="cross")
    top_ire = pd.merge(top_ire, com_map[["Region", "Csets", "CommName"]])
    top_ire.drop(columns=["Csets"], inplace=True)
    top_ire["IO"] = "OUT"
    top_ire = pd.concat([top_ire, com_map[["Region", "TechName", "CommName", "IO"]]])
    top_ire = pd.merge(top_ire, ire_prc)
    top_ire = pd.merge(top_ire, veda_process_sets)
    top_ire["Region2"] = top_ire["Sets"].replace(veda_set_ext_reg_mapping)
    top_ire[["Origin", "Destination", "IN", "OUT"]] = None
    for io in ["IN", "OUT"]:
        index = top_ire["IO"] == io
        top_ire.loc[index, [io]] = top_ire["CommName"].loc[index]
    na_out = top_ire["OUT"].isna()
    top_ire.loc[na_out, ["OUT"]] = top_ire["IN"].loc[na_out]
    na_in = top_ire["IN"].isna()
    top_ire.loc[na_in, ["IN"]] = top_ire["OUT"].loc[na_in]
    is_imp_or_min = top_ire["Sets"].isin(["IMP", "MIN"])
    is_exp = top_ire["Sets"].isin(["EXP"])
    top_ire.loc[is_imp_or_min, ["Origin"]] = top_ire["Region2"].loc[is_imp_or_min]
    top_ire.loc[is_imp_or_min, ["Destination"]] = top_ire["Region"].loc[is_imp_or_min]
    top_ire.loc[is_exp, ["Origin"]] = top_ire["Region"].loc[is_exp]
    top_ire.loc[is_exp, ["Destination"]] = top_ire["Region2"].loc[is_exp]
    top_ire.drop(columns=["Region", "Region2", "Sets", "IO"], inplace=True)
    top_ire.drop_duplicates(keep="first", inplace=True, ignore_index=True)

    tables.append(
        datatypes.EmbeddedXlTable(
            tag="TOP_IRE",
            uc_sets={},
            sheetname="",
            range="",
            filename="",
            dataframe=top_ire,
        )
    )
    return tables


def fill_in_missing_pcgs(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Fill in missing primary commodity groups in FI_Process tables.
    Expand primary commodity groups specified in FI_Process tables by a suffix.
    """

    def expand_pcg_from_suffix(df):
        """
        Return the name of a default primary commodity group based on suffix and process name
        """

        if df["PrimaryCG"] in default_pcg_suffixes:
            return df["TechName"] + "_" + df["PrimaryCG"]
        else:
            return df["PrimaryCG"]

    result = []

    for table in tables:
        if table.tag != datatypes.Tag.fi_process:
            result.append(table)
        else:
            df = table.dataframe.copy()
            df["PrimaryCG"] = df.apply(expand_pcg_from_suffix, axis=1)
            default_pcgs = utils.single_table(tables, "COMM_GROUPS").dataframe.copy()
            default_pcgs = default_pcgs.loc[
                default_pcgs["DefaultVedaPCG"] == 1,
                ["Region", "TechName", "CommodityGroup"],
            ]
            default_pcgs.rename(columns={"CommodityGroup": "PrimaryCG"}, inplace=True)
            default_pcgs = pd.merge(
                default_pcgs,
                df.loc[df["PrimaryCG"].isna(), df.columns != "PrimaryCG"],
                how="right",
            )
            df = pd.concat([df, default_pcgs])
            df.drop_duplicates(
                subset=[
                    "Sets",
                    "Region",
                    "TechName",
                    "TechDesc",
                    "Tact",
                    "Tcap",
                    "Tslvl",
                    "Vintage",
                ],
                keep="last",
                inplace=True,
            )

            result.append(replace(table, dataframe=df))

    return result


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
        if table.tag != datatypes.Tag.fi_comm:
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
    datayears = (
        tables[datatypes.Tag.fi_t]["Year"]
        .apply(lambda x: x if (x is not str) and x >= 1000 else None)
        .dropna()
    )
    datayears = datayears.drop_duplicates().sort_values()
    tables["DataYear"] = pd.DataFrame({"Year": datayears})

    # Pastyears is the set of all years before ~StartYear
    start_year = tables[datatypes.Tag.start_year]["VALUE"][0]
    pastyears = datayears.where(lambda x: x <= start_year).dropna()
    tables["PastYear"] = pd.DataFrame({"Year": pastyears})

    # Modelyears is the union of pastyears and the representative years of the model (middleyears)
    modelyears = (
        pd.concat(
            [pastyears, tables[datatypes.Tag.time_periods]["M"]], ignore_index=True
        )
        .drop_duplicates()
        .sort_values()
    )
    tables["ModelYear"] = pd.DataFrame({"Year": modelyears})

    return tables


def process_processes(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    result = []
    veda_sets_to_times = {"IMP": "IRE", "EXP": "IRE", "MIN": "IRE"}

    processes_and_sets = pd.DataFrame({"Sets": [], "TechName": []})

    for table in tables:
        if table.tag != datatypes.Tag.fi_process:
            result.append(table)
        else:
            df = table.dataframe.copy()
            processes_and_sets = pd.concat(
                [processes_and_sets, df[["Sets", "TechName"]].ffill()]
            )
            df["Sets"].replace(veda_sets_to_times, inplace=True)
            nrows = df.shape[0]
            if "Vintage" not in table.dataframe.columns.values:
                df["Vintage"] = [None] * nrows
            if "Region" not in table.dataframe.columns.values:
                df.insert(1, "Region", [None] * nrows)
            if "Tslvl" not in table.dataframe.columns.values:
                df.insert(6, "Tslvl", ["ANNUAL"] * nrows)
            result.append(replace(table, dataframe=df))

    veda_process_sets = datatypes.EmbeddedXlTable(
        tag="VedaProcessSets",
        uc_sets={},
        sheetname="",
        range="",
        filename="",
        dataframe=processes_and_sets.loc[
            processes_and_sets["Sets"].isin(veda_sets_to_times.keys())
        ],
    )

    result.append(veda_process_sets)

    return result


def process_topology(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    """
    Create topology
    """

    fit_tables = [t for t in tables if t.tag.startswith(datatypes.Tag.fi_t)]

    columns = ["Region", "TechName", "Comm-IN", "Comm-OUT"]
    topology = pd.DataFrame(columns=columns)

    for fit_table in fit_tables:
        cols = [col for col in columns if col in fit_table.dataframe.columns]
        df = fit_table.dataframe[cols]
        topology = pd.concat([topology, df])

    topology = pd.melt(
        topology,
        id_vars=["Region", "TechName"],
        var_name="IO",
        value_name="CommName",
    )

    topology["TechName"].fillna(method="ffill", inplace=True)
    topology["IO"].replace({"Comm-IN": "IN", "Comm-OUT": "OUT"}, inplace=True)
    topology.dropna(how="any", subset=["TechName", "CommName"], inplace=True)
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
    tables: List[datatypes.EmbeddedXlTable], include_dummy_processes=True
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
            columns=["Sets", "TechName", "TechDesc", "Tact", "Tcap", "PrimaryCG"],
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

        process_data_specs = process_declarations[["TechName", "TechDesc"]].copy()
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


# TODO: should we rename this to something more general, since it takes care of more than tfm_ins?
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
            # TODO: CommGrp is an alias of Other_Indexes. What happens if both are present?
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
                df["CommName"] = df["CSet_CN"]
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

            df.rename(columns={"Cset_CN": "CommName"}, inplace=True)

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
        for key, group in by_tag:
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
        for key, group in by_tag:
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

    for tag in [datatypes.Tag.tfm_upd, datatypes.Tag.tfm_ins, datatypes.Tag.tfm_ins_ts]:
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
                        df = df.merge(matching_commodities)
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
        reg_ts = pd.DataFrame({"Region": regions})
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
            pd.DataFrame({"Region": regions}),
            pd.DataFrame({"TSLVL": ["ANNUAL"], "TS_GROUP": ["ANNUAL"]}),
            how="cross",
        )

        if ncols > 1:
            ts_groups = pd.concat(
                [
                    ts_groups,
                    pd.melt(
                        reg_ts,
                        id_vars=["Region"],
                        var_name="TSLVL",
                        value_name="TS_GROUP",
                    ),
                ]
            )

        # Generate timeslice map
        ts_maps = pd.DataFrame([], columns=["Region", "Parent", "TimesliceMap"])
        if ncols > 2:
            ts_maps = pd.concat(
                [
                    ts_maps,
                    reg_ts.iloc[:, [0, 1, 2]].rename(
                        columns={
                            reg_ts.columns[1]: "Parent",
                            reg_ts.columns[2]: "TimesliceMap",
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
                                reg_ts.columns[1]: "Parent",
                                reg_ts.columns[3]: "TimesliceMap",
                            }
                        ),
                    ]
                )
                ts_maps = pd.concat(
                    [
                        ts_maps,
                        reg_ts.iloc[:, [0, 2, 3]].rename(
                            columns={
                                reg_ts.columns[2]: "Parent",
                                reg_ts.columns[3]: "TimesliceMap",
                            }
                        ),
                    ]
                )

            ts_maps.drop_duplicates(keep="first", inplace=True)
            ts_maps.sort_values(by=list(ts_maps.columns), inplace=True)

        result.append(replace(table, tag="TimeSliceMap", dataframe=DataFrame(ts_maps)))

        result.append(
            replace(table, tag="TimeSlicesGroup", dataframe=DataFrame(ts_groups))
        )

    result = []

    # TODO: Timeslices can differ from region to region
    regions = list(
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


def convert_aliases(input: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    output = {}

    # Ensure TIMES names for all attributes
    replacement_dict = {}
    for k, v in aliases_by_attr.items():
        for alias in v:
            replacement_dict[alias] = k

    for table_type, df in input.items():
        if "Attribute" in df.columns:
            df.replace({"Attribute": replacement_dict}, inplace=True)
        output[table_type] = df

    return output


def expand_rows_parallel(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(expand_rows, tables))
