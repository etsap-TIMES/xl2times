from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from itertools import chain
import json
import re
from typing import Dict, Iterable, List, Set, Tuple
from enum import Enum
from pandas.core.frame import DataFrame

# ============================================================================
# ===============                   CLASSES                   ================
# ============================================================================


class Tag(str, Enum):
    """
    Enum class to enumerate all the accepted table tags by this program.
    You can see a list of all the possible tags in section 2.4 of
    https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
    """

    active_p_def = "~ACTIVEPDEF"
    book_regions_map = "~BOOKREGIONS_MAP"
    comagg = "~COMAGG"
    comemi = "~COMEMI"
    currencies = "~CURRENCIES"
    defaultyear = "~DEFAULTYEAR"
    def_units = "~DEFUNITS"
    endyear = "~ENDYEAR"
    fi_comm = "~FI_COMM"
    fi_process = "~FI_PROCESS"
    fi_t = "~FI_T"
    milestoneyears = "~MILESTONEYEARS"
    start_year = "~STARTYEAR"
    tfm_ava = "~TFM_AVA"
    tfm_comgrp = "~TFM_COMGRP"
    tfm_csets = "~TFM_CSETS"
    tfm_dins = "~TFM_DINS"
    tfm_dins_at = "~TFM_DINS-AT"
    tfm_dins_ts = "~TFM_DINS-TS"
    tfm_dins_tsl = "~TFM_DINS-TSL"
    tfm_fill = "~TFM_FILL"
    tfm_fill_r = "~TFM_FILL-R"
    tfm_ins = "~TFM_INS"
    tfm_ins_at = "~TFM_INS-AT"
    tfm_ins_ts = "~TFM_INS-TS"
    tfm_ins_tsl = "~TFM_INS-TSL"
    tfm_ins_txt = "~TFM_INS-TXT"
    tfm_mig = "~TFM_MIG"
    tfm_psets = "~TFM_PSETS"
    tfm_topdins = "~TFM_TOPDINS"
    tfm_topins = "~TFM_TOPINS"
    tfm_upd = "~TFM_UPD"
    tfm_upd_at = "~TFM_UPD-AT"
    tfm_upd_ts = "~TFM_UPD-TS"
    time_periods = "~TIMEPERIODS"
    time_slices = "~TIMESLICES"
    tradelinks = "~TRADELINKS"
    tradelinks_dins = "~TRADELINKS_DINS"
    uc_sets = "~UC_SETS"
    uc_t = "~UC_T"
    # This is used by Veda for unit conversion when displaying results
    unitconversion = "~UNITCONVERSION"

    @classmethod
    def has_tag(cls, tag):
        return tag in cls._value2member_map_


@dataclass
class EmbeddedXlTable:
    """This class defines a table object as a pandas dataframe wrapped with some metadata.

    Attributes:
        tag         Table tag associated with this table in the excel file used as input. You can see a list of all the
                    possible tags in section 2.4 of https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
        uc_sets     User constrained tables are declared with tags which indicate their type and domain of coverage. This variable contains these two values.
                    See section 2.4.7 in https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
        sheetname   Name of the excel worksheet where this table was extracted from.
        range       Range of rows and columns that contained this table in the original excel worksheet.
        filename    Name of the original excel file where this table was extracted from.
        dataframe   Pandas dataframe containing the values of the table.
    """

    tag: str
    uc_sets: Dict[str, str]
    sheetname: str
    range: str
    filename: str
    dataframe: DataFrame

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, EmbeddedXlTable):
            return False
        return (
            self.tag == o.tag
            and self.uc_sets == o.uc_sets
            and self.range == o.range
            and self.filename == o.filename
            and self.dataframe.shape == o.dataframe.shape
            and (
                len(self.dataframe) == 0  # Empty tables don't affect our output
                or self.dataframe.sort_index(axis=1).equals(
                    o.dataframe.sort_index(axis=1)
                )
            )
        )

    def __str__(self) -> str:
        df_str = self.dataframe.to_csv(index=False, lineterminator="\n")
        return f"EmbeddedXlTable(tag={self.tag}, uc_sets={self.uc_sets}, sheetname={self.sheetname}, range={self.range}, filename={self.filename}, dataframe=\n{df_str}{self.dataframe.shape})"


@dataclass
class TimesXlMap:
    """This class defines mapping data objects between the TIMES excel tables
    used by the tool for input and the transformed tables it outputs. The mappings
    are defined in the times_mapping.txt file.

    Attributes:
        times_name      Name of the table in its output form.
        times_cols      Name of the columns that the table will have in its output form.
                        They will be in the header of the output csv files.
        xl_name         Tag for the Excel table used as input. You can see a list of all the
                        possible tags in section 2.4 of https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
        xl_cols         Columns from the Excel table used as input.
        col_map         A mapping from Excel column names to Times column names.
        filter_rows     A map from column name to value to filter rows to. If {}, all
                        rows are outputted. E.g., {'Attribute': 'COM_ELAST'}
    """

    times_name: str
    times_cols: List[str]
    xl_name: str
    xl_cols: List[str]
    col_map: Dict[str, str]
    filter_rows: Dict[str, str]


class Config:
    """Encapsulates all configuration options for a run of the tool, including
    the mapping betwen excel tables and output tables, categories of tables, etc.
    """

    times_xl_maps: List[TimesXlMap]
    dd_table_order: Iterable[str]
    all_attributes: Set[str]
    attr_aliases: Set[str]
    # For each tag, this dictionary maps each column alias to the normalized name
    column_aliases: Dict[Tag, Dict[str, str]]
    # For each tag, this dictionary specifies comment row symbols by column name
    row_comment_chars: Dict[Tag, Dict[str, list]]
    # List of tags for which empty tables should be discarded
    discard_if_empty: Iterable[Tag]
    veda_attr_defaults: Dict[str, Dict[str, list]]

    def __init__(
        self,
        mapping_file: str,
        times_info_file: str,
        veda_tags_file: str,
        veda_attr_defaults_file: str,
    ):
        self.times_xl_maps = Config._read_mappings(mapping_file)
        self.dd_table_order, self.all_attributes = Config._process_times_info(
            times_info_file
        )
        (
            self.column_aliases,
            self.row_comment_chars,
            self.discard_if_empty,
        ) = Config._read_veda_tags_info(veda_tags_file)
        self.veda_attr_defaults, self.attr_aliases = Config._read_veda_attr_defaults(
            veda_attr_defaults_file
        )

    @staticmethod
    def _process_times_info(times_info_file: str) -> Tuple[Iterable[str], Set[str]]:
        # Read times_info_file and compute dd_table_order:
        # We output tables in order by categories: set, subset, subsubset, md-set, and parameter
        with resources.open_text("times_reader.config", times_info_file) as f:
            table_info = json.load(f)
        categories = ["set", "subset", "subsubset", "md-set", "parameter"]
        cat_to_tables = defaultdict(list)
        for item in table_info:
            cat_to_tables[item["gams-cat"]].append(item["name"])
        unknown_cats = {item["gams-cat"] for item in table_info} - set(categories)
        if unknown_cats:
            print(f"WARNING: Unknown categories in times-info.json: {unknown_cats}")
        dd_table_order = chain.from_iterable(
            (sorted(cat_to_tables[c]) for c in categories)
        )

        # Compute the set of all attributes, i.e. all entities with category = parameter
        attributes = {
            item["name"].lower()
            for item in table_info
            if item["gams-cat"] == "parameter"
        }
        return dd_table_order, attributes

    @staticmethod
    def _read_mappings(filename: str) -> List[TimesXlMap]:
        """
        Function to load mappings from a text file between the excel sheets we use as input and
        the tables we give as output. The mappings have the following structure:

        OUTPUT_TABLE[DATAYEAR,VALUE] = ~TimePeriods(Year,B)

        where OUTPUT_TABLE is the name of the table we output and it includes a list of the
        different fields or column names it includes. On the other side, TimePeriods is the type
        of table that we will use as input to produce that table, and the arguments are the
        columns of that table to use to produce the output. The last argument can be of the
        form `Attribute:ATTRNAME` which means the output will be filtered to only the rows of
        the input table that have `ATTRNAME` in the Attribute column.

        The mappings are loaded into TimesXlMap objects. See the description of that class for more
        information of the different fields they contain.

        :param filename:        Name of the text file containing the mappings.
        :return:                List of mappings in TimesXlMap format.
        """
        mappings = []
        dropped = []
        with resources.open_text("times_reader.config", filename) as file:
            while True:
                line = file.readline().rstrip()
                if line == "":
                    break
                (times, xl) = line.split(" = ")
                (times_name, times_cols_str) = list(
                    filter(None, re.split("\[|\]", times))
                )
                (xl_name, xl_cols_str) = list(filter(None, re.split("\(|\)", xl)))
                times_cols = times_cols_str.split(",")
                xl_cols = xl_cols_str.split(",")
                filter_rows = {}
                for i, s in enumerate(xl_cols):
                    if ":" in s:
                        [col_name, col_val] = s.split(":")
                        filter_rows[col_name.strip().lower()] = col_val.strip()
                xl_cols = [s.lower() for s in xl_cols if ":" not in s]

                # TODO remove: Filter out mappings that are not yet finished
                if xl_name != "~TODO" and not any(
                    c.startswith("TODO") for c in xl_cols
                ):
                    col_map = {}
                    assert len(times_cols) <= len(xl_cols)
                    for index, value in enumerate(times_cols):
                        col_map[value] = xl_cols[index]
                    # Uppercase and validate tags:
                    if xl_name.startswith("~"):
                        xl_name = xl_name.upper()
                    entry = TimesXlMap(
                        times_name=times_name,
                        times_cols=times_cols,
                        xl_name=xl_name,
                        xl_cols=xl_cols,
                        col_map=col_map,
                        filter_rows=filter_rows,
                    )
                    mappings.append(entry)
                else:
                    dropped.append(line)

        if len(dropped) > 0:
            print(
                f"WARNING: Dropping {len(dropped)} mappings that are not yet complete"
            )
        return mappings

    @staticmethod
    def _read_veda_tags_info(
        veda_tags_file: str,
    ) -> Tuple[Dict[Tag, Dict[str, str]], Dict[Tag, Dict[str, list]], Iterable[Tag]]:
        def to_tag(s: str) -> Tag:
            # The file stores the tag name in lowercase, and without the ~
            return Tag("~" + s.upper())

        # Read veda_tags_file
        with resources.open_text("times_reader.config", veda_tags_file) as f:
            veda_tags_info = json.load(f)

        # Check that all the tags we use are present in veda_tags_file
        tags = {to_tag(tag_info["tag_name"]) for tag_info in veda_tags_info}
        for tag in Tag:
            if tag not in tags:
                print(
                    f"WARNING: datatypes.Tag has an unknown Tag {tag} not in {veda_tags_file}"
                )

        valid_column_names = {}
        row_comment_chars = {}
        discard_if_empty = []

        for tag_info in veda_tags_info:
            tag_name = to_tag(tag_info["tag_name"])
            if "valid_fields" in tag_info:
                discard_if_empty.append(tag_name)

                valid_column_names[tag_name] = {}
                row_comment_chars[tag_name] = {}
                # Process column aliases and comment chars:
                for valid_field in tag_info["valid_fields"]:
                    valid_field_names = valid_field["aliases"]
                    if (
                        "use_name" in valid_field
                        and valid_field["use_name"] != valid_field["name"]
                    ):
                        field_name = valid_field["use_name"]
                        valid_field_names.append(valid_field["name"])
                    else:
                        field_name = valid_field["name"]

                    for valid_field_name in valid_field_names:
                        valid_column_names[tag_name][valid_field_name] = field_name
                        row_comment_chars[tag_name][field_name] = valid_field[
                            "row_ignore_symbol"
                        ]

            # TODO: Account for differences in valid field names with base_tag
            if "base_tag" in tag_info:
                base_tag = to_tag(tag_info["base_tag"])
                if base_tag in valid_column_names:
                    valid_column_names[tag_name] = valid_column_names[base_tag]
                    discard_if_empty.append(tag_name)
                if base_tag in row_comment_chars:
                    row_comment_chars[tag_name] = row_comment_chars[base_tag]

        return valid_column_names, row_comment_chars, discard_if_empty

    @staticmethod
    def _read_veda_attr_defaults(
        veda_attr_defaults_file: str,
    ) -> Tuple[Dict[str, Dict[str, list]], Set[str]]:
        # Read veda_tags_file
        with resources.open_text("times_reader.config", veda_attr_defaults_file) as f:
            defaults = json.load(f)

        veda_attr_defaults = {
            "aliases": defaultdict(list),
            "commodity": {},
            "limtype": {"FX": [], "LO": [], "UP": []},
            "tslvl": {"DAYNITE": [], "ANNUAL": []},
        }

        attr_aliases = {
            attr for attr in defaults if "times-attribute" in defaults[attr]
        }

        for attr, attr_info in defaults.items():
            # Populate aliases by attribute dictionary
            if "times-attribute" in attr_info:
                times_attr = attr_info["times-attribute"]
                veda_attr_defaults["aliases"][times_attr].append(attr)

            if "defaults" in attr_info:
                attr_defaults = attr_info["defaults"]

                if "commodity" in attr_defaults:
                    veda_attr_defaults["commodity"][attr] = attr_defaults["commodity"]

                if "limtype" in attr_defaults:
                    limtype = attr_defaults["limtype"]
                    veda_attr_defaults["limtype"][limtype].append(attr)

                if "ts-level" in attr_defaults:
                    tslvl = attr_defaults["ts-level"]
                    veda_attr_defaults["tslvl"][tslvl].append(attr)

        return veda_attr_defaults, attr_aliases
