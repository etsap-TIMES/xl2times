import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from importlib import resources
from itertools import chain
from pathlib import PurePath

from loguru import logger
from pandas.core.frame import DataFrame

# ============================================================================
# ===============                   CLASSES                   ================
# ============================================================================


class Tag(str, Enum):
    """Enum class to enumerate all the accepted table tags by this program.

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
    tfm_ava_c = "~TFM_AVA-C"
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
    def has_tag(cls, tag: str) -> bool:
        return tag in cls._value2member_map_


class DataModule(str, Enum):
    """Categorise data into modules based on the file they are coming from."""

    base = "VT_*.*, BY_Trans.*"
    syssettings = "SysSettings.*"
    subres = "SubRES_TMPL/SubRES_*.*"
    sets = "Set*.*"
    lma = "LMA*.*"
    demand = "SuppXLS/Demands/Dem_Alloc+Series.*, SuppXLS/Demands/ScenDem_*.*"
    scen = "SuppXLS/Scen_*.*"
    trade = "SuppXLS/Trades/ScenTrade_*.*"

    @classmethod
    def determine_type(cls, path: str) -> "DataModule | None":
        for data_module in cls:
            if any(
                PurePath(path.lower()).match(pattern.lower().strip())
                for pattern in data_module.value.split(",")
            ):
                return data_module
        return None

    @classmethod
    def module_type(cls, path: str) -> str | None:
        module_type = cls.determine_type(path)
        if module_type:
            return module_type.name
        else:
            return None

    @classmethod
    def submodule(cls, path: str) -> str | None:
        match cls.determine_type(path):
            case DataModule.base | DataModule.subres:
                if PurePath(path.lower()).match("*_trans.*"):
                    return "trans"
                else:
                    return "main"
            case DataModule.trade:
                if PurePath(path.lower()).match("scentrade__trade_links.*"):
                    return "main"
                else:
                    return "trans"
            case DataModule.demand:
                if PurePath(path.lower()).match("dem_alloc+series.*"):
                    return "main"
                else:
                    return "trans"
            case None:
                return None
            case _:
                return "main"

    @classmethod
    def module_name(cls, path: str) -> str | None:
        module_type = cls.determine_type(path)
        match module_type:
            case (
                DataModule.base
                | DataModule.sets
                | DataModule.lma
                | DataModule.demand
                | DataModule.trade
                | DataModule.syssettings
            ):
                return module_type.name.upper()
            case DataModule.subres:
                return re.sub(
                    "^SUBRES_", "", re.sub("_TRANS$", "", PurePath(path).stem.upper())
                )
            case DataModule.scen:
                return re.sub("^SCEN_", "", PurePath(path).stem.upper())
            case None:
                return None


@dataclass
class EmbeddedXlTable:
    """A table object: a pandas dataframe wrapped with some metadata.

    Attributes
    ----------
    tag
        Table tag associated with this table in the excel file used as input.
    defaults
        Defaults for the table that are separated by a colon from the tag.
    uc_sets
        User constrained tables are declared with tags which indicate their type and domain of coverage.
    sheetname
        Name of the excel worksheet where this table was extracted from.
    range
        Range of rows and columns that contained this table in the original excel worksheet.
    filename
        Name of the original excel file where this table was extracted from.
    dataframe
        Pandas dataframe containing the values of the table.
    """

    tag: str
    uc_sets: dict[str, str]
    sheetname: str
    range: str
    filename: str
    dataframe: DataFrame
    defaults: str | None = field(default=None)

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
    """The mapping between the TIMES excel tables used by the tool for input and the
    transformed tables it outputs. The mappings are defined in the
    `times_mapping.txt` and `times-info.json` files.

    Attributes
    ----------
    times_name
        Name of the table in its output form.
    times_cols
        Name of the columns that the table will have in its output form. They will be
        in the header of the output csv files.
    xl_name
        Tag for the Excel table used as input. You can see a list of all the possible
        tags in section 2.4 of
        https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
    xl_cols
        Columns from the Excel table used as input.
    col_map
        A mapping from Excel column names to Times column names.
    filter_rows
        A map from column name to value to filter rows to. If `{}`, all rows are
        outputted. E.g., `{'Attribute': 'COM_ELAST'}`
    """

    times_name: str
    times_cols: list[str]
    xl_name: str  # TODO once we move away from times_mapping.txt, make this type Tag
    xl_cols: list[str]
    col_map: dict[str, str]
    filter_rows: dict[str, str]


@dataclass
class TimesModel:
    """A class containing all the information about the processed TIMES model."""

    internal_regions: set[str] = field(default_factory=set)
    all_regions: set[str] = field(default_factory=set)
    processes: DataFrame = field(default_factory=DataFrame)
    commodities: DataFrame = field(default_factory=DataFrame)
    commodity_groups: DataFrame = field(default_factory=DataFrame)
    topology: DataFrame = field(default_factory=DataFrame)
    implied_topology: DataFrame = field(default_factory=DataFrame)
    trade: DataFrame = field(default_factory=DataFrame)
    attributes: DataFrame = field(default_factory=DataFrame)
    user_constraints: DataFrame = field(default_factory=DataFrame)
    uc_attributes: DataFrame = field(default_factory=DataFrame)
    ts_tslvl: DataFrame = field(default_factory=DataFrame)
    ts_map: DataFrame = field(default_factory=DataFrame)
    time_periods: DataFrame = field(default_factory=DataFrame)
    units: DataFrame = field(default_factory=DataFrame)
    start_year: int = field(default_factory=int)
    files: list[str] = field(default_factory=list)
    data_modules: list[str] = field(default_factory=list)
    custom_sets: DataFrame = field(default_factory=DataFrame)

    @property
    def external_regions(self) -> set[str]:
        return self.all_regions.difference(self.internal_regions)

    @property
    def data_years(self) -> set[int]:
        """data_years are years for which there is data specified."""
        data_years = set()
        for attributes in [self.attributes, self.uc_attributes]:
            if not attributes.empty:
                # Index of the year column with non-empty values
                # index = attributes["year"] != ""
                # data_years.update(attributes["year"][index].astype(int).values)
                # TODO: Ensure that non-parseble vals don't get this far
                int_years = attributes["year"].astype(
                    int, errors="ignore"
                )  # leave non-parseable vals alone
                int_years = [
                    y for y in int_years if isinstance(y, int)
                ]  # remove non-parseable years
                data_years.update(int_years)
        # Remove interpolation rules before return
        return {y for y in data_years if y >= 1000}

    @property
    def past_years(self) -> set[int]:
        """past_years is the set of all years up to and including start year for which
        past investments are specified.

        TIMES populates PASTYEAR on its own, so this could be dropped in the future.
        """
        i = self.attributes["attribute"] == "NCAP_PASTI"
        years = set(self.attributes["year"][i].astype(int).values)
        return {year for year in years if year <= self.start_year}

    @property
    def model_years(self) -> set[int]:
        """model_years is the union of past_years and the representative years of the
        model (milestone years).

        TIMES populates MODLYEAR on its own, so this could be dropped in the future.
        """
        return self.past_years | set(self.time_periods["m"].values)

    # TODO: Invalidate and recompute the below property when self.topology changes.
    @cached_property
    def veda_cgs(self) -> dict[tuple[str, str, str], str]:
        """A dictionary mapping commodities to their Veda commodity groups."""
        cols = ["region", "process", "commodity", "csets"]
        # Exclude auxillary flows
        index = self.topology["io"].isin({"IN", "OUT"})
        veda_cgs = self.topology[cols + ["io"]][index]
        veda_cgs = veda_cgs.drop_duplicates(subset=cols, keep="last")
        veda_cgs["veda_cg"] = veda_cgs["csets"] + veda_cgs["io"].str[:1]
        veda_cgs = veda_cgs.set_index(["region", "process", "commodity"])[
            "veda_cg"
        ].to_dict()
        return veda_cgs


class Config:
    """Encapsulates all configuration options for a run of the tool, including the
    mapping betwen excel tables and output tables, categories of tables, etc.
    """

    times_xl_maps: list[TimesXlMap]
    dd_table_order: Iterable[str]
    all_attributes: set[str]
    attr_aliases: set[str]
    # Attribute by type (commodity, process, flow)
    attr_by_type: dict[str, set[str]]
    # Attribute by index (region, year, etc.)
    attr_by_index: dict[str, set[str]]
    # For each tag, this dictionary maps each column alias to the normalized name
    column_aliases: dict[Tag, dict[str, str]]
    # For each tag, this dictionary maps each column name to its default value
    column_default_value: dict[Tag, dict[str, str]]
    # For each tag, this dictionary specifies comment row symbols by column name
    row_comment_chars: dict[Tag, dict[str, list]]
    # List of tags for which empty tables should be discarded
    discard_if_empty: Iterable[Tag]
    veda_attr_defaults: dict[str, dict[str, list]]
    # Known columns for each tag
    known_columns: dict[Tag, set[str]]
    # Query columns for each tag
    query_columns: dict[Tag, set[str]]
    # Columns that may contain lists for each tag
    lists_columns: dict[Tag, set[str]]
    # Required columns for each tag
    required_columns: dict[Tag, set[str]]
    # Names of regions to include in the model; if empty, all regions are included.
    filter_regions: set[str]
    times_sets: dict[str, list[str]]
    # Switch to prevent overwriting of I/E settings in BASE and SubRES
    ie_override_in_syssettings: bool = False
    # Switch to include dummy imports in the model
    include_dummy_imports: bool

    def __init__(
        self,
        mapping_file: str,
        times_info_file: str,
        times_sets_file: str,
        veda_tags_file: str,
        veda_attr_defaults_file: str,
        regions: str,
        include_dummy_imports: bool,
    ):
        self.times_xl_maps = Config._read_mappings(mapping_file)
        (
            self.dd_table_order,
            self.all_attributes,
            self.attr_by_type,
            self.attr_by_index,
            param_mappings,
        ) = Config._process_times_info(times_info_file)
        self.times_sets = Config._read_times_sets(times_sets_file)
        (
            self.column_aliases,
            self.column_default_value,
            self.row_comment_chars,
            self.discard_if_empty,
            self.query_columns,
            self.lists_columns,
            self.known_columns,
            self.required_columns,
            self.add_columns,
            self.forward_fill_cols,
        ) = Config._read_veda_tags_info(veda_tags_file)
        self.veda_attr_defaults, self.attr_aliases = Config._read_veda_attr_defaults(
            veda_attr_defaults_file, param_mappings
        )
        # Migration in progress: use parameter mappings from times_info_file for now
        name_to_map = {m.times_name: m for m in self.times_xl_maps}
        for m in param_mappings:
            name_to_map[m.times_name] = m
        self.times_xl_maps = list(name_to_map.values())
        self.filter_regions = Config._read_regions_filter(regions)
        self.include_dummy_imports = include_dummy_imports

    @staticmethod
    def _read_times_sets(
        times_sets_file: str,
    ) -> dict[str, list[str]]:
        # Read times_sets_file
        with resources.open_text("xl2times.config", times_sets_file) as f:
            times_sets = json.load(f)

        return times_sets

    @staticmethod
    def _process_times_info(
        times_info_file: str,
    ) -> tuple[
        Iterable[str],
        set[str],
        dict[str, set[str]],
        dict[str, set[str]],
        list[TimesXlMap],
    ]:
        # Read times_info_file and compute dd_table_order:
        # We output tables in order by categories: set, subset, subsubset, md-set, and parameter
        with resources.open_text("xl2times.config", times_info_file) as f:
            table_info = json.load(f)
        categories = ["set", "subset", "subsubset", "md-set", "parameter"]
        cat_to_tables = defaultdict(list)
        for item in table_info:
            cat_to_tables[item["gams-cat"]].append(item["name"])
        unknown_cats = {item["gams-cat"] for item in table_info} - set(categories)
        if unknown_cats:
            logger.warning(f"Unknown categories in times-info.json: {unknown_cats}")
        dd_table_order = chain.from_iterable(
            sorted(cat_to_tables[c]) for c in categories
        )

        # Compute the set of all attributes, i.e. all entities with category = parameter
        attributes = {
            item["name"].upper()
            for item in table_info
            if item["gams-cat"] == "parameter"
        }

        # Determine the attributes by type
        attr_by_type = dict()

        attr_type_conditions = {
            "commodity": {"commodity": True, "process": False},
            "process": {"process": True, "commodity": False},
            "flow": {"process": True, "commodity": True},
        }

        for attr_type, conditions in attr_type_conditions.items():
            attr_by_type[attr_type] = {
                attr["name"]
                for attr in table_info
                if all(
                    (index in attr["mapping"]) is is_present
                    for index, is_present in conditions.items()
                )
            }

        # Categorise attributes by index (region, year, etc.)
        attr_by_index = dict()
        # Set of all indexes in the mapping
        all_indexes = set()
        for attr in table_info:
            if attr["gams-cat"] == "parameter":
                all_indexes = all_indexes.union(attr["mapping"])
        for index in all_indexes:
            list_of_attrs = set()
            for attr in table_info:
                if attr["gams-cat"] != "parameter":
                    continue
                if index in attr["mapping"]:
                    list_of_attrs.add(attr["name"])
            attr_by_index[index] = list_of_attrs

        # Compute the mapping for attributes / parameters:
        def create_mapping(entity):
            assert entity["gams-cat"] == "parameter"
            times_cols = entity["indexes"] + ["VALUE"]
            xl_cols = entity["mapping"] + ["value"]  # TODO map in json
            col_map = dict(zip(times_cols, xl_cols))
            # If tag starts with UC, then the data is in UCAttributes, else Attributes
            xl_name = (
                "UCAttributes"
                if entity["name"].lower().startswith("uc")
                else "Attributes"
            )
            return TimesXlMap(
                times_name=entity["name"],
                times_cols=times_cols,
                xl_name=xl_name,
                xl_cols=xl_cols,
                col_map=col_map,
                filter_rows={"attribute": entity["name"]},  # TODO value:1?
            )

        param_mappings = [
            create_mapping(x)
            for x in table_info
            if x["gams-cat"] == "parameter"
            and "type" not in x  # TODO Generalise derived parameters?
        ]

        return dd_table_order, attributes, attr_by_type, attr_by_index, param_mappings

    @staticmethod
    def _read_mappings(filename: str) -> list[TimesXlMap]:
        """Function to load mappings from a text file between the excel sheets we use as
        input and the tables we give as output.

        The mappings have the following structure:

        OUTPUT_TABLE[DATAYEAR,VALUE] = ~TimePeriods(Year,B)

        where OUTPUT_TABLE is the name of the table we output and it includes a list of the
        different fields or column names it includes. On the other side, TimePeriods is the type
        of table that we will use as input to produce that table, and the arguments are the
        columns of that table to use to produce the output. The last argument can be of the
        form `Attribute:ATTRNAME` which means the output will be filtered to only the rows of
        the input table that have `ATTRNAME` in the Attribute column.

        The mappings are loaded into TimesXlMap objects. See the description of that class for more
        information of the different fields they contain.

        Parameters
        ----------
        filename
            Name of the text file containing the mappings.

        Returns
        -------
        list[TimesXlMap]
            List of mappings in TimesXlMap format.
        """
        mappings = []
        dropped = []
        with resources.open_text("xl2times.config", filename) as file:
            while True:
                line = file.readline().rstrip()
                if line == "":
                    break
                (times, xl) = line.split(" = ")
                (times_name, times_cols_str) = list(
                    filter(None, re.split(r"\[|\]", times))
                )
                (xl_name, xl_cols_str) = list(filter(None, re.split(r"\(|\)", xl)))
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
            logger.info(f"Dropping {len(dropped)} mappings that are not yet complete")
        return mappings

    @staticmethod
    def _read_veda_tags_info(
        veda_tags_file: str,
    ) -> tuple[
        dict[Tag, dict[str, str]],
        dict[Tag, dict[str, str]],
        dict[Tag, dict[str, list]],
        Iterable[Tag],
        dict[Tag, set[str]],
        dict[Tag, set[str]],
        dict[Tag, set[str]],
        dict[Tag, set[str]],
        dict[Tag, set[str]],
        dict[Tag, set[str]],
    ]:
        def to_tag(s: str) -> Tag:
            # The file stores the tag name in lowercase, and without the ~
            return Tag("~" + s.upper())

        # Read veda_tags_file
        with resources.open_text("xl2times.config", veda_tags_file) as f:
            veda_tags_info = json.load(f)

        # Check that all the tags we use are present in veda_tags_file
        tags = {to_tag(tag_info["tag_name"]) for tag_info in veda_tags_info}
        for tag in Tag:
            if tag not in tags:
                logger.info(
                    f"WARNING: datatypes.Tag has an unknown Tag {tag} not in {veda_tags_file}"
                )

        valid_column_names = {}
        column_default_value = {}
        row_comment_chars = {}
        discard_if_empty = []
        query_cols = defaultdict(set)
        lists_cols = defaultdict(set)
        known_cols = defaultdict(set)
        required_cols = defaultdict(set)
        add_cols = defaultdict(set)
        forward_fill_cols = defaultdict(set)

        for tag_info in veda_tags_info:
            tag_name = to_tag(tag_info["tag_name"])
            if "valid_fields" in tag_info:
                discard_if_empty.append(tag_name)
                valid_column_names[tag_name] = {}
                column_default_value[tag_name] = {}
                row_comment_chars[tag_name] = {}
                # Process column aliases and comment chars:
                for valid_field in tag_info["valid_fields"]:
                    valid_field_names = valid_field.get("aliases", list())
                    if (
                        "use_name" in valid_field
                        and valid_field["use_name"] != valid_field["name"]
                    ):
                        field_name = valid_field["use_name"]
                        valid_field_names.append(valid_field["name"])
                    else:
                        field_name = valid_field["name"]

                    if "default_to" in valid_field:
                        column_default_value[tag_name][field_name] = valid_field[
                            "default_to"
                        ]

                    if valid_field.get("query_field", False):
                        query_cols[tag_name].add(field_name)

                    if valid_field.get("comma-separated-list", False):
                        lists_cols[tag_name].add(field_name)

                    if valid_field.get("add_if_absent", False):
                        add_cols[tag_name].add(field_name)

                    if valid_field.get("remove_any_row_if_absent", False):
                        required_cols[tag_name].add(field_name)

                    if valid_field.get("inherit_above", False):
                        forward_fill_cols[tag_name].add(field_name)

                    known_cols[tag_name].add(field_name)

                    for valid_field_name in valid_field_names:
                        valid_column_names[tag_name][valid_field_name] = field_name

                    row_comment_chars[tag_name][field_name] = valid_field.get(
                        "row_ignore_symbol", list()
                    )

            # TODO: Account for differences in valid field names with base_tag
            if "base_tag" in tag_info:
                base_tag = to_tag(tag_info["base_tag"])
                mod_type = tag_info["mod_type"]
                if base_tag in valid_column_names:
                    valid_column_names[tag_name] = valid_column_names[base_tag]
                    discard_if_empty.append(tag_name)
                if base_tag in column_default_value:
                    column_default_value[tag_name] = column_default_value[base_tag]
                if base_tag in row_comment_chars:
                    row_comment_chars[tag_name] = row_comment_chars[base_tag]
                if base_tag in query_cols:
                    query_cols[tag_name] = query_cols[base_tag].difference({mod_type})
                if base_tag in lists_cols:
                    lists_cols[tag_name] = lists_cols[base_tag].difference({mod_type})
                if base_tag in known_cols:
                    known_cols[tag_name] = known_cols[base_tag].difference({mod_type})
                if base_tag in add_cols:
                    add_cols[tag_name] = add_cols[base_tag].difference({mod_type})
                if base_tag in required_cols:
                    required_cols[tag_name] = required_cols[base_tag].difference(
                        {mod_type}
                    )
                if base_tag in forward_fill_cols:
                    forward_fill_cols[tag_name] = forward_fill_cols[
                        base_tag
                    ].difference({mod_type})

        return (
            valid_column_names,
            column_default_value,
            row_comment_chars,
            discard_if_empty,
            query_cols,
            lists_cols,
            known_cols,
            required_cols,
            add_cols,
            forward_fill_cols,
        )

    @staticmethod
    def _read_veda_attr_defaults(
        veda_attr_defaults_file: str, attr_mappings: list[TimesXlMap]
    ) -> tuple[dict[str, dict[str, list]], set[str]]:
        # Read veda_tags_file
        with resources.open_text("xl2times.config", veda_attr_defaults_file) as f:
            defaults = json.load(f)

        veda_attr_defaults = {
            "aliases": defaultdict(list),
            "commodity": {},
            "other_indexes": {},
            "cg": {},
            "limtype": {"FX": [], "LO": [], "UP": []},
            "tslvl": {"DAYNITE": [], "ANNUAL": []},
            "year2": {"EOH": []},
        }

        group_defaults = {"limtype", "tslvl", "year2"}
        individual_defaults = {"commodity", "other_indexes", "cg"}

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

                for item in group_defaults.intersection(attr_defaults.keys()):
                    group_value = attr_defaults[item]
                    veda_attr_defaults[item][group_value].append(attr)

                for item in individual_defaults.intersection(attr_defaults.keys()):
                    veda_attr_defaults[item][attr] = attr_defaults[item]

        # Specify default values for the attributes that are not defined in the file
        attr_with_cg = {
            attr_mapping.times_name
            for attr_mapping in attr_mappings
            if "cg" in set(attr_mapping.xl_cols)
        }
        for attr in attr_with_cg.difference(veda_attr_defaults["cg"].keys()):
            veda_attr_defaults["cg"][attr] = ["other_indexes"]

        return veda_attr_defaults, attr_aliases

    @staticmethod
    def _read_regions_filter(regions_list: str) -> set[str]:
        if regions_list == "":
            return set()
        else:
            return set(regions_list.strip(" ").upper().split(sep=","))
