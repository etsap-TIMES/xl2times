import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import PurePath

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
    drvr_allocation = "~DRVR_ALLOCATION"
    drvr_table = "~DRVR_TABLE"
    defaultyear = "~DEFAULTYEAR"
    def_units = "~DEFUNITS"
    endyear = "~ENDYEAR"
    fi_comm = "~FI_COMM"
    fi_process = "~FI_PROCESS"
    fi_t = "~FI_T"
    milestoneyears = "~MILESTONEYEARS"
    series = "~SERIES"
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
    custom_psets: DataFrame = field(default_factory=DataFrame)
    user_psets: DataFrame = field(default_factory=DataFrame)
    user_csets: DataFrame = field(default_factory=DataFrame)

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
