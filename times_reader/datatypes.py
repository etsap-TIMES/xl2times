from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
from pandas.core.frame import DataFrame

# ============================================================================
# ===============                   CLASSES                   ================
# ============================================================================


class Col(str):
    """A subclass of string to be used as column names for our DataFrames.
    This class normalizes case and whitespace of the supplied column names.
    """

    def __new__(cls, content):
        if isinstance(content, cls):
            return super().__new__(cls, str(content))
        if isinstance(content, str):
            return super().__new__(cls, str(content).strip().upper())

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, Col):
            return super().__eq__(other)
        if isinstance(other, str):
            # raise ValueError("Cannot compare Col and str -- use only Col!")
            return self == Col(other)
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self) -> str:
        return f"Col({super().__str__()})"

    def __str__(self) -> str:
        return super().__str__()


def Cols(col_names: List[str]) -> List[Col]:
    return [Col(x) for x in col_names]


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
    times_cols: List[Col]
    xl_name: str
    xl_cols: List[Col]
    col_map: Dict[Col, Col]
    filter_rows: Dict[Col, str]


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
    # unitconversion = "~UNITCONVERSION"

    @classmethod
    def has_tag(cls, tag):
        return tag in cls._value2member_map_
