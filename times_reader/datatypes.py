from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
from pandas.core.frame import DataFrame

# ============================================================================
# ===============                   CLASSES                   ================
# ============================================================================


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


@dataclass
class TimesXlMap:
    """This class defines mapping data objects between the TIMES excel tables
    used by the tool for input and the transformed tables it outputs. The mappings
    are defined in the times_mapping.txt file.

    Attributes:
        times_name      Name of the table in its output form.
        times_cols      Name of the columns that the table will have in its output form.
                        They will be in the header of the output csv files.
        xl_name         Tag for the excel table used as input. You can see a list of all the
                        possible tags in section 2.4 of https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
        xl_cols         Columns from the excel table used as input.
        col_map         A mapping associating times_cols values with col_map values.
        filter_rows     Boolean indicating that only rows with the desired value in the
                        Attribute column should be outputted. If false all rows are outputted.
    """

    times_name: str
    times_cols: List[str]
    xl_name: str
    xl_cols: List[str]
    col_map: Dict[str, str]
    filter_rows: bool


class Tag(str, Enum):
    """
    Enum class to enumerate all the accepted table tags by this program.
    You can see a list of all the possible tags in section 2.4 of
    https://iea-etsap.org/docs/Documentation_for_the_TIMES_Model-Part-IV.pdf
    """

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
