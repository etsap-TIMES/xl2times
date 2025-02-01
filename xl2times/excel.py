import re
import time

import numpy
from loguru import logger
from openpyxl import load_workbook
from openpyxl.worksheet.cell_range import CellRange
from pandas.core.frame import DataFrame

from . import datatypes, utils


def extract_tables(filename: str) -> list[datatypes.EmbeddedXlTable]:
    """Run the extract_table function on each individual table in each worksheet of the
    given excel file.

    Parameters
    ----------
    filename
        Path to the excel file we will extract tables from.

    Returns
    -------
    list[datatypes.EmbeddedXlTable]
        List of table objects in EmbeddedXlTable format.
    """
    start_time = time.time()

    workbook = load_workbook(filename=filename, data_only=True)

    tables = []
    for sheet in workbook.worksheets:
        # Creating dataframe with dtype=object solves problems with ints being cast to floats
        # https://stackoverflow.com/questions/40251948/stop-pandas-from-converting-int-to-float-due-to-an-insertion-in-another-column
        df = DataFrame(sheet.values, dtype=object)
        active_uc_sets = {}
        formatted_uc_sets = {}
        sheet_tables = []

        for row_index, row in df.iterrows():
            for colname in df.columns:
                value = str(row[colname])
                if value.startswith("~"):
                    match = re.match(
                        f"{datatypes.Tag.uc_sets.value}:(.*)", value, re.IGNORECASE
                    )
                    if match:
                        updated, active_uc_sets = _update_uc_sets(
                            active_uc_sets, match.group(1)
                        )
                        if updated:
                            formatted_uc_sets = {
                                k: v
                                for type, values in active_uc_sets.items()
                                for k, v in values.items()
                            }
                        else:
                            logger.warning(
                                f"Malformed {match.group(0)} in {sheet.title}, {filename}"
                            )
                    else:
                        col_index = df.columns.get_loc(colname)
                        sheet_tables.append(
                            extract_table(
                                row_index,
                                col_index,
                                formatted_uc_sets,
                                df,
                                sheet.title,
                                filename,
                            )
                        )

        tables += sheet_tables

    end_time = time.time()
    if end_time - start_time > 2:
        logger.info(f"Loaded {filename} in {end_time-start_time:.2f} seconds")

    return tables


def extract_table(
    tag_row: int,
    tag_col: int,
    uc_sets: dict[str, str],
    df: DataFrame,
    sheetname: str,
    filename: str,
) -> datatypes.EmbeddedXlTable:
    """For each individual table tag found in a worksheet, this function aims to extract
    the associated table.

    We recognise several types of tables:

    - Single cell tables: Tables with only one value, either below or to the right of
      the table tag. We interpret these as a single data item with
      a column name VALUE.
    - Multiple cell tables: Tables with multiple values, possibly extending accross
      several rows and columns. We delimitate them using empty
      spaces around them and the column names are determined by the
      values in the row immediately below the table tag

    Parameters
    ----------
    tag_row
        Row number for the tag designating the table to be extracted
    tag_col
        Column number for the tag designating the table to be extracted
    uc_sets
        Sets (regions and timeslices) for user constraints
    df
        Dataframe object containing all values for the worksheet being evaluated
    sheetname
        Name of the worksheet being evaluated
    filename
        Path to the excel file being evaluated.

    Returns
    -------
        Table object in the EmbeddedXlTable format.
    """
    # If the cell to the right is not empty then we read a scalar from it
    # Otherwise the row below is the header
    if df.shape[1] > tag_col + 1 and not cell_is_empty(df.iloc[tag_row, tag_col + 1]):
        table_range = str(
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
        table_range = str(
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
            table_df = df.iloc[header_row + 1 : end_row, start_col:end_col]
            # Make all columns names strings as some are integers e.g. years
            table_df.columns = [str(x) for x in df.iloc[header_row, start_col:end_col]]

    table_df = table_df.reset_index(drop=True)

    # Don't use applymap because it can convert ints to floats
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#gotchas-intna
    for i in range(table_df.shape[0]):
        for j in range(table_df.shape[1]):
            value = table_df.iat[i, j]
            if isinstance(value, float):
                table_df.iat[i, j] = utils.round_sig(value, 15)

    return datatypes.EmbeddedXlTable(
        filename=filename,
        sheetname=sheetname,
        range=table_range,
        tag=df.iloc[tag_row, tag_col],
        uc_sets=uc_sets,
        dataframe=table_df,
    )


def are_cells_all_empty(df, row: int, start_col: int, end_col: int) -> bool:
    """Check if all cells in a given row are empty by calling cell_is_empty() on them.

    Parameters
    ----------
    df
        Dataframe object containing all values for the worksheet being evaluated
    row
        Row of the dataframe to be evaluated.
    start_col
        Initial column of the dataframe to be evaluated.
    end_col
        Final column of the dataframe to be evaluated.

    Returns
    -------
    bool
        Boolean indicating if all the cells are empty.
    """
    for col in range(start_col, end_col):
        if not cell_is_empty(df.iloc[row, col]):
            return False
    return True


def cell_is_empty(value) -> bool:
    """Check if the given cell is empty.

    Parameters
    ----------
    value
        Cell value.

    Returns
    -------
    bool
        Boolean indicating if the cells are empty.
    """
    return (
        value is None
        or (isinstance(value, numpy.floating) and numpy.isnan(value))
        or (isinstance(value, str) and len(value.strip()) == 0)
    )


def _update_uc_sets(
    uc_sets: dict[str, dict[str, str]], new_element: str
) -> tuple[bool, dict[str, dict[str, str]]]:

    # Overview of the sets: https://times.readthedocs.io/en/latest/part-4/part-4.html#uc-sets-in-veda2
    # Categorise by type
    mapping = {
        "R_E": "region",
        "R_S": "region",
        "T_E": "period",
        "T_S": "period",
        "T_SUC": "period",
        "TS_E": "timeslice",
        "TS_S": "timeslice",
    }
    updated = False
    parts = [new_element.strip() for new_element in new_element.split(":")]
    # Check if the new element seems valid and if it is, update the uc_sets
    if len(parts) == 2 and parts[0] in mapping:
        uc_sets[mapping[parts[0]]] = {parts[0]: parts[1]}
        updated = True

    return updated, uc_sets
