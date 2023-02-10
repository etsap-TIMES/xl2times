from openpyxl import load_workbook
from openpyxl.worksheet.cell_range import CellRange
from typing import Dict, List
import time
from pandas.core.frame import DataFrame
import pandas as pd
import numpy
import re
from . import datatypes
from . import utils


def extract_tables(filename: str) -> List[datatypes.EmbeddedXlTable]:
    """
    This function calls the extract_table function on each individual table in each worksheet of the
    given excel file.

    :param filename:      Path to the excel file we will extract tables from.
    :return:              List of table objects in EmbeddedXlTable format.
    """
    start_time = time.time()

    workbook = load_workbook(filename=filename, data_only=True)

    tables = []
    for sheet in workbook.worksheets:
        # Creating dataframe with dtype=object solves problems with ints being cast to floats
        # https://stackoverflow.com/questions/40251948/stop-pandas-from-converting-int-to-float-due-to-an-insertion-in-another-column
        df = pd.DataFrame(sheet.values, dtype=object)
        uc_sets = {}

        for row_index, row in df.iterrows():
            for colname in df.columns:
                value = str(row[colname])
                if value.startswith("~"):
                    match = re.match(
                        f"~{datatypes.Tag.uc_sets}:(.*)", value, re.IGNORECASE
                    )
                    if match:
                        parts = match.group(1).split(":")
                        if len(parts) == 2:
                            uc_sets[parts[0].strip()] = parts[1].strip()
                        else:
                            print(
                                f"WARNING: Malformed UC_SET in {sheet.title}, {filename}"
                            )
                    else:
                        col_index = df.columns.get_loc(colname)
                        tables.append(
                            extract_table(
                                row_index, col_index, uc_sets, df, sheet.title, filename
                            )
                        )

    end_time = time.time()
    if end_time - start_time > 2:
        print(f"Loaded {filename} in {end_time-start_time:.2f} seconds")

    return tables


def extract_table(
    tag_row: int,
    tag_col: int,
    uc_sets: Dict[str, str],
    df: DataFrame,
    sheetname: str,
    filename: str,
) -> datatypes.EmbeddedXlTable:
    """
    For each individual table tag found in a worksheet, this function aims to extract
    the associated table. We recognise several types of tables:
    - Single cell tables:   Tables with only one value, either below or to the right of
                            the table tag. We interpret these as a single data item with
                            a column name VALUE.
    - Multiple cell tables: Tables with multiple values, possibly extending accross
                            several rows and columns. We delimitate them using empty
                            spaces around them and the column names are determined by the
                            values in the row immediately below the table tag

    :param tag_row:         Row number for the tag designating the table to be extracted
    :param tag_col:         Column number for the tag designating the table to be extracted
    :param df:              Dataframe object containing all values for the worksheet being evaluated
    :param sheetname:       Name of the worksheet being evaluated
    :param filename:        Path to the excel file being evaluated.
    :return:                Table object in the EmbeddedXlTable format.
    """
    # If the cell to the right is not empty then we read a scalar from it
    # Otherwise the row below is the header
    if not cell_is_empty(df.iloc[tag_row, tag_col + 1]):
        range = str(
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
        range = str(
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
            table_df = DataFrame(df.iloc[header_row + 1 : end_row, start_col:end_col])
            # Make all columns names strings as some are integers e.g. years
            table_df.columns = [str(x) for x in df.iloc[header_row, start_col:end_col]]

    table_df.reset_index(drop=True, inplace=True)
    table_df = table_df.applymap(
        lambda cell: cell if not isinstance(cell, float) else utils.round_sig(cell, 15)
    )

    return datatypes.EmbeddedXlTable(
        filename=filename,
        sheetname=sheetname,
        range=range,
        tag=df.iloc[tag_row, tag_col],
        uc_sets=uc_sets,
        dataframe=table_df,
    )


def are_cells_all_empty(df, row: int, start_col: int, end_col: int) -> bool:
    """
    Check if all cells in a given row are empty by calling cell_is_empty() on them.

    :param df:              Dataframe object containing all values for the worksheet being evaluated
    :param row:             Row of the dataframe to be evaluated.
    :param start_col:       Initial column of the dataframe to be evaluated.
    :param end_col:         Final column of the dataframe to be evaluated.
    :return:                Boolean indicating if all the cells are empty.
    """
    for col in range(start_col, end_col):
        if not cell_is_empty(df.iloc[row, col]):
            return False
    return True


def cell_is_empty(value) -> bool:
    """
    Check if the given cell is empty.

    :param value:           Cell value.
    :return:                Boolean indicating if the cells are empty.
    """
    return (
        value is None
        or (isinstance(value, numpy.float64) and numpy.isnan(value))
        or (isinstance(value, str) and len(value.strip()) == 0)
    )
