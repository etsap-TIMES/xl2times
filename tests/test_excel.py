import pandas as pd

from xl2times import excel


def test_extract_table_handles_empty_terminal_tag():
    df = pd.DataFrame([[None, None, "~TFM_DINS-TS"]], dtype=object)

    table = excel.extract_table(
        tag_row=0,
        tag_col=2,
        uc_sets={},
        df=df,
        sheetname="SolWin AF",
        filename="BY_Trans.xlsx",
    )

    assert table.tag == "~TFM_DINS-TS"
    assert table.dataframe.empty
    assert list(table.dataframe.columns) == ["VALUE"]


def test_extract_table_handles_tag_with_blank_header_row():
    df = pd.DataFrame([["~TFM_DINS-TS"], [None]], dtype=object)

    table = excel.extract_table(
        tag_row=0,
        tag_col=0,
        uc_sets={},
        df=df,
        sheetname="SolWin AF",
        filename="BY_Trans.xlsx",
    )

    assert table.tag == "~TFM_DINS-TS"
    assert table.dataframe.empty
