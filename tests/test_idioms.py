"""Behavior-pinning tests for the pandas-idiom micro-optimizations in
transforms.py (PERFORMANCE.md section D).

Each test class pins the exact observable behavior (values AND dtypes) of the
function it targets, verified against the original row/cell-wise
implementation before the vectorized rewrite.
"""

from typing import Any, cast

import pandas as pd
import pytest

from xl2times import transforms, utils
from xl2times.datatypes import Config, EmbeddedXlTable

utils.setup_logger(None)

# capitalise_table_values takes config/model params it doesn't use; pass a
# typed None so pyright doesn't flag the test call sites.
_NONE = cast(Any, None)


def _table(df: pd.DataFrame) -> EmbeddedXlTable:
    return EmbeddedXlTable(
        tag="~TEST",
        uc_sets={},
        sheetname="sheet",
        range="",
        filename="",
        dataframe=df,
    )


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(
        mapping_file="times_mapping.txt",
        times_info_file="times-info.json",
        times_sets_file="times-sets.json",
        veda_tags_file="veda-tags.json",
        veda_attr_defaults_file="veda-attr-defaults.json",
        regions="",
        include_dummy_imports=False,
        case=None,
    )


class TestExpandYearRanges:
    """D1: transforms._expand_year_ranges (used by prepare_for_querying)."""

    def test_mixed_range_and_non_range_overwrites_existing_year2(self):
        df = pd.DataFrame(
            {
                "year": ["2010-2030", "2015", "", pd.NA],
                "year2": [999, 888, 777, 1999],
            }
        )
        transforms._expand_year_ranges(df)

        assert df["year"].tolist() == [2010, "2015", pd.NA, pd.NA]
        assert isinstance(df["year"].iloc[0], int)
        # year2 overwritten to NA for non-range rows even though it had a
        # pre-existing value; the NA-year row (index 3) is left untouched.
        assert df["year2"].iloc[0] == 2030
        assert isinstance(df["year2"].iloc[0], int)
        assert pd.isna(df["year2"].iloc[1])
        assert pd.isna(df["year2"].iloc[2])
        assert df["year2"].iloc[3] == 1999
        assert df["year"].dtype == object
        assert df["year2"].dtype == object

    def test_all_na_year_creates_float64_year2(self):
        df = pd.DataFrame({"year": [pd.NA, pd.NA]})
        transforms._expand_year_ranges(df)
        assert "year2" in df.columns
        assert df["year2"].dtype == "float64"
        assert df["year2"].isna().all()

    def test_all_na_year_leaves_existing_year2_untouched(self):
        df = pd.DataFrame({"year": [pd.NA, pd.NA], "year2": [1, 2]})
        transforms._expand_year_ranges(df)
        assert df["year2"].tolist() == [1, 2]
        assert df["year2"].dtype == "int64"

    def test_all_ranges_no_na_elsewhere(self):
        df = pd.DataFrame({"year": ["2010-2030", "2020-2040"]})
        transforms._expand_year_ranges(df)
        assert df["year"].tolist() == [2010, 2020]
        assert df["year2"].tolist() == [2030, 2040]

    def test_all_ranges_with_na_row_elsewhere_year2_becomes_float(self):
        # Partial-index assignment into a brand-new "year2" column upcasts
        # int64 to float64 (matching the original .apply-based assignment).
        df = pd.DataFrame({"year": ["2010-2030", "2020-2040", pd.NA]})
        transforms._expand_year_ranges(df)
        assert df["year2"].dtype == "float64"
        assert df["year2"].tolist()[:2] == [2030.0, 2040.0]
        assert pd.isna(df["year2"].iloc[2])

    def test_float_year_column_with_nan_unchanged_dtype(self):
        df = pd.DataFrame({"year": [2010.0, 2020.0, float("nan")]})
        transforms._expand_year_ranges(df)
        # No ranges/empty-strings among non-NA rows -> year column dtype and
        # values are left completely untouched (stays float64).
        assert df["year"].dtype == "float64"
        assert df["year"].tolist()[:2] == [2010.0, 2020.0]
        assert pd.isna(df["year"].iloc[2])
        # year2 is still (re)computed as an object column of NA.
        assert df["year2"].dtype == object
        assert pd.isna(df["year2"].iloc[0])
        assert pd.isna(df["year2"].iloc[1])
        assert pd.isna(df["year2"].iloc[2])

    def test_multi_hyphen_range_uses_first_two_parts(self):
        # No maxsplit: "2010-2030-2040" -> year=2010, year2=2030 (3rd part
        # dropped), matching the original `.split("-")` (no `n=1`).
        df = pd.DataFrame({"year": ["2010-2030-2040"]})
        transforms._expand_year_ranges(df)
        assert df["year"].iloc[0] == 2010
        assert df["year2"].iloc[0] == 2030

    def test_empty_string_year_becomes_na(self):
        df = pd.DataFrame({"year": [""]})
        transforms._expand_year_ranges(df)
        assert pd.isna(df["year"].iloc[0])
        assert pd.isna(df["year2"].iloc[0])

    def test_no_year_column_is_noop(self):
        df = pd.DataFrame({"other": [1, 2]})
        transforms._expand_year_ranges(df)
        assert list(df.columns) == ["other"]

    def test_empty_dataframe_with_year_column(self):
        df = pd.DataFrame({"year": pd.Series([], dtype=object)})
        transforms._expand_year_ranges(df)
        assert len(df) == 0

    def test_object_dtype_int_years_non_range(self):
        df = pd.DataFrame({"year": [2015, 2020, pd.NA]})
        transforms._expand_year_ranges(df)
        assert df["year"].tolist() == [2015, 2020, pd.NA]
        assert pd.isna(df["year2"].iloc[0])
        assert pd.isna(df["year2"].iloc[1])


class TestCapitaliseTableValues:
    """D2: transforms.capitalise_table_values."""

    def test_mixed_str_int_none_nan_object_column(self):
        df = pd.DataFrame({"c": ["abc ", 5, None, float("nan"), " def"]})
        out = transforms.capitalise_table_values(_NONE, [_table(df)], _NONE)[
            0
        ].dataframe
        assert out["c"].tolist()[0] == "ABC"
        assert out["c"].tolist()[1] == 5
        assert out["c"].tolist()[2] is None
        assert pd.isna(out["c"].tolist()[3])
        assert out["c"].tolist()[4] == "DEF"

    def test_all_numeric_object_column_is_noop(self):
        # .str accessor raises AttributeError on an object column with no
        # actual string entries at all; must be handled as a no-op, not a
        # crash.
        df = pd.DataFrame({"c": pd.Series([1, 2, 3], dtype=object)})
        out = transforms.capitalise_table_values(_NONE, [_table(df)], _NONE)[
            0
        ].dataframe
        assert out["c"].tolist() == [1, 2, 3]

    def test_non_object_column_untouched(self):
        df = pd.DataFrame({"c": [1, 2, 3]})
        out = transforms.capitalise_table_values(_NONE, [_table(df)], _NONE)[
            0
        ].dataframe
        assert out["c"].dtype == "int64"
        assert out["c"].tolist() == [1, 2, 3]

    def test_whitespace_stripped_and_upper(self):
        df = pd.DataFrame({"c": ["  mixed Case  "]})
        out = transforms.capitalise_table_values(_NONE, [_table(df)], _NONE)[
            0
        ].dataframe
        assert out["c"].iloc[0] == "MIXED CASE"

    def test_empty_table_returned_as_is(self):
        df = pd.DataFrame({"c": pd.Series([], dtype=object)})
        out = transforms.capitalise_table_values(_NONE, [_table(df)], _NONE)[
            0
        ].dataframe
        assert len(out) == 0


class TestConvertAliases:
    """D3: transforms.convert_aliases."""

    def test_exact_case_sensitive_full_value_match(self, config: Config):
        # VAROM -> ACT_COST and AF -> NCAP_AF are real entries in
        # veda-attr-defaults.json's "aliases" table.
        df = pd.DataFrame({"attribute": ["VAROM", "AF", "af", "UNKNOWN_ATTR"]})
        tables = {"t": df}
        out = transforms.convert_aliases(config, tables, _NONE)["t"]

        assert out["original_attr"].tolist() == ["VAROM", "AF", "af", "UNKNOWN_ATTR"]
        assert out["attribute"].tolist() == [
            "ACT_COST",
            "NCAP_AF",
            "af",
            "UNKNOWN_ATTR",
        ]

    def test_none_and_nan_attribute_values_untouched(self, config: Config):
        df = pd.DataFrame({"attribute": ["VAROM", None, float("nan")]})
        tables = {"t": df}
        out = transforms.convert_aliases(config, tables, _NONE)["t"]

        assert out["attribute"].iloc[0] == "ACT_COST"
        assert out["attribute"].iloc[1] is None
        assert pd.isna(out["attribute"].iloc[2])

    def test_table_without_attribute_column_untouched(self, config: Config):
        df = pd.DataFrame({"other": [1, 2]})
        tables = {"t": df}
        out = transforms.convert_aliases(config, tables, _NONE)["t"]
        assert list(out.columns) == ["other"]

    def test_empty_attribute_column(self, config: Config):
        df = pd.DataFrame({"attribute": pd.Series([], dtype=object)})
        tables = {"t": df}
        out = transforms.convert_aliases(config, tables, _NONE)["t"]
        assert len(out) == 0
        assert "original_attr" in out.columns
