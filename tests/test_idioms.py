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
from xl2times.datatypes import Config, EmbeddedXlTable, Tag, TimesModel

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


class TestExpandRows:
    """D4: transforms.expand_rows."""

    def test_basic_comma_split_and_explode(self):
        df = pd.DataFrame(
            {
                "region": ["R1,R2", "R3"],
                "other": ["x", "y"],
            }
        )
        out = transforms.expand_rows(set(), {"region"}, _table(df)).dataframe
        assert out["region"].tolist() == ["R1", "R2", "R3"]
        assert out["other"].tolist() == ["x", "x", "y"]

    def test_query_columns_kept_as_list_not_exploded(self):
        df = pd.DataFrame({"region": ["R1,R2", "R3"], "other": ["x", "y"]})
        out = transforms.expand_rows({"region"}, {"region"}, _table(df)).dataframe
        assert len(out) == 2
        # Only the comma-containing entry is turned into a list; the
        # non-comma entry is left as a bare string (matches _split_by_commas).
        assert out["region"].tolist() == [["R1", "R2"], "R3"]

    def test_non_candidate_column_with_commas_untouched(self):
        # "other" has commas but isn't in lists_columns -> left alone, not
        # exploded, no crash.
        df = pd.DataFrame({"region": ["R1", "R2"], "other": ["a,b", "c,d"]})
        out = transforms.expand_rows(set(), {"region"}, _table(df)).dataframe
        assert out["other"].tolist() == ["a,b", "c,d"]
        assert out["region"].tolist() == ["R1", "R2"]

    def test_non_string_cells_mixed_with_comma_strings(self):
        df = pd.DataFrame({"region": ["R1,R2", 5, None, float("nan")]})
        out = transforms.expand_rows(set(), {"region"}, _table(df)).dataframe
        # Row 0 explodes into 2 rows (R1, R2); the other rows are untouched
        # (no comma to split) and keep their original position/value.
        assert out["region"].tolist()[:2] == ["R1", "R2"]
        assert out["region"].tolist()[2] == 5
        assert out["region"].tolist()[3] is None
        assert pd.isna(out["region"].tolist()[4])

    def test_empty_lists_columns_is_noop(self):
        df = pd.DataFrame({"region": ["R1,R2", "R3"]})
        out = transforms.expand_rows(set(), set(), _table(df)).dataframe
        assert out["region"].tolist() == ["R1,R2", "R3"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"region": pd.Series([], dtype=object)})
        out = transforms.expand_rows(set(), {"region"}, _table(df)).dataframe
        assert len(out) == 0

    def test_multi_column_explode(self):
        df = pd.DataFrame({"region": ["R1,R2"], "year": ["2020,2021"]})
        out = transforms.expand_rows(set(), {"region", "year"}, _table(df)).dataframe
        # Both columns exploded independently -> cross-product-like blow-up
        # via sequential .explode calls (region first, then year).
        assert len(out) == 4
        assert set(out["region"]) == {"R1", "R2"}
        assert set(out["year"]) == {"2020", "2021"}

    def test_all_numeric_object_column_in_lists_columns_no_crash(self):
        df = pd.DataFrame({"region": pd.Series([1, 2, 3], dtype=object)})
        out = transforms.expand_rows(set(), {"region"}, _table(df)).dataframe
        assert out["region"].tolist() == [1, 2, 3]

    def test_no_commas_anywhere(self):
        df = pd.DataFrame({"region": ["R1", "R2", "R3"]})
        out = transforms.expand_rows(set(), {"region"}, _table(df)).dataframe
        assert out["region"].tolist() == ["R1", "R2", "R3"]

    def test_lists_columns_entry_not_a_real_column_is_skipped(self):
        df = pd.DataFrame({"region": ["R1", "R2"]})
        out = transforms.expand_rows(
            set(), {"region", "nonexistent_col"}, _table(df)
        ).dataframe
        assert out["region"].tolist() == ["R1", "R2"]
        assert "nonexistent_col" not in out.columns


def _basic_model() -> TimesModel:
    model = TimesModel()
    model.processes = pd.DataFrame(
        {"process": ["P1", "P2"], "description": ["d1", "d2"], "sets": ["ELE", "DEM"]}
    )
    model.commodities = pd.DataFrame(
        {"commodity": ["C1"], "description": ["c1"], "csets": ["NRG"]}
    )
    model.topology = pd.DataFrame(
        {"process": ["P1"], "commodity": ["C1"], "io": ["IN"]}
    )
    return model


class TestProcessUserDefinedSets:
    """D5: transforms.process_user_defined_sets / generate_topology_dictionary.

    "ELE" is a real entry in times-sets.json's PRC_GRP set (confirmed via
    the config fixture's times_sets_file), so a pset_set value of "ELE"
    resolves against config.times_sets["PRC_GRP"] on the first fixpoint
    iteration.
    """

    def test_psets_fixpoint_chunking_dependent_set(self, config: Config):
        # SETB's pset_set references SETA, which is only defined once SETA
        # itself is resolved in an earlier fixpoint iteration -> exercises
        # the chunked/fixpoint resolution order this fix must not disturb.
        model = _basic_model()
        tables = {
            Tag.tfm_psets: pd.DataFrame(
                {"set_name": ["SETA", "SETB"], "pset_set": ["ELE", "SETA"]}
            ),
        }
        transforms.process_user_defined_sets(config, tables, model)

        assert model.user_psets["sets"].tolist() == ["SETA", "SETB"]
        assert model.user_psets["process"].tolist() == ["P1", "P1"]
        assert model.user_csets.empty

    def test_csets_basic_resolution(self, config: Config):
        model = _basic_model()
        tables = {
            Tag.tfm_csets: pd.DataFrame({"set_name": ["CSETA"], "cset_set": ["NRG"]}),
        }
        transforms.process_user_defined_sets(config, tables, model)

        assert model.user_csets["csets"].tolist() == ["CSETA"]
        assert model.user_csets["commodity"].tolist() == ["C1"]
        assert model.user_psets.empty

    def test_both_tags_processed_independently(self, config: Config):
        # csets is processed before psets (to_process order); confirms both
        # branches run without interfering with each other's results.
        model = _basic_model()
        tables = {
            Tag.tfm_csets: pd.DataFrame({"set_name": ["CSETA"], "cset_set": ["NRG"]}),
            Tag.tfm_psets: pd.DataFrame({"set_name": ["SETA"], "pset_set": ["ELE"]}),
        }
        transforms.process_user_defined_sets(config, tables, model)

        assert model.user_csets["csets"].tolist() == ["CSETA"]
        assert model.user_psets["sets"].tolist() == ["SETA"]
        assert model.user_psets["process"].tolist() == ["P1"]

    def test_no_set_type_column_all_rows_independent(self, config: Config):
        # No "pset_set" column -> df_rows is just [df] (no fixpoint
        # chunking) -> exercises the "else" branch feeding into the
        # generate_topology_dictionary/per-df_row loop. Uses "pset_pn"
        # (a different process_map wildcard column) so _match_wildcards
        # still has a non-empty wild_cols to merge on.
        model = _basic_model()
        tables = {
            Tag.tfm_psets: pd.DataFrame({"set_name": ["SETA"], "pset_pn": ["P1"]}),
        }
        transforms.process_user_defined_sets(config, tables, model)
        assert model.user_psets["sets"].tolist() == ["SETA"]
        assert model.user_psets["process"].tolist() == ["P1"]


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
