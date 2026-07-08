"""Tests for the vectorized `_generate_new_records` (TFM_UPD / TFM_MIG processing).

The `ref_*` functions below are verbatim copies (modulo renames) of the
pre-vectorization implementation from xl2times/transforms.py (as of commit
afb1d92), kept here as a reference implementation. Every test asserts that the
new vectorized `_generate_new_records` produces output identical to the
reference.

The two `test_fixture_equivalence_*` tests additionally validate equivalence
on the real TIMES-GEO data (~350k-row fi_t table). They require large pickle
fixtures in the repo root which are not committed; they are skipped when the
files are absent. Note that they are slow (minutes), as the reference
implementation is the very bottleneck being optimized.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm

from xl2times import utils
from xl2times.datatypes import Tag
from xl2times.transforms import _generate_new_records

utils.setup_logger(None)

# ---------------------------------------------------------------------------
# Reference implementation: verbatim copy of the previous (per-row) code.
# ---------------------------------------------------------------------------


def ref_query(
    table: DataFrame,
    process: str | list[str] | None,
    commodity: str | list[str] | None,
    attribute: str | None,
    region: str | list[str] | None,
    year: int | list | None,
    limtype: str | list[str] | None,
    val: int | float | None,
    module: str | list[str] | None,
) -> pd.Index:
    query_fields = {
        "process": process,
        "commodity": commodity,
        "attribute": attribute,
        "region": region,
        "year": year,
        "limtype": limtype,
        "value": val,
        "module_name": module,
    }

    def is_missing(field):
        return pd.isna(field) if not isinstance(field, list) else False

    qs = [
        f"{k} in {v if isinstance(v, list) else [v]}"
        for k, v in query_fields.items()
        if not is_missing(v)
    ]

    query_str = " and ".join(qs)
    row_idx = table.query(query_str).index
    return row_idx


def ref_eval_and_update(
    table: DataFrame, rows_to_update: pd.Index, new_value: str
) -> None:
    """Performs an inplace update of rows `rows_to_update` of `table` with
    `new_value`, which can be a update formula like `*2.3`.
    """
    if isinstance(new_value, str) and new_value[0] in {"*", "+", "-", "/"}:
        # Do not perform arithmetic operations on rows with i/e options
        if "year" in table.columns:
            rows_to_update = rows_to_update.intersection(
                table.index[table["year"] != 0]
            )
        old_values = table.loc[rows_to_update, "value"]
        updated = old_values.astype(float).map(lambda x: eval("x" + new_value))
        table.loc[rows_to_update, "value"] = updated
    else:
        table.loc[rows_to_update, "value"] = new_value


def ref_process_query(
    idx_and_row: tuple[Any, pd.Series], table: DataFrame, tag: Tag
) -> DataFrame | None:
    """Process a single TFM_MIG or TFM_UPD query."""
    # Check whether the tag is as expected. Raise an error if not
    if tag not in {Tag.tfm_mig, Tag.tfm_upd}:
        raise ValueError(f"Unexpected tag {tag.value} in _process_query.")

    _, row = idx_and_row
    if row["module_type"] == "trans":
        source_module = row["module_name"]
    else:
        source_module = row.get("sourcescen")
    rows_to_update = ref_query(
        table,
        row.get("process"),
        row.get("commodity"),
        row["attribute"],
        row.get("region"),
        row.get("year"),
        row.get("limtype"),
        row.get("val_cond"),
        source_module,
    )
    if rows_to_update.empty:
        logger.info(f"A {tag.value} row generated no records.")
        return None

    new_rows = table.loc[
        rows_to_update
    ].copy()  # Create a copy to avoid SettingWithCopyWarning

    if tag == Tag.tfm_mig:
        # Modify values in all '*2' columns
        for c, v in row.items():
            if str(c).endswith("2") and v is not None:
                new_rows.loc[:, str(c)[:-1]] = v

    # Evaluate 'value' column based on existing values
    ref_eval_and_update(
        new_rows,
        rows_to_update,
        str(row["value"]) if tag == Tag.tfm_mig else row["value"],
    )
    new_rows["source_filename"] = row["source_filename"]
    new_rows["module_name"] = row["module_name"]
    new_rows["module_type"] = row["module_type"]
    new_rows["submodule"] = row["submodule"]
    return new_rows


def ref_process_query_chunk(
    queries: DataFrame, table: DataFrame, tag: Tag
) -> list[DataFrame | None]:
    return [ref_process_query(q, table, tag) for q in queries.iterrows()]


def ref_generate_new_records(
    table: DataFrame, updates: DataFrame, tag: Tag, data_module: str
) -> list[DataFrame]:
    """Generate new records based on the given updates in TFM_UPD and TFM_MIG."""
    # Check whether the tag is as expected. Raise an error if not
    if tag not in {Tag.tfm_mig, Tag.tfm_upd}:
        raise ValueError(f"Unexpected tag {tag.value} in _generate_new_records.")

    results = []
    # Heuristic for deciding when to process in parallel
    if len(updates) > 100 and cpu_count() > 3:
        # Process queries in parallel using ProcessPoolExecutor
        n_workers = cpu_count() // 2

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            actual_n_workers = executor._max_workers  # pyright: ignore
            # Split queries into chunks based on worker count
            chunk_size = max(1, len(updates) // actual_n_workers)
            chunks = [
                updates.iloc[i : i + chunk_size]
                for i in range(0, len(updates), chunk_size)
            ]

            # Submit all tasks and tag each future with its chunk index
            future_info = {
                executor.submit(ref_process_query_chunk, chunk, table, tag): (
                    i,
                    len(chunk),
                )
                for i, chunk in enumerate(chunks)
            }
            results += [None] * len(future_info)
            with tqdm(
                total=len(updates),
                desc=f"Applying transformations concurrently from {tag.value} in {data_module}",
            ) as pbar:
                for f in as_completed(future_info):
                    idx, chunk_len = future_info[f]
                    results[idx] = f.result()
                    pbar.update(chunk_len)

            new_tables = [
                t for r in results if r is not None for t in r if t is not None
            ]
    else:
        # Process sequentially
        for q in tqdm(
            updates.iterrows(),
            total=len(updates),
            desc=f"Applying transformations from {tag.value} in {data_module}",
        ):
            results.append(ref_process_query(q, table, tag))
        new_tables = [t for t in results if t is not None]

    return new_tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent.parent


def concat_records(frames: list[DataFrame]) -> DataFrame:
    """Concatenate generated records exactly as the caller
    (apply_transform_tables) does.
    """
    assert isinstance(frames, list)
    if not frames:
        return DataFrame()
    return pd.concat(frames, ignore_index=True)


def assert_equivalent(
    table: DataFrame, updates: DataFrame, tag: Tag, data_module: str = "TEST"
) -> DataFrame:
    """Assert reference and new implementations produce identical output.

    Returns the (concatenated) output of the new implementation.
    """
    ref = ref_generate_new_records(
        table.copy(deep=True), updates.copy(deep=True), tag, data_module
    )
    new = _generate_new_records(
        table.copy(deep=True), updates.copy(deep=True), tag, data_module
    )
    ref_df = concat_records(ref)
    new_df = concat_records(new)
    pd.testing.assert_frame_equal(new_df, ref_df, check_exact=True)
    return new_df


def make_table() -> DataFrame:
    """A small fi_t-like table.

    All columns are object dtype, as in the real pipeline; `year` mixes floats
    and ints (incl. a 0 for an i/e option row) and `value` mixes int/float/str,
    as observed on TIMES-GEO data.
    """
    data = {
        "process": ["P1", "P2", "P1", "P3", None, "P1"],
        "commodity": ["C1", "C2", "C1", "C3", "C1", None],
        "attribute": ["ACT_BND", "ACT_BND", "COST", "ACT_BND", "COST", "ACT_BND"],
        "region": ["R1", "R2", "R1", "R1", "R2", "R1"],
        "year": [2020.0, 2030.0, 2020, 0, 2030.0, 2020.0],
        "limtype": ["UP", "LO", None, "UP", None, "UP"],
        "value": [1.0, 2, 5.5, 4.0, "3", 6],
        "module_name": ["M1", "M1", "M2", "M1", "M2", "M1"],
        "source_filename": [f"f{i}" for i in range(6)],
        "module_type": ["base"] * 6,
        "submodule": ["s0"] * 6,
        "timeslice": ["ANNUAL"] * 6,
    }
    return DataFrame({k: pd.Series(v, dtype=object) for k, v in data.items()})


def make_updates(rows: list[dict]) -> DataFrame:
    """Build an updates table (object dtype) with sensible defaults."""
    defaults = {
        "process": None,
        "commodity": None,
        "attribute": None,
        "region": None,
        "year": None,
        "limtype": None,
        "val_cond": None,
        "sourcescen": None,
        "year2": None,
        "value": None,
        "source_filename": "uf",
        "module_type": "scen",
        "module_name": "UM",
        "submodule": "us",
    }
    recs = [{**defaults, **r} for r in rows]
    return DataFrame(recs, dtype=object)


# ---------------------------------------------------------------------------
# Synthetic tests (always run)
# ---------------------------------------------------------------------------


class TestGenerateNewRecords:
    def test_unexpected_tag_raises(self):
        table = make_table()
        updates = make_updates([{"attribute": "ACT_BND", "value": 1}])
        with pytest.raises(ValueError, match="Unexpected tag"):
            _generate_new_records(table, updates, Tag.fi_t, "TEST")

    def test_no_match_row_logs_and_returns_empty(self):
        table = make_table()
        updates = make_updates(
            [
                {"attribute": "NOPE", "value": 1},
                {"attribute": "ALSO_NOPE", "value": 2},
            ]
        )
        messages: list[str] = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
        try:
            result = _generate_new_records(table, updates, Tag.tfm_upd, "TEST")
        finally:
            logger.remove(handler_id)
        assert result == []
        assert (
            sum(
                f"A {Tag.tfm_upd.value} row generated no records." in m
                for m in messages
            )
            == 2
        )
        # And the reference agrees there is nothing to generate
        assert ref_generate_new_records(table, updates, Tag.tfm_upd, "TEST") == []

    def test_mixed_match_and_no_match_logs_once_per_row(self):
        table = make_table()
        updates = make_updates(
            [
                {"attribute": "ACT_BND", "value": 1},
                {"attribute": "NOPE", "value": 2},
            ]
        )
        messages: list[str] = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
        try:
            result = _generate_new_records(table, updates, Tag.tfm_upd, "TEST")
        finally:
            logger.remove(handler_id)
        assert (
            sum(
                f"A {Tag.tfm_upd.value} row generated no records." in m
                for m in messages
            )
            == 1
        )
        assert len(concat_records(result)) == 4
        assert_equivalent(table, updates, Tag.tfm_upd)

    def test_literal_value_upd(self):
        table = make_table()
        updates = make_updates([{"attribute": "COST", "value": 7}])
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        # UPD assigns non-formula values literally, keeping their type
        assert out["value"].tolist() == [7, 7]
        assert all(v == 7 and not isinstance(v, str) for v in out["value"])

    def test_arithmetic_value_and_year_zero_exclusion(self):
        table = make_table()
        updates = make_updates([{"attribute": "ACT_BND", "value": "*2"}])
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        # Matches rows 0, 1, 3, 5. Row 3 has year == 0 (i/e option) and must
        # keep its old value; the others are doubled.
        assert out["value"].tolist() == [2.0, 4.0, 4.0, 12.0]
        # Metadata is set on all rows, including the excluded one
        assert out["source_filename"].tolist() == ["uf"] * 4

    def test_arithmetic_division_and_addition(self):
        table = make_table()
        updates = make_updates(
            [
                {"attribute": "COST", "value": "/2"},
                {"attribute": "COST", "value": "+1.5"},
            ]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        # COST matches rows 2 and 4 (value 5.5 and "3" -> cast to float)
        assert out["value"].tolist() == [2.75, 1.5, 7.0, 4.5]

    def test_mig_value_str_cast(self):
        table = make_table()
        updates = make_updates([{"attribute": "COST", "value": 7}])
        out = assert_equivalent(table, updates, Tag.tfm_mig)
        # TFM_MIG casts the new value to str before assignment
        assert out["value"].tolist() == ["7", "7"]

    def test_mig_star2_column_override(self):
        table = make_table()
        updates = make_updates(
            [{"attribute": "ACT_BND", "region": "R1", "year2": 2050, "value": 9}]
        )
        out = assert_equivalent(table, updates, Tag.tfm_mig)
        assert out["year"].tolist() == [2050, 2050, 2050]
        assert out["value"].tolist() == ["9", "9", "9"]

    def test_mig_star2_none_is_not_applied(self):
        table = make_table()
        updates = make_updates(
            [{"attribute": "ACT_BND", "region": "R1", "year2": None, "value": 9}]
        )
        out = assert_equivalent(table, updates, Tag.tfm_mig)
        # year2 is None -> no override; original years preserved
        assert out["year"].tolist() == [2020.0, 0, 2020.0]

    def test_mig_star2_pdna_is_applied(self):
        # Quirk: the check is `v is not None`, so pd.NA *is* applied,
        # and arithmetic still happens on such rows (NA != 0 evaluates True).
        table = make_table()
        updates = make_updates(
            [{"attribute": "ACT_BND", "region": "R1", "year2": pd.NA, "value": "*2"}]
        )
        out = assert_equivalent(table, updates, Tag.tfm_mig)
        assert all(v is pd.NA for v in out["year"])
        # All matched rows updated (incl. the year==0 row, since its year
        # was overridden to NA before the exclusion check)
        assert out["value"].tolist() == [2.0, 8.0, 12.0]

    def test_mig_year2_zero_disables_arithmetic(self):
        # Overriding year to 0 makes every row an "i/e option" row, so the
        # arithmetic update is skipped and old values are kept.
        table = make_table()
        updates = make_updates(
            [{"attribute": "ACT_BND", "region": "R1", "year2": 0, "value": "*2"}]
        )
        out = assert_equivalent(table, updates, Tag.tfm_mig)
        assert out["year"].tolist() == [0, 0, 0]
        assert out["value"].tolist() == [1.0, 4.0, 6]

    def test_upd_ignores_star2_columns(self):
        table = make_table()
        updates = make_updates(
            [{"attribute": "ACT_BND", "region": "R1", "year2": 2050, "value": 9}]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        assert out["year"].tolist() == [2020.0, 0, 2020.0]
        assert out["value"].tolist() == [9, 9, 9]

    def test_list_filter_fields(self):
        table = make_table()
        updates = make_updates(
            [
                {"attribute": "ACT_BND", "process": ["P1", "P3"], "value": 1},
                {"attribute": "ACT_BND", "limtype": ["UP", "FX"], "value": 2},
            ]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        assert len(out) == 6

    def test_empty_list_matches_nothing(self):
        table = make_table()
        updates = make_updates([{"attribute": "ACT_BND", "process": [], "value": 1}])
        messages: list[str] = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
        try:
            result = _generate_new_records(table, updates, Tag.tfm_upd, "TEST")
        finally:
            logger.remove(handler_id)
        assert result == []
        assert any(
            f"A {Tag.tfm_upd.value} row generated no records." in m for m in messages
        )
        assert ref_generate_new_records(table, updates, Tag.tfm_upd, "TEST") == []

    def test_list_with_duplicates_matches_once(self):
        table = make_table()
        updates = make_updates(
            [{"attribute": "ACT_BND", "process": ["P1", "P1"], "value": 1}]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        assert len(out) == 2

    def test_val_cond_filters_on_value_column(self):
        table = make_table()
        updates = make_updates(
            [
                # int 1 must match the float 1.0 in the table
                {"attribute": "ACT_BND", "val_cond": 1, "value": "*3"},
                {"attribute": "ACT_BND", "val_cond": 2, "value": "*3"},
            ]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        assert out["value"].tolist() == [3.0, 6.0]

    def test_year_int_filter_matches_float_years(self):
        table = make_table()
        updates = make_updates([{"attribute": "ACT_BND", "year": 2020, "value": 8}])
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        # 2020 (int) must match both 2020.0 (float, rows 0 and 5)
        assert len(out) == 2

    def test_year_list_filter(self):
        table = make_table()
        updates = make_updates(
            [{"attribute": "ACT_BND", "year": [2020, 2030], "value": 8}]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        assert len(out) == 3

    def test_source_module_trans_uses_module_name(self):
        table = make_table()
        updates = make_updates(
            [
                {
                    "attribute": "COST",
                    "value": 1,
                    "module_type": "trans",
                    "module_name": "M2",
                }
            ]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        # module_name filter M2 -> rows 2 and 4; new module_name is the
        # update row's own
        assert len(out) == 2
        assert out["module_name"].tolist() == ["M2", "M2"]

    def test_source_module_non_trans_uses_sourcescen(self):
        table = make_table()
        updates = make_updates(
            [
                {
                    "attribute": "ACT_BND",
                    "value": 1,
                    "module_type": "scen",
                    "sourcescen": "M1",
                },
                # NaN sourcescen -> module unconstrained
                {
                    "attribute": "ACT_BND",
                    "value": 1,
                    "module_type": "scen",
                    "sourcescen": float("nan"),
                },
            ]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        assert len(out) == 8

    def test_multi_row_output_order(self):
        # Output must be ordered by update row first (iterrows order), then
        # by table row order within each update row.
        table = make_table()
        updates = make_updates(
            [
                {"attribute": "ACT_BND", "value": 10},
                {"attribute": "COST", "value": 20},
                {"attribute": "ACT_BND", "region": "R1", "value": 30},
            ]
        )
        out = assert_equivalent(table, updates, Tag.tfm_upd)
        assert out["value"].tolist() == [10, 10, 10, 10, 20, 20, 30, 30, 30]
        # Non-updated columns are copied through from the matched table rows
        assert out["timeslice"].tolist() == ["ANNUAL"] * 9

    def test_empty_updates(self):
        table = make_table()
        updates = make_updates([]).reindex(columns=make_updates([{}]).columns)
        result = _generate_new_records(table, updates, Tag.tfm_upd, "TEST")
        assert result == []


# ---------------------------------------------------------------------------
# Equivalence tests on real TIMES-GEO data (skipped if fixtures are absent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name,tag",
    [
        ("tfm_mig_queries-12276.pkl", Tag.tfm_mig),
        ("tfm_mig_queries-217.pkl", Tag.tfm_upd),
        ("tfm_mig_queries-217.pkl", Tag.tfm_mig),
    ],
)
def test_fixture_equivalence(fixture_name, tag):
    """New implementation must produce byte-identical output to the reference
    on the real TIMES-GEO fi_t table and TFM_MIG/TFM_UPD updates.
    """
    import pickle

    fixture = FIXTURE_DIR / fixture_name
    if not fixture.exists():
        pytest.skip(f"fixture {fixture_name} not available")
    with fixture.open("rb") as f:
        table, updates = pickle.load(f)
    assert_equivalent(table, updates, tag, data_module="GEO")
