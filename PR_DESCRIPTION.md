# Vectorize TFM_UPD/TFM_MIG record generation with grouped hash joins

## What changed

`_generate_new_records` in `xl2times/transforms.py` (the function that expands
`~TFM_UPD` / `~TFM_MIG` update tables against the merged `fi_t` table) is
rewritten to replace per-row `DataFrame.query` scans with a small number of
vectorized hash joins:

- Update rows are grouped by "filter signature" — the set of non-null filter
  fields among `process`, `commodity`, `attribute`, `region`, `year`,
  `limtype`, `val_cond` (→ `fi_t["value"]`), and `source_module` (→
  `fi_t["module_name"]`, computed as `row["module_name"]` if
  `row["module_type"] == "trans"` else `row.get("sourcescen")`).
- Within each signature group, list-valued filter columns are exploded, and
  the group is inner-merged against `fi_t` (reset-index first to carry an
  explicit row id) on the shared signature columns — one hash join per
  distinct signature instead of one full-table scan per update row.
- Value updates (`eval_and_update`) are vectorized: matched pairs are grouped
  by distinct formula string and `eval("x" + formula)` is applied to whole
  Series at once (already vectorized when `x` is a Series), excluding rows
  where `year == 0` from arithmetic (they keep the old value); literal values
  are assigned directly.
- The `ProcessPoolExecutor` chunking path (which pickled the entire `fi_t`
  table to every worker per chunk) is removed entirely — the vectorized path
  needs no parallelism.
- `eval_and_update`, `_process_query`, `_process_query_chunk`, and the
  now-unused `as_completed`/`cpu_count` imports are deleted as dead code.
  `query()` itself is kept (still used by `_remove_invalid_rows` and the
  TFM_INS-TXT loop, which this PR does not touch).

## Why the new implementation is equivalent to the old one

The core semantic argument: `DataFrame.query("col in [v]")` for a scalar
filter field is elementwise equality, and pandas merge on the same key
column(s) is also elementwise equality — so grouping rows by identical filter
*signature*, exploding list-valued filters to one row per candidate value,
and inner-joining on those columns produces exactly the same (update row,
fi_t row) match set as running the equivalent query string per row, just
computed as one hash join instead of N linear scans.

Quirks from the original code that are deliberately preserved (all covered by
tests in `tests/test_generate_records.py`, which also keeps a verbatim copy
of the old `_generate_new_records`/`_process_query`/`query`/`eval_and_update`
as a reference implementation):

- **Numeric cross-dtype matching**: `query("x in [v]")` treats `2020` (int)
  and `2020.0` (float) as equal. Verified empirically that pandas merge on
  object-dtype key columns has the same behavior, so no dtype normalization
  changes match sets. All `fi_t`/update columns here are object dtype.
- **NaN never appears among merge keys**: unconstrained (all-null) fields are
  excluded from the merge entirely, and exploded empty-list cells are dropped
  via `dropna` before the join, so merge's `NaN == NaN` semantics can't
  introduce spurious matches that the old query-string approach wouldn't
  have had.
- **MIG value `str()` cast**: for TFM_MIG, the new value is cast to `str(...)`
  before assignment, so a numeric MIG value ends up stored as a *string* —
  preserved exactly.
- **`*2`-column overrides for TFM_MIG**: for each update column whose name
  ends in `2` with value `v`, matched rows get column `name[:-1]` set to `v`.
  The override check is `v is not None` (not "truthy" / not "notna"), so a
  `pd.NA` value in a `*2` column IS applied as an override — verified this is
  exactly what the old code does, and it matters: the 12,276-row fixture has
  `year2 = pd.NA` on every row, so matched rows get `year = <NA>`, and since
  `<NA> != 0` evaluates `True`, arithmetic on `value` still applies afterward.
  `year2 = 0` (a real 0), by contrast, disables arithmetic for the row.
- **`year == 0` exclusion from arithmetic**: rows where the (possibly
  `*2`-overridden) `year` is `0` keep their old `value` even when the update's
  value is an arithmetic formula.
- **Metadata assignment order**: `source_filename`, `module_name`,
  `module_type`, `submodule` are set from the update row *after* the value
  evaluation step, matching the old ordering.
- **Output order**: update rows are processed in `iterrows()` order, and
  within one update row, matched `fi_t` rows appear in original index order —
  reproduced with a stable sort on (update row position, `fi_t` row id)
  before concatenation.
- **No-match logging**: an update row with zero matches still logs
  `"A {tag.value} row generated no records."` exactly once (via anti-join
  against the merge).
- **Unexpected tag**: still raises `ValueError`, as before.
- **Empty filter signature**: reproduces the old `table.query("")` behavior
  (raises), rather than being special-cased away.

### Divergences knowingly accepted (disclosed, not present in real fixtures)

Three edge cases where the new implementation's behavior differs from the
old one in principle, none of which are exercised by the TIMES-GEO fixtures
or the 24-model benchmark suite (all confirmed byte-identical against the old
implementation on both fixtures and zero-regression across all models):

1. If a `*2` override column's target name (`name[:-1]`) is *not* an existing
   `fi_t` column, the old code's `pd.concat` could in some edge cases produce
   a `float64` column where the new code produces `object` dtype. All real
   `*2` columns actually used (`year2`, etc.) already exist in `fi_t`, so this
   never triggers in practice.
2. A `NaN` occurring *inside* a filter list (as opposed to the whole field
   being null) would match nothing under the new merge-based approach; the
   old code's behavior here was already effectively broken (a `nan` embedded
   in a generated query string is not valid query syntax), so this is not a
   regression in any meaningful sense. Fixtures confirm filter lists never
   contain NA.
3. A list-valued `*2` override value would be assigned as a literal cell
   value (a Python list) under the new code, versus the old code's row-wise
   assignment semantics, which were arguably already broken for list values.
   Not present in any fixture or benchmark model.

## Why it's faster

The old path ran one `DataFrame.query()` — a full linear scan of the merged
`fi_t` table (~350k rows × 20 columns for TIMES-GEO) — per update row. The
12,276-row TFM_MIG table in TIMES-GEO alone required 12,276 such full-table
scans (plus the `ProcessPoolExecutor` chunking pickled the entire `fi_t`
table to every worker on top of that). The new path groups update rows by
filter signature and performs one hash join per group: cost is
O(fi_t rows + exploded update rows + matches) rather than
O(update rows × fi_t rows). This removes both the linear-scan-per-row cost
and the repeated worker-process pickling of the full table.

Per `PERFORMANCE.md` section A, `apply_transform_tables` (dominated by this
code path) took 186.4s of TIMES-GEO's ~298s total runtime on `main`
(commit `5537def`). Actual before/after timing on this change has **not**
been measured in this session (the machine may be running concurrent work,
so self-measured wall-clock numbers here would not be meaningful) — a
dedicated timing run on an idle machine is deferred to a follow-up step.
Only the expected/previously-measured baseline above is cited.

## Validation performed

- `pytest tests/ -q`: **33 passed** (394.10s), including:
  - 22 synthetic unit tests in `tests/test_generate_records.py` covering:
    no-match row + its log message, `"*2"` arithmetic, literal values,
    string-cast MIG numeric values, `*2`-column overrides (including the
    `pd.NA` quirk), `year == 0` exclusion, `year2 == 0` disabling arithmetic,
    list vs. scalar filter fields (including empty list and duplicates),
    `val_cond` filtering, int/float year cross-matching, `module_type ==
    "trans"` vs. `sourcescen`, output ordering, and the unexpected-tag
    `ValueError`.
  - Fixture equivalence tests (`pd.testing.assert_frame_equal`,
    `check_exact=True`, on concatenated output) against a verbatim copy of
    the old implementation:
    - `tfm_mig_queries-217.pkl` with `Tag.tfm_upd`: **PASS**.
    - `tfm_mig_queries-217.pkl` with `Tag.tfm_mig`: **PASS**.
    - `tfm_mig_queries-12276.pkl` (the exact 350,043×20 `fi_t` / 12,276-row
      TFM_MIG update table from TIMES-GEO) with `Tag.tfm_mig`: **PASS**
      (`1 passed in 237.47s`).
- Demo benchmarks (correctness only): `python utils/run_benchmarks.py
  benchmarks-demos.yml --skip_csv --skip_regression` — all 18 `Demo*` models'
  Correct/Additional counts matched the expected table exactly (e.g.
  `DemoS_001-all` 100.0% 118/3 ... `DemoS_special-t1` 92.2% 2108/42), no
  deviations.
- TIMES-GEO: `python utils/run_benchmarks.py benchmarks.yml --run TIMES-GEO
  --skip_csv --verbose` printed exactly **`96.1% (823412 correct, 14985
  additional)`**, matching the required value bit-for-bit.
- Full 24-model suite vs. pre-seeded `main` results: `python
  utils/run_benchmarks.py benchmarks.yml --skip_csv --skip_main` reported
  **zero change** in Correct rows (+0, +0.0%) and Additional rows (+0,
  +0.0%) across all models, concluding "No regressions. You're awesome!"
  Runtime figures from this run are not reported/relied upon (shared/busy
  machine); actual performance measurement is deferred to a dedicated run on
  an idle machine, where the expected win is on `apply_transform_tables`
  (186.4s of TIMES-GEO's total on `main`, per `PERFORMANCE.md` section A).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
