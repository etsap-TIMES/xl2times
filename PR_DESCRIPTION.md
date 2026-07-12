# Vectorize default-PCG selection and comm-group naming

## What changed

`xl2times/transforms.py`, `include_cgs_in_topology` calls two helpers that
previously drove their work through pandas' slow per-group/per-row Python
callback machinery:

1. **`_process_comm_groups_vectorised`** (selection of the default primary
   commodity group per region/process) used
   `comm_groups.groupby(["region", "process"], sort=False, as_index=False).apply(_set_default_veda_pcg)`,
   invoking a Python callback once per (region, process) group — tens of
   thousands of groups on real models — each of which ran a nested
   `for io in ["OUT", "IN"]: for cset in csets_ordered_for_pcg]` scan and
   returned a DataFrame slice that pandas then had to reconcatenate. This is
   also the source of a pandas `FutureWarning` about `DataFrameGroupBy.apply`
   operating on grouping columns.
2. **`name_comm_group`** was a row-wise `comm_groups.apply(name_comm_group, axis=1)`
   over the same frame.

Both are replaced with vectorized, C-level pandas operations:

- `_process_comm_groups_vectorised` now computes an `eligible` mask via
  `csets.isin(csets_ordered_for_pcg).groupby([region, process]).transform("all")`,
  a numeric `rank = io_rank * num_csets + cset_rank` (`OUT` → 0, `IN` → 1,
  everything else → NaN; `cset_rank` = position in `csets_ordered_for_pcg`),
  and `min_rank = rank.groupby([region, process]).transform("min")`. The
  winning rows are `eligible & (rank == min_rank)`, plus one more explicit
  term for a preserved quirk (see below). No new commodity-group naming
  helper class was introduced — the caller and the returned frame's
  shape/dtype/index/row order are unchanged.
- The row-wise `name_comm_group` apply is replaced by a new
  `_name_comm_groups_vectorised`, using boolean masks
  (`commoditygroup > 1` / `== 1`) and vectorized string concatenation
  (`process + "_" + csets + io.str[:1]`) instead of a per-row Python call.

## Why the new implementation is semantically identical (including a quirk)

The old `_set_default_veda_pcg(group)` callback, per (region, process) group:

```python
if not group["csets"].isin(csets_ordered_for_pcg).all():
    return group  # guard: any out-of-list cset -> group left untouched

for io in ["OUT", "IN"]:
    for cset in csets_ordered_for_pcg:
        group.loc[(group["io"] == io) & (group["csets"] == cset), "DefaultVedaPCG"] = True
        if group["DefaultVedaPCG"].any():
            break  # <-- only breaks the inner `cset` loop!
```

**The `break` only exits the inner `cset` loop, not the outer `io` loop.**
So whenever the group has *any* `OUT` row (which, given the guard passed,
means its cset is in `csets_ordered_for_pcg`), the first `OUT` cset in
priority order gets flagged, the inner loop breaks — but the outer loop then
proceeds to `io = "IN"`, whose first iteration (`cset = "DEM"`, the first
entry in `csets_ordered_for_pcg`) unconditionally flags any
`(io="IN", csets="DEM")` rows in the group **before** the `.any()` check on
that iteration fires and breaks the (now pointless) outer loop.

Net effect: whenever a group has any `OUT` row, its `(IN, DEM)` rows (if
any) are *also* flagged `True` — in addition to the "real" winning
`(OUT, first-present-cset)` rows. This looks like a latent bug in the
original code (the intent was clearly "OUT wins outright, only fall back to
IN if there's no OUT"), but it is observable behavior in the current output,
so it must be preserved bit-for-bit rather than "fixed" as a drive-by change
in a pure-performance PR.

The vectorized version reproduces this exactly with an explicit extra
disjunct:

```python
default_pcg = eligible & (
    (rank == min_rank) | ((min_rank < num_csets) & (rank == num_csets))
)
```

- `rank == min_rank` is the normal winner-takes-all case (equivalent to "the
  row(s) with the lowest `io_rank*num_csets + cset_rank` in the group").
- `(min_rank < num_csets) & (rank == num_csets)` is the quirk: `min_rank <
  num_csets` means the winning row is an `OUT` row (rank 0..num_csets-1);
  `rank == num_csets` is exactly `io_rank=1 (IN), cset_rank=0 (DEM)` —
  i.e. `(IN, DEM)` rows. This is flagged in a code comment at the
  implementation site as a possible latent bug worth separate review — it is
  intentionally **not** fixed here.

This equivalence (guard, winner selection, and the `(IN, DEM)` quirk) is
checked directly: `tests/test_transforms.py` keeps a verbatim copy of the
old `_set_default_veda_pcg`-based implementation as
`_process_comm_groups_reference`, and asserts
`pd.testing.assert_frame_equal` between the reference and vectorized outputs
on real austimes data (`tests/data/austimes_pcg_test_data.parquet`) plus a
set of synthetic edge cases that specifically exercise: the quirk triggering
(OUT + IN/DEM present), IN-only groups, the ineligibility guard (a cset
outside the ordered list), io values outside IN/OUT, an OUT winner that
isn't DEM (so no quirk row), and single-row groups. `name_comm_group` gets
the same treatment via `_name_comm_groups_reference`.

## Why it's faster

`groupby.apply` with a Python callback pays, per group: Python function-call
overhead, a `.loc` boolean-mask assignment inside a nested Python loop, an
`.any()` reduction, and — critically — pandas has to *reconcatenate* every
group's returned DataFrame slice back into one frame afterward (hence the
caller's `reset_index(level=0, drop=True).sort_index()` to undo the
grouping-induced reindex). On TIMES-GEO this callback runs across tens of
thousands of (region, process) groups over an ~86k-row frame. The row-wise
`name_comm_group` apply pays similar per-row Python-call overhead across the
same frame.

The vectorized replacement does the equivalent work with two
`groupby(...).transform(...)` calls (`"all"` and `"min"`), which run in
compiled/vectorized pandas code across the whole column at once — no
per-group Python callback, no per-group DataFrame slicing, no
reconcatenation step. `_name_comm_groups_vectorised` similarly replaces a
per-row Python call with column-level boolean masking and string
concatenation. This is expected to remove essentially all of
`include_cgs_in_topology`'s cost, which PERFORMANCE.md section B measured as
62.6s of TIMES-GEO's ~298s baseline runtime (see PERFORMANCE.md section B for
the full rationale). Actual wall-clock improvement will be measured
separately on an idle (non-shared) machine — the runs performed during this
work were on a machine shared with concurrent sibling agents, so no
self-measured timing numbers are reported here.

## Validation performed

- `pytest tests/ -q` → **12 passed** (9 in `test_transforms.py`, including 4
  new tests: `test_default_pcg_vectorised_matches_reference`,
  `test_default_pcg_vectorised_edge_cases`,
  `test_name_comm_groups_vectorised_matches_reference`,
  `test_name_comm_groups_vectorised_edge_cases`; 3 in `test_utils.py`).
- Demo benchmark suite (18 `DemoS_*` models, `--skip_csv --skip_regression`):
  **exact match** to the expected accuracy/correct/additional table for
  every model (e.g. `DemoS_001-all` 100.0%/118/3 ... `DemoS_special-t1`
  92.2%/2108/42). Zero deviations.
- TIMES-GEO (`--run TIMES-GEO --skip_csv --verbose`): **exact match**,
  `96.1% (823412 correct, 14985 additional)`. Confirmed the
  `DataFrameGroupBy.apply` `FutureWarning` about grouping columns no longer
  appears in the log.
- Full 24-model suite vs `benchmarks/out-main/` (`--skip_csv --skip_main`):
  **zero regressions.** Every one of the 24 benchmarks (18 `DemoS_*`,
  `TIMES-IE-all`/`TIMES-IE-NoM`/`TIMES-IE-MCB`, `TIMES-NZ-KEA`/`TIMES-NZ-TUI`,
  `TIMES-GEO`) shows identical Correct and Additional counts to the
  pre-change baseline. Summary line from the run: `Change in correct rows
  (higher == better): +0 (+0.0%)`, `Change in additional rows: +0 (+0.0%)`,
  `SUCCESS: No regressions. You're awesome!`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
