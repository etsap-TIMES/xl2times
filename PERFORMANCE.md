# Performance notes and optimization plan

This document records a performance analysis of `xl2times` on large models
(primarily TIMES-GEO), the root causes found, and a plan for fixing them.
It also collects project facts useful for anyone working on performance.

Analysis date: 2026-07-07, on commit `5537def` (main).

## Where the time goes (TIMES-GEO)

Total runtime ~298s (all other benchmark models finish in under a minute).
Per-transform timings from a verbose run (`benchmarks/out/TIMES-GEO/stdout`):

| Transform | Time (s) | Share |
|---|---|---|
| `apply_transform_tables` | 186.4 | 63% |
| `include_cgs_in_topology` | 62.6 | 21% |
| final `<lambda>` (merged_tables.txt dump) | 4.7 | |
| `process_flexible_import_tables` | 4.1 | |
| `prepare_for_querying` | 3.5 | |
| `apply_final_fixup` | 3.4 | |
| `expand_rows_parallel` | 3.0 | |
| `explode_process_commodity_cols` | 2.7 | |
| `process_user_defined_sets` | 2.6 | |
| `capitalise_table_values` | 1.7 | |
| `process_wildcards` | 1.5 | |
| `convert_aliases` | 1.4 | |

Prior art: PR #338 parallelized TFM_MIG row processing and added `--profile`
(TIMES-GEO 838s → 229s); PR #343 improved UPD handling. The remaining 186s is
what parallelism alone couldn't remove.

## Root causes

### 1. `apply_transform_tables`: per-row full-table queries (186s)

The dominant cost is one ~TFM_MIG table with 12,276 rows
(`Scen_01_DemandProjections_SSP2`). Path:
`_generate_new_records` → `_process_query` → `query` (`xl2times/transforms.py`).

Each update row calls `table.query(query_str)` — a full scan of the merged
`fi_t` table (~350k rows × 20 cols at that point), with predicates like
`process in [<wildcard-matched list>]`. That's 12,276 sequential full-table
scans per worker chunk. The wildcard-match caches (`_match_wildcards` dedup,
`lru_cache`d regexes in `utils.py`) only help `process_wildcards`, not this path.

The `ProcessPoolExecutor` parallelism splits rows into `cpu_count()//2` chunks
and pickles the full `fi_t` to every worker. The tqdm progress bar updates only
when a whole ~1534-row chunk completes, so the log's "slow first 1500 rows,
then fast" pattern is an artifact — wall time is the slowest chunk.

Key insight: within a data module, all rows query the same immutable `fi_t`
snapshot and results are concatenated afterwards — so the work can be done
with a handful of vectorized hash joins instead of 12k scans.

Note: `tests/test_query.py` documents an earlier attempt using boolean-mask
cross-products (N×Q), which was abandoned as too slow/large. A grouped
merge/join does not have that blowup: cost is O(N + exploded_Q + matches).

### 2. `include_cgs_in_topology`: groupby.apply over ~40k groups (62.6s)

`_process_comm_groups_vectorised` does
`groupby(["region", "process"]).apply(_set_default_veda_pcg)` over an ~86k-row
frame — a Python callback returning a DataFrame per group, for tens of
thousands of groups, plus pandas' slice-and-reconcat machinery. (This function
was itself a big improvement over an older per-row loop — see the docstring of
`test_default_pcg_vectorised` — but `groupby.apply` is still the bottleneck.)

Behavioral quirk to preserve: in `_set_default_veda_pcg`, the `break` only
exits the inner cset loop. So when an OUT cset wins, the loop still runs
io="IN", flags any (IN, DEM) rows True, and only then stops. Possibly a latent
bug, but it affects output; a faithful rewrite must reproduce it (decision:
preserve bit-for-bit, flag for separate review).

Also row-wise: `comm_groups.apply(name_comm_group, axis=1)`.

### 3. Debug table dumps on the critical path (~5s)

`dump_tables(...merged_tables.txt)` (and `raw_tables.txt`) run whenever
`output_dir` is set — i.e. on every normal run — purely for debugging.

### 4. Miscellaneous pandas anti-patterns (~10-15s combined)

Row-wise `.apply(axis=1)`, per-cell `.map`/`isinstance` checks, elementwise
`DataFrame.replace`, and a repeatedly rebuilt lookup dictionary. Details in
plan D below.

## Optimization plan

### A. Vectorize TFM_UPD/TFM_MIG record generation (branch `claude-perf-vectorize-tfm-updates`)

Rewrite `_generate_new_records` to replace per-row scans with merges:

1. Reset the `fi_t` index to get an explicit row-id column.
2. Compute a `source_module` column on updates (`module_name` if
   `module_type == "trans"` else `sourcescen`).
3. Group update rows by "filter signature": the set of non-null query fields
   among {process, commodity, attribute, region, year, limtype, val_cond,
   module}.
4. Per signature: explode list-valued filter columns, then inner-merge with
   `fi_t` on the signature columns (`val_cond`→`value`, module→`module_name`)
   to get (update_id, fi_t_row_id) match pairs in one hash join.
   Normalize join-key dtypes first (`.query("x in [v]")` compares across
   numeric dtypes; merge keys must match).
5. Log updates with zero matches (anti-join) to keep the
   "A tfm_mig row generated no records" messages.
6. Build the new rows with `table.take(...)`, apply TFM_MIG `*2` column
   overrides vectorized from aligned update values, set the 4 metadata columns.
7. Vectorize `eval_and_update`: group matched pairs by distinct formula
   string; `eval("x" + formula)` where `x` is a whole Series is vectorized;
   keep the year != 0 exclusion for arithmetic formulas; assign literals
   directly otherwise.
8. Stable-sort pairs by (update row position, fi_t row id) so the concat
   order matches the current per-row output exactly.
9. Drop the `ProcessPoolExecutor` path (also removes the repeated pickling of
   `fi_t` to workers).

Validation fixture: `tfm_mig_queries-12276.pkl` (repo root, untracked) holds
the exact `(fi_t, updates)` pair from TIMES-GEO. A unit test can assert the
new implementation produces identical output to the old one (skip if the file
is absent; don't commit it). The same treatment can optionally be applied to
the TFM_INS-TXT loop.

### B. Vectorize default-PCG selection (branch `claude-perf-vectorize-default-pcg`)

Replace the `groupby.apply` in `_process_comm_groups_vectorised` with
transform-based ranking:

- `valid` = `csets.isin(csets_ordered_for_pcg)` → `groupby(...).transform("all")`
- `rank = io_rank * 5 + cset_rank` (OUT=0, IN=1; cset rank = index in
  `csets_ordered_for_pcg`; NaN for other io values)
- `min_rank = rank.where(valid_group).groupby([...]).transform("min")`
- flag = `valid_group & (rank == min_rank)`, **plus the quirk term**
  `| (valid_group & (min_rank <= 4) & (rank == 5))` to reproduce the (IN, DEM)
  side-effect described above
- Set `DefaultVedaPCG` True on flagged rows, None elsewhere (keep dtype).

Also vectorize `name_comm_group` with `np.select`. Extend
`test_default_pcg_vectorised` (which uses
`tests/data/austimes_pcg_test_data.parquet`) to assert the new version equals
the current implementation's output. Removes the pandas FutureWarning at the
`groupby.apply` call site.

### C. Gate debug table dumps behind verbose mode (branch `claude-perf-lazy-debug-dumps`)

Only call `dump_tables` (`raw_tables.txt` in `read_xl`, `merged_tables.txt` in
the final lambda) when running at verbose/debug log level. User-visible
change: these files disappear from default runs (the regression-debugging
workflow in the README uses `-v` anyway, and benchmarks run without `-v`).

### D. Pandas idiom fixes (branch `claude-perf-pandas-idioms`)

Five independent small fixes:

1. **`prepare_for_querying` year-range expansion**: replace the two
   `df[i].apply(..., axis=1)` calls with vectorized string ops on
   `df.loc[i, "year"].astype(str)`: a `.str.contains("-", regex=False)` mask,
   `.str.split("-", n=1).str[0/1]` for the parts, combined with `where`/mask
   assignment. Preserve the exact semantics: `year2` gets the second part as
   int only for ranges; `year` gets the first part as int for ranges, NA for
   empty string, unchanged otherwise.
2. **`capitalise_table_values`**: drop the per-cell
   `apply(lambda x: isinstance(x, str))`. `Series.str.upper().str.strip()`
   yields NaN for non-string entries, so
   `upper.where(upper.notna(), original)` (or masking on `.str` result
   notna) gives the same result in one vectorized pass per column.
3. **`convert_aliases`**: replace
   `df.replace({"attribute": replacement_dict})` (elementwise over the whole
   frame) with
   `df["attribute"].map(replacement_dict).fillna(df["attribute"])`.
4. **`expand_rows`**: `df.map(_has_comma)` scans every cell of every table.
   Restrict to candidate columns
   (`df.columns ∩ lists_columns`) and use vectorized
   `.str.contains(",", regex=False).fillna(False)` per column instead of a
   Python call per cell.
5. **`process_user_defined_sets`**: `generate_topology_dictionary` (which
   concats and re-indexes all process/commodity lookup tables) is rebuilt for
   every resolved chunk inside the `for df_row in df_rows` loop. Only the
   `processes_by_sets` / `commodities_by_sets` entries depend on
   `model.user_psets` / `model.user_csets` (mutated in the loop) — build the
   invariant entries once and refresh only the dependent entry per iteration.

## Verification procedure

For any performance change:

1. Unit tests: `pytest tests/ -q`.
2. Quick regression on small models: filter `benchmarks.yml` to the DemoS
   entries and run
   `python utils/run_benchmarks.py benchmarks-demos.yml --skip_csv --skip_regression`;
   compare the accuracy table before/after your change.
3. TIMES-GEO specifically:
   `python utils/run_benchmarks.py benchmarks.yml --run TIMES-GEO --skip_csv --verbose`
   must report exactly **96.1% (823412 correct, 14985 additional)**.
4. Full suite vs main:
   `python utils/run_benchmarks.py benchmarks.yml --skip_csv` (checks out main
   in-repo to compare; in a git worktree use `--skip_main` with a pre-generated
   `benchmarks/out-main/`). Correct/additional row deltas must be zero;
   runtime comparisons are noisy on a busy machine.
5. Profile: `python utils/run_benchmarks.py benchmarks.yml --run TIMES-GEO --profile`
   → `xl2times.prof`, view with snakeviz. Caveat: cProfile doesn't see inside
   `ProcessPoolExecutor` child processes; use py-spy `--subprocesses` or force
   the sequential path for those sections.

## Project facts (for future performance work)

- Pipeline: `read_xl` in `xl2times/main.py` runs a list of ~44 transforms,
  each `(config, tables, model) -> tables`; per-transform wall time is logged
  at INFO. Tables start as `list[EmbeddedXlTable]` and become
  `dict[str, DataFrame]` after `merge_tables`; results accumulate in the
  `TimesModel` dataclass (`datatypes.py`).
- `transforms.py` (~3.8k lines) contains essentially all compute.
- Parallelism (`ProcessPoolExecutor`): Excel reading (`main.py`),
  `expand_rows_parallel`, and `_generate_new_records` (TFM_UPD/MIG). Worker
  count: `utils.max_workers`. Remember arguments are pickled per task —
  passing large DataFrames to workers is expensive.
- Excel extraction results are cached as pickles in `~/.cache/xl2times/`
  keyed by file content hash (`--no_cache` to bypass).
- Benchmark harness: `utils/run_benchmarks.py` + `benchmarks.yml` (24 models).
  Ground truth CSVs in `benchmarks/csv/<name>` are generated from DD files by
  `xl2times/dd_to_csv.py` (skip regeneration with `--skip_csv`).
- The regression comparison literally checks out `main` in the working repo
  and re-runs everything, unless `--skip_main` (reads stored results from
  `benchmarks/out-main/<name>/stdout`) or `--skip_regression` is given.
- Deps: pandas >=2.1,<3.0, Python >=3.11. CI checks pyright (1.1.304) and
  black formatting via pre-commit.
