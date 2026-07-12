# Only write debug table dumps in verbose mode

## What changed

`xl2times/main.py` wrote two debug files, `raw_tables.txt` (after reading the
input workbooks) and `merged_tables.txt` (as the final step of the transform
pipeline), on every run that had an `output_dir` — i.e. on all normal runs.
These files serialize every table as text and exist purely to support the
regression-debugging workflow described in the README, which runs the tool
with `-v` anyway.

This PR gates the dumps behind verbose mode:

- `read_xl` gains a `verbose: bool = False` parameter.
- `raw_tables.txt` is written only when `verbose` is set **or** when running
  with `--only_read` (`stop_after_read`), since producing `raw_tables.txt` is
  the entire purpose of that flag.
- The final `merged_tables.txt` dump runs only when `verbose` is set.
- Both `read_xl` call sites in `run()` pass `verbose=bool(args.verbose)`.

## User-visible change

Default (non-`-v`) runs no longer produce `raw_tables.txt` or
`merged_tables.txt` in the output directory. With `-v` (and, for
`raw_tables.txt`, with `--only_read`) behavior is exactly as before. A grep of
the repo (`tests/`, `utils/`, `docs/`) found no consumers that depend on these
files existing by default. All other outputs (CSVs, DD files, logging) are
unchanged.

## Why faster

The dumps build a full text serialization of every table. Per the performance
analysis (PERFORMANCE.md, section C), the final `merged_tables.txt` dump alone
takes ~4.7s on TIMES-GEO (~5s combined with `raw_tables.txt`) — pure overhead
on non-verbose runs, which now skip the work entirely. Performance will be
measured separately on an idle machine; this change was validated for
correctness only (the verification machine was shared, so runtimes there are
meaningless).

## Validation

All on the branch with the change (venv-local `xl2times`):

- `pytest tests/ -q` → 8 passed.
- Manual file-existence checks on DemoS_001:
  - `xl2times benchmarks/xlsx/DemoS_001 --output_dir /tmp/xl2t-c-default`
    → run succeeds; `raw_tables.txt` and `merged_tables.txt` NOT created.
  - `xl2times benchmarks/xlsx/DemoS_001 --output_dir /tmp/xl2t-c-verbose -v`
    → both files created and non-empty (raw_tables.txt 9.1K,
    merged_tables.txt 8.0K).
  - `xl2times benchmarks/xlsx/DemoS_001 --output_dir /tmp/xl2t-c-onlyread
    --only_read` → `raw_tables.txt` created (7.7K), preserving the
    `--only_read` contract.
- Demo benchmark suite (`utils/run_benchmarks.py`, 18 DemoS models,
  `--skip_csv --skip_regression`): Correct/Additional counts identical to
  main for every model (e.g. DemoS_001-all 118/3 … DemoS_012-all 7149/53,
  DemoS_special-t1 2108/42).
- TIMES-GEO (`--run TIMES-GEO --skip_csv --verbose`): printed exactly
  `96.1% (823412 correct, 14985 additional)` — identical to main.
- Full 24-model suite vs stored main results (`--skip_csv --skip_main`):
  zero changes in Correct and Additional for all models.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
