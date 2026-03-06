# Copilot Instructions for xl2times

## Project Summary

**xl2times** is a Python CLI tool that converts TIMES energy system models specified in Excel spreadsheets (Veda-TIMES format) into a format ready for processing by GAMS. It is used by researchers and energy modellers working with the [ETSAP TIMES model generator](https://github.com/etsap-TIMES/TIMES_model).

- **Language:** Python 3.11+
- **Type:** CLI application / data-processing library
- **Key dependencies:** `pandas >= 2.1`, `openpyxl >= 3.1.3`, `loguru`, `tqdm`, `more-itertools`, `pyarrow`, `GitPython`
- **Build system:** `setuptools`; task automation via `poethepoet` (`poe`)
- **Linter/formatter:** `ruff` (v0.3.2) + `pyright` (v1.1.304)

## Bootstrap / Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[dev]              # installs runtime + dev extras
pre-commit install                 # install git hooks (ruff, pyright, etc.)
```

## Running Tests

Unit tests live in `tests/` and use **pytest**.

```bash
pytest                             # quick run
poe test                           # run with coverage report (html + term)
```

Test files: `tests/test_transforms.py`, `tests/test_utils.py`, `tests/test_query.py`.

Pytest options are configured in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Linting and Type Checking

Always lint before committing. The pre-commit hooks run **ruff** (lint + format) and **pyright**.

```bash
poe lint                           # run pre-commit on staged files
poe lint-all                       # run pre-commit on ALL files
```

Or run tools directly:

```bash
ruff check --fix xl2times tests    # lint and auto-fix
ruff format xl2times tests         # format code
pyright                            # type-check (config: pyrightconfig.json)
```

Ruff config: `pyproject.toml` `[tool.ruff]` – target Python 3.11, line-length 88.  
Selected rule groups: `E, W, F, UP, N, I, TID, NPY, PL, D`.  
Ignored: `PLR`, `E501`, `D100-D105`, `D205`, `D401`.  
Docstring convention: NumPy style.

## Project Layout

```
xl2times/
├── xl2times/               # Main package
│   ├── __main__.py         # CLI entry point → main.parse_args() / main.run()
│   ├── main.py             # Core orchestration (reads xlsx, runs transforms, writes output)
│   ├── transforms.py       # All data transformations (~3750 lines) – most logic lives here
│   ├── datatypes.py        # Dataclasses / enums: Tag, EmbeddedXlTable, TimesModel, Config
│   ├── excel.py            # Excel parsing – extracts EmbeddedXlTable objects from workbooks
│   ├── utils.py            # Shared helpers (tag handling, composite tags, formatting)
│   ├── dd_to_csv.py        # Utility: converts GAMS DD files to CSV
│   └── config/             # JSON/txt config: veda-tags.json, times-info.json, times-sets.json,
│                           #   times_mapping.txt, veda-attr-defaults.json
├── tests/                  # Unit tests (pytest)
├── utils/                  # Dev utilities: run_benchmarks.py, compare_results.py, etc.
├── docs/                   # Sphinx documentation source
├── pyproject.toml          # Project metadata, ruff config, pytest config, poe tasks
├── pyrightconfig.json      # Pyright type-checker config
├── .pre-commit-config.yaml # Pre-commit hooks (ruff, pyright, yaml check, etc.)
├── benchmarks.yml          # Benchmark model definitions
└── setup-benchmarks.sh     # Script to clone benchmark model repos
```

## CLI Usage

```bash
xl2times <input_dir_or_files> [options]
# Options: --output_dir, --regions, --dd, --ground_truth_dir, --verbose (-v/-vv), --no_cache
```

## CI Checks (`.github/workflows/ci.yml`)

The CI pipeline (Ubuntu, Python 3.11) runs on every push/PR to `main`:

1. `pip install -e .[dev]`
2. `ruff check` and `ruff format --check` (code style)
3. `pyright` (type checking)
4. `pre-commit run --all-files`
5. `pytest` (unit tests)
6. Benchmark regression tests (requires benchmark repos cloned by `setup-benchmarks.sh`)

**All of the above must pass before a PR can be merged.** Always run `poe lint-all` and `pytest` locally before pushing.

## Coding Conventions

- **Style:** PEP 8, enforced by ruff. 88-char line length.
- **Type hints:** Required on all functions (pyright strict mode).
- **Docstrings:** NumPy style; currently not enforced for missing docstrings (D100–D105 ignored).
- **Naming:** `PascalCase` for classes, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- **Imports:** Organised by isort rules (via ruff `I`); stdlib → third-party → local.
- **No `# noqa` or `type: ignore` unless strictly necessary** – prefer fixing the underlying issue.
- **Caching:** Excel parse results are cached to `~/.cache/xl2times/` (auto-invalidates after 365 days). Use `--no_cache` to bypass.
