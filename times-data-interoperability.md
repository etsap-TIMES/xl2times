# `xl2times` ↔ `times-data` Interoperability

## Overview

[`xl2times`](https://github.com/etsap-TIMES/xl2times) converts Veda-TIMES
Excel workbooks into TIMES model data. Its public API (`main.read_xl`) returns
a **`TimesModel` Python object** that holds all processed model data — regions,
processes, commodities, topology, attributes, time slices, time periods, trade
links, user constraints, and units — as pandas DataFrames and plain Python
sets. A separate step (`main.to_tables` / `main.write_dd_files`) can serialise
that object into DD files, but DD output is **optional**; the in-memory
`TimesModel` is already a complete, structured representation of the model.

[`times-data`](https://github.com/MMobir/times-data) provides a typed,
validated, scriptable data layer for TIMES models. Its `Model` object stores
commodities, processes, and parameters in a queryable graph. It can import DD
files, build models from scratch in Python, validate against a full TIMES
schema (288 parameters, 178 sets, 37 indexes), compile back to DD, or export
to YAML.

Both projects target the same problem domain — representing and manipulating
TIMES models in Python — but from different entry points. This document
analyses how they can help each other and which TIMES model-generator data they
both duplicate.

---

## 1 How `times-data` can be useful for `xl2times`

### 1.1 Output validation without GAMS

`xl2times` currently validates its output by comparing DD/CSV files against
ground-truth tables produced by Veda/GAMS. This requires the ground truth to
exist in the first place.

`times-data` contains an independent validation engine that checks parameter
values against the full TIMES specification — correct index signatures, valid
set memberships, and structural integrity (no orphan commodities, no
disconnected processes). If `xl2times` were to export its `TimesModel` into a
`times-data` `Model` (or pass through DD files), `times-data` could flag
errors that the current ground-truth comparison cannot catch — for example a
parameter that has a valid value but is applied to the wrong index combination.

### 1.2 Authoritative TIMES schema as a shared dependency

`xl2times` encodes TIMES model-generator knowledge in five JSON/text
configuration files:

| File | Content |
|---|---|
| `config/times-info.json` | 283 TIMES parameters with GAMS indexes and column mappings |
| `config/times-sets.json` | 11 set enumerations (`COM_TYPE`, `PRC_GRP`, `TSLVL`, `LIM`, …) |
| `config/times_mapping.txt` | 41 DD output table definitions (table → GAMS indexes) |
| `config/veda-attr-defaults.json` | 136 Veda attribute aliases with default bound types, timeslice levels, and commodity-group rules |
| `config/veda-tags.json` | ~40 Veda input tag definitions with column specs and parsing rules |

`times-data` maintains a parallel but richer typed schema:

| Module | Content |
|---|---|
| `schema/parameters.py` (`PARAMETER_REGISTRY`) | 288 parameters — name, index tuple, description, category, units/range info, default interpolation, related sets, affected equations |
| `schema/sets.py` (`SET_REGISTRY`) | 178 sets — base name, indexes, aliases, description, category (User Input / Internal) |
| `schema/indexes.py` (`INDEX_REGISTRY`) | 37 indexes — name, aliases, related indexes, description |

The overlap is significant. Both need to know, for instance, that `ACT_BND`
takes indexes `(r, datayear, p, s, bd)` and that `COM_TYPE` contains
`{DEM, NRG, MAT, ENV, FIN}`. Today each project maintains its own copy; a
shared upstream source would prevent them from drifting apart when the TIMES
model generator is updated.

### 1.3 Round-trip and cross-project testing

Both projects benchmark against the official IEA-ETSAP DemoS_001–DemoS_012
models. A cross-project integration test could work as follows:

1. `xl2times` reads the Demo Excel workbooks and returns a `TimesModel`.
2. The `TimesModel` is converted to DD files (already implemented).
3. `times-data` imports those DD files (`import-dd`) and validates the
   resulting `Model`.
4. `times-data` compiles the `Model` back to DD and solves with GAMS.
5. The objective values are compared against the known reference values.

This would give end-to-end coverage — from spreadsheet through two independent
code paths to solver — that neither project achieves alone.

### 1.4 Programmatic model querying after conversion

Once a `TimesModel` exists in memory, a user may want to query it — "which
processes produce ELC?", "what is the NCAP_COST of WIND_ON in 2030?" —
without going through DD serialisation. `times-data`'s `Model` offers exactly
that kind of API (`model.producers_of("ELC")`,
`model.process_cost("WIND_ON", 2030)`). A thin adapter that populates a
`times-data` `Model` from the `xl2times` `TimesModel` would unlock these
queries immediately after conversion.

---

## 2 How `xl2times` can be useful for `times-data`

### 2.1 Excel-to-programmatic bridge

`times-data` explicitly does **not** read Veda-TIMES spreadsheets (see its
"What this is not" section). Many TIMES models are specified and maintained in
Excel. `xl2times` is the open-source path from those spreadsheets into the
programmatic world:

```
Excel workbooks  ──▶  xl2times (read_xl)  ──▶  TimesModel (Python object)
                                                     │
                                   ┌─────────────────┼─────────────────┐
                                   ▼                 ▼                 ▼
                              DD files          CSV tables       (future) times-data Model
                                   │
                                   ▼
                         times-data import-dd  ──▶  times-data Model
```

The critical insight is that the DD-file step is **not required** for
interoperability. `xl2times`'s `TimesModel` already holds all the structural
data — regions, commodities (with type, timeslice level, unit), processes
(with primary commodity group, sets, vintage, activity/capacity units),
topology (process × commodity × IO direction), trade links, time periods, time
slices, and all parameter values — in pandas DataFrames. A direct
`TimesModel → times-data Model` adapter could skip DD serialisation and
deserialisation entirely, preserving richer type information (e.g., commodity
types as enums rather than strings).

### 2.2 `TimesModel` fields and their `times-data` counterparts

The table below maps each `TimesModel` field to the corresponding `times-data`
concept, showing where a direct programmatic bridge is feasible.

| `xl2times` `TimesModel` field | Type | `times-data` equivalent |
|---|---|---|
| `internal_regions` | `set[str]` | `ModelConfig.regions` |
| `all_regions` | `set[str]` | `ModelConfig.regions` + special trade regions (`IMPEXP`, `MINRNW`) |
| `processes` | `DataFrame` (columns: process, description, sets, primarycg, tslvl, tact, tcap, vintage, region) | `dict[str, Process]` — each `Process` has `name`, `description`, `process_type`, `inputs`, `outputs`, `regions`, `primary_group` |
| `commodities` | `DataFrame` (columns: commodity, description, csets, ctype, ctslvl, limtype, peakts, unit, region) | `dict[str, Commodity]` — each `Commodity` has `name`, `ctype` (enum), `timeslice_level`, `balance_type`, `unit` |
| `topology` | `DataFrame` (columns: region, process, commodity, io, csets) | Implicit in `Process.inputs` / `Process.outputs` (each `FlowSpec` has `commodity`, `group`, `efficiency`) |
| `trade` | `DataFrame` (origin, in, destination, out, process) | `_top_ire` passthrough list; `Process` objects with `process_type="IRE"` |
| `attributes` | `DataFrame` (region, process, commodity, attribute, year, value, limtype, timeslice, …) | `ParameterTable.values` — list of `ParameterValue(parameter, indexes, value)` |
| `user_constraints` / `uc_attributes` | `DataFrame` | `ParameterTable` entries for UC parameters (`UC_ACT`, `UC_FLO`, `UC_CAP`, …) |
| `ts_tslvl` / `ts_map` | `DataFrame` | `ModelConfig.timeslices` (nested `TimesliceLevel` tree) |
| `time_periods` | `DataFrame` (columns: d, b, e, m) | `ModelConfig.periods` (milestone years as a list) |
| `start_year` | `int` | `ModelConfig.start_year` |
| `units` | `DataFrame` (unit, type) | Derived from `Commodity.unit` + `ModelConfig.currencies` |
| `commodity_groups` | `DataFrame` | Derived from topology via `COM_GMAP` |

The mapping is close enough that a conversion function of a few hundred lines
could translate between the two representations without going through text
serialisation.

### 2.3 Veda transformation coverage as test-case generator

`xl2times` implements the full Veda tag set (~40 tags) and applies 50+
transformation stages to resolve aliases, fill defaults, expand wildcards,
merge scenarios, and build the final model topology. The `TimesModel` objects
produced from the DemoS benchmark models are the most complete open-source
representation of those models available. `times-data` could use them as
reference test cases for its own importer and validator.

### 2.4 Attribute alias and default mapping

`xl2times/config/veda-attr-defaults.json` maps 136 Veda attribute names to
their canonical TIMES parameter names (e.g., `ACTBND → ACT_BND`,
`AF → NCAP_AF`, `CEFF → ACT_EFF`) and records default values for bound types,
timeslice levels, and commodity-group handling. While the alias names are
Veda-specific, the underlying TIMES default values apply to any workflow and
could inform `times-data`'s validation rules for parameters.

---

## 3 Shared TIMES model-generator data

Both projects independently encode reference data that is defined by the TIMES
model generator itself (the GAMS source code maintained by IEA-ETSAP). The
table below lists the specific overlap areas, the files in each project, and
notes on reconciliation.

### 3.1 Parameter definitions

| | `xl2times` | `times-data` |
|---|---|---|
| **File** | `config/times-info.json` | `schema/parameters.py` (`PARAMETER_REGISTRY`) |
| **Count** | 283 parameters | 288 parameters |
| **Per entry** | name, GAMS category, GAMS indexes, column mapping | name, indexes, description, category, units/range/defaults, default I/E, related sets/params, affected equations, details |
| **Source** | Manually maintained JSON | Code-generated from `raw/reference/parameters.json` via `schema/generate.py` |

The five-parameter gap (283 vs 288) likely reflects version drift. Reconciling
the two lists would surface any parameters that `xl2times` is missing or that
`times-data` includes from a newer TIMES release.

### 3.2 Set enumerations

| | `xl2times` | `times-data` |
|---|---|---|
| **File** | `config/times-sets.json` | `schema/sets.py` (`SET_REGISTRY`) + `schema/indexes.py` (`INDEX_REGISTRY`) |
| **Scope** | 11 fixed enumerations (COM_TYPE, PRC_GRP, TSLVL, LIM, IMPEXP, IN_OUT, SIDE, UC_GRPTYPE, UC_NAME, NRG_TYPE, UPT) | 178 set definitions (including multi-dimensional sets with index signatures) + 37 index definitions with aliases and related indexes |

`xl2times` stores only the set _values_; `times-data` stores richer metadata
(descriptions, aliases, relationships, user-input vs. internal classification).
A shared dataset could serve both: `xl2times` would consume the value lists,
`times-data` would consume the full definitions.

### 3.3 Attribute defaults and aliases

| | `xl2times` | `times-data` |
|---|---|---|
| **File** | `config/veda-attr-defaults.json` | Partial — `ParameterDef.default_ie` field |
| **Content** | 136 entries mapping Veda alias → TIMES parameter name, plus defaults for `limtype`, `tslvl`, `cg`, `year2` | Each `ParameterDef` records `default_ie` (interpolation/extrapolation: STD, MIG, N/A, etc.) |

The Veda alias layer (e.g., `ACTBND → ACT_BND`) is `xl2times`-specific, but
the default bound types (`UP`, `FX`, `LO`) and default timeslice levels
(`ANNUAL`, `DAYNITE`) are properties of TIMES itself. These could be attached
to the shared parameter definitions.

### 3.4 DD output table schema

| | `xl2times` | `times-data` |
|---|---|---|
| **File** | `config/times_mapping.txt` (41 table definitions) | `compiler/dd_compiler.py` (hard-coded in `_write_derived` and `SET_ORDER`) |
| **Content** | Declarative mapping: `TABLE[GAMS_INDEXES] = SourceTable(Columns)` | Procedural: Python code that writes each SET/PARAMETER block |

Both need to agree on which GAMS sets and parameters appear in a DD file and
in what order. A shared, declarative table schema would make this agreement
explicit.

### 3.5 Benchmark models

| | `xl2times` | `times-data` |
|---|---|---|
| **File** | `benchmarks.yml` + `setup-benchmarks.sh` | `tests/test_demos_solve.py` |
| **Content** | References to DemoS_001–DemoS_012 repos, expected accuracy thresholds | Hard-coded expected objective values for DemoS_001–DemoS_007 |

Both projects clone and test against the same upstream IEA-ETSAP demonstration
model repositories. A shared benchmark registry — listing repo URLs, expected
objective values, and GAMS license requirements — would keep the two test
suites in sync.

### 3.6 Commodity type semantics

Both projects hard-code the same `COM_TYPE` → default-balance mapping:

```python
# times-data: model/commodity.py
_DEFAULT_BALANCE = {"NRG": "UP", "DEM": "UP", "ENV": "UP", "MAT": "FX", "FIN": "FX"}

# xl2times: derived from config/veda-attr-defaults.json + transforms.py logic
# COM_LIM defaults are applied per commodity type during processing
```

This is TIMES model-generator knowledge that should live in one place.

---

## 4 Proposed next steps

1. **Direct `TimesModel → times-data Model` adapter.** Write a function that
   converts `xl2times`'s `TimesModel` directly into a `times-data` `Model`,
   without DD serialisation. This would let users run
   `xl2times.main.read_xl(…)` and immediately query, validate, or re-export
   the model via `times-data`'s API.

2. **Extract shared TIMES reference data.** Move the parameter definitions,
   set enumerations, and commodity-type defaults into a standalone data package
   (or a shared repository under the `etsap-TIMES` organisation) that both
   projects consume as a dependency. The `times-data` code-generation approach
   (`schema/generate.py` reading from `raw/reference/*.json`) could serve as
   the upstream source.

3. **Cross-project benchmark CI.** Add a CI job that runs the full pipeline —
   `xl2times` reads DemoS Excel → `times-data` validates the output → GAMS
   solves → objective value compared — so regressions in either project are
   caught early.

4. **Reconcile parameter/set counts.** Compare `times-info.json` (283 params)
   against `PARAMETER_REGISTRY` (288 params) entry by entry. Document which
   parameters are missing from each side and whether the gap reflects a version
   difference or a deliberate omission.
