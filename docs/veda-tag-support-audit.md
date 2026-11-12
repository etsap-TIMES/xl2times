# Veda Tag Language support gap analysis

**Audit date:** 12 November 2026  
**Reference audited:** [Veda Tag Language reference](https://kanors-emr.github.io/Veda-documentation/tags/), Veda2.0 version 5.0 (November 2026)  
**Code audited:** the `xl2times` checkout including the pending `~TFM_MIG-AT` and `~TFM_MIG-TS` change in open PR #362; this is not yet a statement about a merged or released version.

## Scope and interpretation

This is a feature-gap inventory, not a claim that `xl2times` should implement
all of Veda. The project README describes Veda support as partial and says
support for other spreadsheet approaches may be added over time. The goal is to
identify which documented Veda inputs can be converted, which are only partly
represented, and which are not currently converted. LMA and report-definition
tags are excluded.

Veda's reference covers much more than model-data conversion: it includes
workbook synchronization, case/layer management, reports, parametric
recalculation, and writing data back into workbooks. A converter that produces
TIMES/GAMS data need not reproduce those Veda application features. The items
below distinguish conversion-relevant gaps from Veda-only behavior and use
“missing” to mean missing from xl2times's current conversion path, not
necessarily a defect.

The code observations refer to `xl2times/datatypes.py`, `xl2times/excel.py`,
`xl2times/main.py`, `xl2times/transforms.py`, and
`xl2times/config/veda-tags.json`. Tests and benchmark comparisons demonstrate
specific model behavior but are not an exhaustive Veda compatibility suite.

## Summary

The core Veda-to-TIMES path is present: flexible imports and declarations,
ordinary insert/update/migrate processing, user constraints, user sets,
process availability, basic trade links, and demand driver tables. The largest
conversion-impacting gaps are:

1. Topology additions are not applied, although topology filters depend on
   them.
2. Commodity repository availability and repository process replication are
   not implemented.
3. Several documented table layouts and trade-link forms are declared or
   described by Veda but are not normalized into xl2times's supported forms.
4. System settings and empty-index behavior are incomplete, particularly
   `~DefaultYear`, `~MileStoneYears`, region groups, default units, and
   region-specific timeslices.
5. Shared table semantics (especially tag text and workbook/tag compatibility)
   differ from the current Veda reference.
6. Metadata, parametric workbook generation, and fill-back tables are outside
   the converter's current capabilities.

These are candidates for staged work, not a recommendation to make
Veda-specific semantics mandatory for all input formats.

## Coverage by feature family

Status definitions: **Implemented** means there is an identifiable conversion
path; **Partial** means only a subset of the documented semantics is present;
**Missing** means there is no corresponding processing path or the input is
explicitly discarded.

| Family | Status | Current behavior and gap |
| --- | --- | --- |
| Excel table extraction and tag recognition | Partial | `extract_tables` scans worksheet cells and constructs tables from adjacent cells. It does not encode the reference's per-tag corner/floating table kind, and `Tag` is a closed enumeration. Unknown tags are warned about and dropped. See `excel.py:13-80`, `datatypes.py:20-78`, `transforms.py:218-250`. |
| Workbook type and tag admission | Partial | `DataModule` classifies files into broad categories, but `Config._read_veda_tags_info` does not load or enforce the catalog's `tag_allowed_in` restrictions. Veda's detailed distinction between layer/file types is therefore not validated at the tag boundary. See `datatypes.py:79-153`, `:599-725`. |
| Tag text and table-wide defaults | Partial | Key-value defaults are applied, but the code splits them on commas rather than Veda's documented semicolon separator (commas can belong to a value such as a region list). Token defaults are treated as attributes rather than inferred as region, year, timeslice, currency, limit type, or other indexes. The code itself has TODOs for alias/allowed-column handling and token resolution. See `utils.py:31-81`. |
| Headers, comments, and ignored columns | Partial | Header aliases and per-column comment-row markers are configured. The global `~NOpCol` behavior is missing, and only headers beginning with `*` are removed as comment columns. Veda also defines additional ignored-header forms and error-cell handling. See `transforms.py:116-143`, `datatypes.py:640-686`. |
| Files, layers, and seed values | Partial; needs conformance testing | xl2times groups inputs into modules and applies transforms in a fixed module order, but it does not implement Veda's full distinction among regular, NSV, parametric, and case layers. UPD/MIG seed selection also has a commented-out TODO for resolving the preferred source when multiple modules contain a match. If users supply several layers, output may differ from a Veda-synchronized case. See `datatypes.py:79-153`, `main.py:97-124`, `transforms.py:2730-2788`, `:2854-2995`. |
| Declarations and flexible imports (`~FI_Process`, `~FI_Comm`, `~FI_T`) | Implemented, with edge-case gaps | The pipeline processes declarations and FI data into model entities and topology. Verify Veda defaults, tag text, region-group expansion, headers, and less common FI attributes against representative workbooks; the current tests do not establish exhaustive conformance. See `transforms.py:491-600`, `:1677-1779`. |
| System time settings | Partial | `~StartYear`, `~ActivePDef`, and `~TimePeriods` drive the selected period structure. `~MileStoneYears`, `~EndYear`, and `~DefaultYear` do not have equivalent processing; missing years are defaulted to `model.start_year`. See `transforms.py:1031-1055`, `:1249-1287`. |
| Regions and units | Partial | `~BookRegions_Map` provides internal/external regions. There is no `~RegionGroup_Map` expansion. `~DefUnits` has no processing path, so sector-specific unit defaults are not read from that tag. `~UnitConversion` is intentionally dropped. See `transforms.py:845-890`, `:1058-1104`, `:218-234`. |
| Timeslices | Partial | The `~TimeSlices` tree is converted for all internal regions using fixed levels. Region-scoped trees are explicitly a TODO. See `transforms.py:3050-3170`. |
| Main data transformations (`~TFM_DINS`, `~TFM_INS`, `~TFM_UPD`, `~TFM_MIG`) | Implemented, with layout gaps | Base tags are normalized and applied. `-AT`/`-TS` layouts are normalized for DINS, INS, UPD, and MIG in this checkout. `-TSL` is configured for DINS/INS but is not expanded into long-form rows. See `datatypes.py:47-66`, `transforms.py:1993-2072`, `:2075-2182`, `:2854-2993`. |
| Topology edits (`~TFM_TOPINS`, `~TFM_TOPINS-A`, `~TFM_TOPDINS`) | Missing | `~TFM_TOPINS` enters the transform tag list but is sent to the dropped-table path. `~TFM_TOPDINS` and `~TFM_TOPINS-A` have no effective topology-application path. `create_model_topology` currently derives topology from `~FI_T`; the later topology-generation note also identifies TOPINS as unimplemented. This also makes `pset_ci` / `pset_co` filters incomplete when they rely on added links. See `transforms.py:1714-1779`, `:2082-2182`, `:1368-1370`. |
| Repository availability (`~TFM_AVA`, `~TFM_AVA-C`, `~ReplicateInRegions`) | Partial | Process availability (`~TFM_AVA`) filters repository processes and their data. Commodity availability has no corresponding enforcement; `~ReplicateInRegions` is not represented in `Tag`. See `datatypes.py:43-45`, `transforms.py:2185-2203`, `:3364-3413`. |
| Commodity/process groups and sets (`~TFM_COMGRP`, `~TFM_PSets`, `~TFM_CSets`) | Implemented, with limitations to verify | Commodity groups and user-defined process/commodity sets have processing paths. Veda's additional built-in groups, region scope, ordering, and all filter-combination rules should be validated separately before claiming full compatibility. See `transforms.py:1303-1372`, `:2310-2389`. |
| Trade | Partial | Matrix `~TradeLinks` is harmonized and row-form `~TradeLinks_DINS` is converted. `~TradeLinks_c1c2`, `~TradeLinks_Desc`, `~FI_Process_Trades`, and `~commodity_geo` are not represented in the tag enum/pipeline. See `datatypes.py:67-74`, `transforms.py:1467-1510`, `:1840-1910`. |
| User constraints (`~UC_T`, `~UC_Sets`, `~UC_GENT`) | Partial | `~UC_T` and user-constraint sets are processed, including UC metadata and topology verification. `~UC_GENT` is not represented. Period/timeslice domains and the association/precedence rules for `~UC_Sets` merit focused conformance tests. See `transforms.py:601-820`, `:3213-3271`. |
| Demand drivers (`~DRVR_Allocation`, `~DRVR_Table`, `~Series`) | Implemented, not exhaustively audited | A dedicated driver-table processing path exists and demand projections are generated. Veda's full driver/allocation semantics need comparison against real demand workbooks. See `transforms.py` `process_drvr_tables` and `_project_demands`; `main.py:170-175`. |
| Fill-back (`~TFM_FILL-R`) | Missing by design | Both legacy `~TFM_FILL` and `~TFM_FILL-R` are removed. Fill-R queries model values and writes them into the source workbook, which is not an output this converter currently produces. See `transforms.py:1612-1624`. |
| Parametric scenarios (`~INPUTCELL`) | Missing by design | There is no parametric controller/child-file generation or formula recalculation. `xl2times` converts the input workbook; it does not replay a spreadsheet for each run index. See `datatypes.py:20-78`, `excel.py:27-35`, `main.py:150-208`. |
| Metadata (`~MD-*`, `~MD_Dimensions`) | Missing | These tags do not have enum members or conversion paths. Decide whether provenance should be preserved in a sidecar/metadata output rather than silently treating it as model data. |
| Other system/catalog tags | Missing or intentionally irrelevant | `~NOpCol`, `~ParentRegions_Map`, `~VbeViews`, `~ScenMap`, and `~OsemosysCsv` have no corresponding conversion feature. `~UnitConversion` is explicitly discarded because it is for Veda result browsing. |
| Legacy and retired tags | Mixed | `~COMEMI` and `~COMAGG` are converted into FI data. Other legacy tags may be unsupported, intentionally retired, or not relevant to TIMES/GAMS conversion. Avoid implementing retired tags just to mirror catalog membership. See `transforms.py:1627-1674` and the Veda [legacy-tag guidance](https://kanors-emr.github.io/Veda-documentation/tags/not-recommended/). |

The current Veda catalog also contains model-input tags for which xl2times has
no conversion path: `~RegionGroup_Map`, `~ReplicateInRegions`,
`~TFM_TOPINS-A`, `~TradeLinks_c1c2`, `~TradeLinks_Desc`,
`~FI_Process_Trades`, and `~UC_GENT`. `~TFM_TOPDINS` is recognized in the
schema but is not applied as a topology edit. `~TFM_AVA-C` has an enum member
but no schema/enforcement path. Of the still-effectful legacy tags,
`~PRCCOMEMI` and `~COMAGG_R1C2` are not implemented. These are worth
prioritizing according to whether real input models depend on them; this list
does not imply that retired tags should be restored.

### Rule selection and conditions

The filter resolver intersects populated process/commodity filter columns,
which provides the usual AND behavior. The Veda reference also allows
per-row AND/OR controls; those control columns are present in some schemas but
are not used by the resolver. The local wildcard implementation handles
common `*`, `?`, `_`, comma-list, and exclusion forms, but does not implement
all documented `%` and escaped-special-character rules. See
`transforms.py:2220-2236`, `:2439-2508`, and `utils.py:250-339`.

UPD/MIG selection also needs a focused compatibility audit. The query path
uses equality membership for its `val_cond` input and does not pass
`attrib_cond` or the `avc_*` condition columns into the query. Arithmetic
operators are applied to existing values, but the documented `~` prefix for
an absolute value is not interpreted as an operator marker. These differences
can change which values are modified and what values are written. See
`transforms.py:2511-2561`, `:2730-2773`.

## Conversion-relevant gaps in more detail

### Topology edits

The Veda topology reference describes `~TFM_TOPINS`, `~TFM_TOPINS-A`, and
`~TFM_TOPDINS` as a way to add connections after declarations. These links
affect both generated topology and rule selection through `pset_ci` /
`pset_co`. xl2times currently constructs topology from FI data and drops
TOPINS rather than applying its rows. This is the highest-priority behavioral
gap because it can change which later transformation rows match.

Suggested investigation: implement enumerated TOPDINS first, then rule-based
TOPINS and compact TOPINS-A through a common topology representation; verify
that these links are visible before topology-based filters and that region,
process, commodity, direction, and ordering semantics match the intended
input.

### Repository contents

Veda's availability family includes process availability, commodity
availability, and process replication. xl2times implements only the first.
Availability-C should be added only with a clear model representation for
region-specific commodity existence and consistent filtering of dependent
data. Process replication is a larger transformation: it must create regional
process declarations and consistently rename/process associated topology and
attributes.

### Wide transformation layouts

The current branch handles attribute-wide and year-wide DINS/INS/UPD/MIG
layouts. DINS/INS `-TSL` tags appear in the schema but the variant normalizer
has no timeslice-wide case. These should be implemented as layout
normalization, not as separate transformation semantics, and tested for
headers that collide with ordinary index columns, empty cells, and tag
defaults.

### System settings and defaults

Veda settings have direct effects on the input-to-TIMES interpretation:
period definitions, default year, region groups, default units, and
region-specific time-slice maps. The current pipeline selects durations from
`~TimePeriods` and `~ActivePDef`, defaults missing years to the start year,
and expands one timeslice tree across internal regions. Missing settings can
therefore alter output values or model structure, rather than merely omit
Veda UI metadata.

The next step should be a settings compatibility table with a test fixture
for every setting, especially `~MileStoneYears`, `~DefaultYear`,
`~RegionGroup_Map`, `~DefUnits`, and region-scoped `~TimeSlices`.

### Tag text and table reading

The reference defines general tag-cell behavior, tag-text key/value and token
forms, precedence, table-kind-specific bounds, and the treatment of Excel
errors and comment columns. These are shared across many tags, so a parsing
difference can affect otherwise implemented features. In particular,
`utils.apply_composite_tag` currently comma-splits key/value text even though
Veda uses semicolons between settings and permits comma-separated values. It
also treats a token such as a currency or limit type as an attribute. This
should be fixed and tested centrally rather than worked around separately in
each tag handler.

Likewise, tag admission should use Veda's file-type restrictions only in the
Veda input adapter. This keeps validation useful for Veda models without
rejecting files or tag vocabularies from other TIMES specification
approaches.

### Other input models and generality

The audit should not turn `xl2times` into a Veda-only implementation.
Veda's filename/layer concepts and formula-driven controller behavior are
not prerequisites for other TIMES input methods. Keep common model concepts
(regions, period definitions, topology, attributes, and data layers) separate
from Veda-specific workbook parsing and compatibility rules. Optional
metadata or unsupported Veda features should be surfaced through diagnostics
or preserved as metadata where useful, rather than forcing every input
adapter to emulate them.

## Proposed plan

This is a sequencing proposal, not a commitment to implement every Veda
feature.

1. **Make support observable.** Maintain a tag capability matrix from the
   Veda catalog with statuses (converted, partially converted, ignored by
   design, unsupported). Add a clear warning/diagnostic when a recognized
   Veda tag has no conversion path; do not error on unknown tags from other
   spreadsheet approaches.
2. **Close structural and rule gaps.** Add TOPDINS then TOPINS/TOPINS-A;
   add DINS/INS `-TSL`; implement the documented filter combinators and
   UPD/MIG conditions; add regression tests proving topology edits influence
   `pset_ci` / `pset_co`. Add AVA-C only after defining region-specific
   commodity behavior.
3. **Correct model-defining settings.** Add fixture-backed support or explicit
   unsupported diagnostics for milestone-year definitions, default year,
   default units, region groups, and region-specific timeslices. Compare
   generated TIMES DD output with Veda-produced output.
4. **Audit existing tag families.** Build small, isolated golden fixtures for
   FI, trade, demand drivers, sets/groups, user constraints, MIG/UPD
   conditions, and layer/seed ordering. Focus on semantics that affect resulting
   TIMES tables; do not duplicate Veda's application/UI behavior.
5. **Decide on preservation versus implementation.** For metadata and
   workbook-oriented features such as FILL-R, parametric recalculation,
   VbeViews, and result unit conversion, decide whether to preserve useful
   information, report an unsupported feature, or document deliberate
   non-support. Implement workbook mutation/recalculation only if it becomes
   a clear product goal.
6. **Keep the input boundary extensible.** Put Veda-specific parsing behind
   the spreadsheet input layer and keep the internal model representation
   independent of Veda tag names. Add tests for a non-Veda input path whenever
   shared model structures are changed.

## References

- [Veda Tag Language overview and tag catalog](https://kanors-emr.github.io/Veda-documentation/tags/)
- [How a table is read](https://kanors-emr.github.io/Veda-documentation/tags/reading-tables/)
- [Values, wildcards, filters, and defaults](https://kanors-emr.github.io/Veda-documentation/tags/values-and-filters/)
- [System settings](https://kanors-emr.github.io/Veda-documentation/tags/system-settings/)
- [Declarations and flexible imports](https://kanors-emr.github.io/Veda-documentation/tags/res-templates/)
- [Insert and update layouts](https://kanors-emr.github.io/Veda-documentation/tags/data-tags/)
- [Topology](https://kanors-emr.github.io/Veda-documentation/tags/topology/)
- [Repository availability](https://kanors-emr.github.io/Veda-documentation/tags/availability/)
- [Trade](https://kanors-emr.github.io/Veda-documentation/tags/trade/)
- [User constraints](https://kanors-emr.github.io/Veda-documentation/tags/user-constraints/)
- [Sets and groups](https://kanors-emr.github.io/Veda-documentation/tags/sets-and-groups/)
- [Files and layers](https://kanors-emr.github.io/Veda-documentation/tags/files-and-layers/)
- [Processing order](https://kanors-emr.github.io/Veda-documentation/tags/processing-order/)
- [Fill](https://kanors-emr.github.io/Veda-documentation/tags/fill/)
- [Parametric scenarios](https://kanors-emr.github.io/Veda-documentation/tags/parametric/)
- [Metadata](https://kanors-emr.github.io/Veda-documentation/tags/metadata/)
- [Tags not recommended for new models](https://kanors-emr.github.io/Veda-documentation/tags/not-recommended/)
