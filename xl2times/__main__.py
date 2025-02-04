import argparse
import hashlib
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pandas.core.frame import DataFrame

from . import excel, transforms, utils
from .datatypes import Config, DataModule, EmbeddedXlTable, TimesModel

_log_sep = "=" * 80 + "\n"


cache_dir = Path.home() / ".cache/xl2times/"
cache_dir.mkdir(exist_ok=True, parents=True)


def invalidate_cache(max_age: timedelta = timedelta(days=365)):
    """
    Delete any cache files older than max_age.

    Args:
        max_age: Maximum age of a cache file to be considered valid. Any cache files older than this are deleted.
    """
    for file in cache_dir.glob("*.pkl"):
        if datetime.now() - datetime.fromtimestamp(file.lstat().st_mtime) > max_age:
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete old cache file {file}. {e}")


def _read_xlsx_cached(filename: str | Path) -> list[EmbeddedXlTable]:
    """Extract EmbeddedXlTables from xlsx file (cached).

    Since excel.extract_tables is quite slow, we cache its results in `cache_dir`.
    Each cache file is named {filename}_{hash}.pkl, and contains a pickled
    `[EmbeddedXlTable]`.

    Args:
        filename: Path to the xlsx file to extract tables from.
    """
    filename = Path(filename).resolve()
    with filename.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256")  # pyright: ignore
    hsh = digest.hexdigest()
    cached_file = (cache_dir / f"{Path(filename).stem}_{hsh}.pkl").resolve()

    if cached_file.exists():
        # just load and return the cached pickle
        with cached_file.open("rb") as f:
            tables = pickle.load(f)
            logger.info(f"Using cached data for {filename} from {cached_file}")
    else:
        # extract data and write it to cache before returning it
        tables = excel.extract_tables(str(filename))
        with cached_file.open("wb") as f:
            pickle.dump(tables, f)
        logger.info(f"Saved cache for {filename} to {cached_file}")

    return tables


def convert_xl_to_times(
    input_files: list[str],
    output_dir: str,
    config: Config,
    model: TimesModel,
    no_cache: bool,
    stop_after_read: bool = False,
) -> dict[str, DataFrame]:
    start_time = datetime.now()

    invalidate_cache()
    with ProcessPoolExecutor(utils.max_workers) as executor:
        raw_tables = executor.map(
            excel.extract_tables if no_cache else _read_xlsx_cached, input_files
        )
    # raw_tables is a list of lists, so flatten it:
    raw_tables = [t for ts in raw_tables for t in ts]
    logger.info(
        f"Extracted (potentially cached) {len(raw_tables)} tables,"
        f" {sum(table.dataframe.shape[0] for table in raw_tables)} rows"
        f" in {datetime.now() - start_time}"
    )

    if stop_after_read:
        # Convert absolute paths to relative paths to enable comparing raw_tables.txt across machines
        raw_tables.sort(key=lambda x: (x.filename, x.sheetname, x.range))
        input_dir = os.path.commonpath([t.filename for t in raw_tables])
        raw_tables = [strip_filename_prefix(t, input_dir) for t in raw_tables]

    dump_tables(raw_tables, os.path.join(output_dir, "raw_tables.txt"))
    if stop_after_read:
        return {}

    transform_list = [
        transforms.normalize_tags_columns,
        transforms.remove_fill_tables,
        lambda config, tables, model: [
            transforms.remove_comment_cols(t) for t in tables
        ],
        transforms.validate_input_tables,
        transforms.remove_tables_with_formulas,  # slow
        transforms.normalize_column_aliases,
        transforms.remove_comment_rows,
        transforms.revalidate_input_tables,
        transforms.capitalise_table_values,
        transforms.process_regions,
        transforms.convert_com_tables,
        transforms.process_time_periods,
        transforms.remove_exreg_cols,
        transforms.generate_dummy_processes,
        transforms.process_time_slices,
        transforms.process_transform_table_variants,
        transforms.apply_tag_specified_defaults,
        transforms.process_transform_tables,
        transforms.process_transform_availability,
        transforms.process_flexible_import_tables,  # slow
        transforms.process_user_constraint_tables,
        transforms.harmonise_tradelinks,
        transforms.include_tables_source,
        transforms.process_processes,
        transforms.fill_in_column_defaults,
        transforms.create_model_topology,
        transforms.generate_uc_properties,
        transforms.expand_rows_parallel,  # slow
        transforms.process_tradelinks,
        transforms.merge_tables,
        transforms.remove_invalid_values,
        transforms.include_cgs_in_topology,
        transforms.fill_in_missing_pcgs,
        transforms.complete_processes,
        transforms.create_model_units,
        transforms.process_wildcards,
        transforms.convert_aliases,
        transforms.enforce_availability,
        transforms.complete_model_trade,
        transforms.create_model_cgs,
        transforms.prepare_for_querying,
        transforms.apply_transform_tables,
        transforms.generate_implied_topology,
        transforms.verify_uc_topology,
        transforms.explode_process_commodity_cols,
        transforms.apply_final_fixup,
        transforms.assign_model_attributes,
        transforms.resolve_remaining_cgs,
        lambda config, tables, model: dump_tables(
            tables, os.path.join(output_dir, "merged_tables.txt")
        ),
        transforms.complete_dictionary,
        transforms.convert_to_string,
        lambda config, tables, model: produce_times_tables(config, tables, model),
    ]

    input = raw_tables
    output = {}
    for transform in transform_list:
        start_time = time.time()
        output = transform(config, input, model)
        end_time = time.time()
        logger.opt(raw=True).debug(_log_sep)
        logger.info(
            f"transform {transform.__code__.co_name} took {end_time - start_time:.2f} seconds"
        )
        logger.opt(raw=True).debug(_log_sep)
        # Way to conditionally evaluate the table dump only on debug log level
        # https://loguru.readthedocs.io/en/stable/overview.html#lazy-evaluation-of-expensive-functions
        logger.opt(lazy=True).debug(
            "All tables:\n{dump}", dump=lambda: _all_table_dump(output)
        )
        input = output
    assert isinstance(output, dict)

    logger.info(
        f"Conversion complete, {len(output)} tables produced,"
        f" {sum(df.shape[0] for df in output.values())} rows"
    )

    return output


def _all_table_dump(tables: list[EmbeddedXlTable] | dict[str, DataFrame]) -> str:
    """A dump of current values in all tables, for debugging."""
    result = StringIO()
    if isinstance(tables, list):
        for table in sorted(
            tables, key=lambda t: (t.tag, t.filename, t.sheetname, t.range)
        ):
            result.write(str(table))
            result.write("\n")
    elif isinstance(tables, dict):
        for tag, df in tables.items():
            df_str = df.to_csv(index=False, lineterminator="\n")
            result.write(f"{tag}\n{df_str}{df.shape}\n")
    return result.getvalue()


def write_csv_tables(tables: dict[str, DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".csv"):
            os.remove(os.path.join(output_dir, item))
    for tablename, df in tables.items():
        df.to_csv(os.path.join(output_dir, tablename + "_output.csv"), index=False)
    logger.success(
        f"Excel files successfully converted to CSV and written to {output_dir}"
    )


def read_csv_tables(input_dir: str) -> dict[str, DataFrame]:
    result = {}
    csv_files = list(Path(input_dir).glob("*.csv"))
    for filename in csv_files:
        result[filename.stem] = pd.read_csv(filename)
    return result


def compare(
    data: dict[str, DataFrame], ground_truth: dict[str, DataFrame], output_dir: str
) -> str:
    logger.info(
        f"Ground truth contains {len(ground_truth)} tables,"
        f" {sum(df.shape[0] for _, df in ground_truth.items())} rows"
    )

    missing = set(ground_truth.keys()).difference(data.keys())
    missing_str = ", ".join(
        [f"{x} ({ground_truth[x].shape[0]})" for x in sorted(missing)]
    )
    if len(missing) > 0:
        logger.info(f"Missing {len(missing)} tables: {missing_str}")

    additional_tables = set(data.keys()).difference(ground_truth.keys())
    additional_str = ", ".join(
        [f"{x} ({data[x].shape[0]})" for x in sorted(additional_tables)]
    )
    if len(additional_tables) > 0:
        logger.info(f"{len(additional_tables)} additional tables: {additional_str}")
    # Additional rows starts as the sum of lengths of additional tables produced
    total_additional_rows = sum(len(data[x]) for x in additional_tables)

    total_gt_rows = 0
    total_correct_rows = 0
    for table_name, gt_table in sorted(
        ground_truth.items(), reverse=True, key=lambda t: len(t[1])
    ):
        if table_name in data:
            data_table = data[table_name]

            # Remove .integer suffix added to duplicate column names by CSV reader (mangle_dupe_cols=False not supported)
            transformed_gt_cols = [col.split(".")[0] for col in gt_table.columns]
            data_cols = list(data_table.columns)
            if transformed_gt_cols != data_cols:
                logger.info(
                    f"Table {table_name} header incorrect, was"
                    f" {data_cols}, should be {transformed_gt_cols}"
                )

            # Convert rows to lowercase strings for case-insensitive comparison
            gt_rows = set(str(row).lower() for row in gt_table.to_numpy().tolist())
            data_rows = set(str(row).lower() for row in data_table.to_numpy().tolist())
            total_gt_rows += len(gt_rows)
            total_correct_rows += len(gt_rows.intersection(data_rows))
            additional = data_rows.difference(gt_rows)
            total_additional_rows += len(additional)
            missing = gt_rows.difference(data_rows)
            if len(additional) != 0 or len(missing) != 0:
                logger.info(
                    f"Table {table_name} ({data_table.shape[0]} rows,"
                    f" {gt_table.shape[0]} GT rows) contains {len(additional)}"
                    f" additional rows and is missing {len(missing)} rows"
                )
            if len(additional) != 0:
                DataFrame(additional).to_csv(
                    os.path.join(output_dir, table_name + "_additional.csv"),
                    index=False,
                )
            if len(missing) != 0:
                DataFrame(missing).to_csv(
                    os.path.join(output_dir, table_name + "_missing.csv"),
                    index=False,
                )
    result = (
        f"{(total_correct_rows / total_gt_rows) if total_gt_rows!=0 else np.nan :.1%} of ground truth rows present"
        f" in output ({total_correct_rows}/{total_gt_rows})"
        f", {total_additional_rows} additional rows"
    )

    logger.success(result)
    return result


def produce_times_tables(
    config: Config, input: dict[str, DataFrame], model: TimesModel
) -> dict[str, DataFrame]:
    logger.info(
        f"produce_times_tables: {len(input)} tables incoming,"
        f" {sum(len(value) for (_, value) in input.items())} rows"
    )
    file_order = defaultdict(lambda: -1)
    for i, f in enumerate(model.files):
        file_order[f] = i
    # Keep only those mappings for which parameters that are defined in the input
    par_tables = {"Attributes", "UCAttributes"}.intersection(input.keys())
    defined_pars = set()
    for table in par_tables:
        defined_pars = defined_pars.union(set(input[table]["attribute"]))
    mappings = [
        m
        for m in config.times_xl_maps
        if m.xl_name not in par_tables or m.filter_rows.get("attribute") in defined_pars
    ]

    def keep_last_by_file_order(df):
        """Drop duplicate rows, keeping the last dupicate row (including value) as per
        input file order, and remove the `source_filename` column from the DataFrame.

        Note: we do not remove duplicate values for the same query columns for parameters
        here, because in the future we might want to re-use the processed tables and
        select the rows coming from different scenarios/files after processing just once.
        If so, at that point we can use the info in the `source_filename` column to do
        this.
        """
        if "source_filename" in df.columns:
            df["file_order"] = df["source_filename"].map(file_order)
            df = df.sort_values(by="file_order", kind="stable")
            df = df.drop(columns=["source_filename", "file_order"])
        df = df.drop_duplicates(keep="last")
        return df.reset_index(drop=True)

    result = {}
    used_tables = set()
    for mapping in mappings:
        if mapping.xl_name not in input:
            logger.info(
                f"Cannot produce table {mapping.times_name} because"
                f" {mapping.xl_name} does not exist"
            )
        else:
            used_tables.add(mapping.xl_name)
            df = input[mapping.xl_name]
            # Filter rows according to filter_rows mapping:
            for filter_col, filter_val in mapping.filter_rows.items():
                if filter_col not in df.columns:
                    logger.info(
                        f"Cannot produce table {mapping.times_name} because"
                        f" {mapping.xl_name} does not contain column {filter_col}"
                    )
                    # TODO break this loop and continue outer loop?
                filter = set(x.lower() for x in (filter_val,))
                i = df[filter_col].str.lower().isin(filter)
                df = df[i]
            if not set(mapping.xl_cols).issubset(df.columns):
                missing = set(mapping.xl_cols).difference(df.columns)
                logger.info(
                    f"Cannot produce table {mapping.times_name} because"
                    f" {mapping.xl_name} does not contain the required columns"
                    f" - {', '.join(missing)}"
                )
            else:
                # Ensure that df is not a view
                df = df.reset_index(drop=True)
                # Excel columns can be duplicated into multiple Times columns
                for times_col, xl_col in mapping.col_map.items():
                    df[times_col] = df[xl_col]
                # Keep only the required columns
                cols_to_keep = set(mapping.times_cols).union({"source_filename"})
                cols_to_drop = [x for x in df.columns if x not in cols_to_keep]
                df = df.drop(columns=cols_to_drop)
                # Drop duplicates, keeping last seen rows as per file order
                df = keep_last_by_file_order(df)
                # Drop rows with missing values
                # TODO this is a hack. Use pd.StringDtype() so that notna() is sufficient
                i = (
                    df[mapping.times_cols[-1]].notna()
                    & (df != "None").all(axis=1)
                    & (df != "nan").all(axis=1)
                    & (df != "").all(axis=1)
                    & (df != "<NA>").all(axis=1)
                )
                df = df.loc[i, mapping.times_cols]
                # Drop tables that are empty after filtering and dropping Nones:
                if len(df) == 0:
                    continue
                result[mapping.times_name] = df

    unused_tables = set(input.keys()).difference(used_tables)
    if len(unused_tables) > 0:
        logger.info(
            f"{len(unused_tables)} unused tables: {', '.join(sorted(unused_tables))}"
        )

    return result


def write_dd_files(tables: dict[str, DataFrame], config: Config, output_dir: str):
    encoding = "utf-8"
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".dd"):
            os.remove(os.path.join(output_dir, item))

    def convert_set(df: DataFrame):
        has_description = "TEXT" in df.columns
        # Remove duplicate rows, ignoring text column
        if has_description:
            query_columns = [c for c in df.columns if c != "TEXT"]
            df = df.drop_duplicates(subset=query_columns, keep="last")
        for row in df.itertuples(index=False):
            row_str = "'.'".join(
                (str(x) for k, x in row._asdict().items() if k != "TEXT")
            )
            desc = f" '{row.TEXT}'" if has_description else ""
            yield f"'{row_str}'{desc}\n"

    def convert_parameter(tablename: str, df: DataFrame):
        if "VALUE" not in df.columns:
            raise KeyError(f"Unable to find VALUE column in parameter {tablename}")
        # Remove duplicate rows, ignoring value column
        query_columns = [c for c in df.columns if c != "VALUE"] or None
        df = df.drop_duplicates(subset=query_columns, keep="last")
        for row in df.itertuples(index=False):
            val = row.VALUE
            row_str = "'.'".join(
                (str(x) for k, x in row._asdict().items() if k != "VALUE")
            )
            yield f"'{row_str}' {val}\n" if row_str else f"{val}\n"

    sets = {m.times_name for m in config.times_xl_maps if "VALUE" not in m.col_map}

    # Compute map fname -> tables: put ALL_TS and MILESTONYR in separate files
    tables_in_file = {
        "ts.dd": ["ALL_TS"],
        "milestonyr.dd": ["MILESTONYR"],
        "output.dd": [
            t for t in config.dd_table_order if t not in {"ALL_TS", "MILESTONYR"}
        ],
    }

    for fname, tablenames in tables_in_file.items():
        with open(os.path.join(output_dir, fname), "w", encoding=encoding) as fout:
            # Include GAMS dollar control options
            if fname == "output.dd":
                fout.write("$ONEPS\n$ONWARNING\n\n")
            for tablename in [t for t in tablenames if t in tables]:
                df = tables[tablename]
                if tablename in sets:
                    fout.write(f"SET {tablename}\n/\n")
                    lines = convert_set(df)
                else:
                    fout.write(f"PARAMETER\n{tablename} ' '/\n")
                    lines = convert_parameter(tablename, df)
                # Sort lines to ensure consistent output, except for ALL_TS
                if tablename != "ALL_TS":
                    lines = sorted(lines)
                fout.writelines(lines)
                fout.write("\n/;\n")

    logger.success(
        f"Excel files successfully converted to DD and written to {output_dir}"
    )


def strip_filename_prefix(table, prefix):
    if isinstance(table, EmbeddedXlTable):
        if table.filename.startswith(prefix):
            table.filename = table.filename[len(prefix) + 1 :]
    return table


def dump_tables(tables: list, filename: str) -> list:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as text_file:
        for t in tables if isinstance(tables, list) else tables.items():
            if isinstance(t, EmbeddedXlTable):
                tag = t.tag
                text_file.write(f"sheetname: {t.sheetname}\n")
                text_file.write(f"range: {t.range}\n")
                text_file.write(f"filename: {t.filename}\n")
                if t.uc_sets:
                    text_file.write(f"uc_sets: {t.uc_sets}\n")
                df = t.dataframe
            else:
                tag = t[0]
                df = t[1]
            text_file.write(f"tag: {tag}\n")
            types = ", ".join([f"{i} ({v})" for i, v in df.dtypes.items()])
            text_file.write(f"types: {types}\n")
            text_file.write(df.to_csv(index=False, lineterminator="\n"))
            text_file.write("\n" * 2)

    return tables


def run(args: argparse.Namespace) -> str | None:
    """Runs the xl2times conversion.

    Parameters
    ----------
    args
        Pre-parsed command line arguments

    Returns
    -------
        comparison with ground-truth string if `ground_truth_dir` is provided, else None.
    """
    utils.setup_logger(args.verbose)

    config = Config(
        "times_mapping.txt",
        "times-info.json",
        "times-sets.json",
        "veda-tags.json",
        "veda-attr-defaults.json",
        args.regions,
        args.include_dummy_imports,
    )

    model = TimesModel()

    if len(args.input) == 1:
        assert os.path.isdir(args.input[0])
        input_files = [
            str(path)
            for path in Path(args.input[0]).rglob("*")
            if path.suffix in [".xlsx", ".xlsm"] and not path.name.startswith("~")
        ]
        if utils.is_veda_based(input_files):
            input_files = utils.filter_veda_filename_patterns(input_files)
        logger.info(f"Loading {len(input_files)} files from {args.input[0]}")
    else:
        input_files = args.input

    model.files = [Path(path).stem for path in input_files]

    processing_order = ["base", "subres", "trade", "demand", "scen", "syssettings"]
    for data_module in processing_order:
        model.data_modules = model.data_modules + sorted(
            [
                item
                for item in {
                    DataModule.module_name(path)
                    for path in input_files
                    if DataModule.module_type(path) == data_module
                }
                if item is not None
            ]
        )

    if args.only_read:
        tables = convert_xl_to_times(
            input_files,
            args.output_dir,
            config,
            model,
            args.no_cache,
            stop_after_read=True,
        )
        sys.exit(0)

    tables = convert_xl_to_times(
        input_files, args.output_dir, config, model, args.no_cache
    )

    if args.dd:
        write_dd_files(tables, config, args.output_dir)
    else:
        write_csv_tables(tables, args.output_dir)

    if args.ground_truth_dir:
        ground_truth = read_csv_tables(args.ground_truth_dir)
        # Use the same convert_to_string transform on GT so that comparisons are fair
        ground_truth = transforms.convert_to_string(config, ground_truth, model)
        comparison = compare(tables, ground_truth, args.output_dir)
        return comparison
    else:
        return None


def parse_args(arg_list: None | list[str]) -> argparse.Namespace:
    """Parses command line arguments.

    Parameters
    ----------
    arg_list
        List of command line arguments. Uses sys.argv (default argparse behaviour) if `None`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "input",
        nargs="*",
        help="Either an input directory, or a list of input xlsx/xlsm files to process",
    )
    args_parser.add_argument(
        "--regions",
        type=str,
        default="",
        help="Comma-separated list of regions to include in the model",
    )
    args_parser.add_argument(
        "--include_dummy_imports",
        action="store_true",
        help="Include dummy import processes in the model",
    )
    args_parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    args_parser.add_argument(
        "--ground_truth_dir",
        type=str,
        help="Ground truth directory to compare with output",
    )
    args_parser.add_argument("--dd", action="store_true", help="Output DD files")
    args_parser.add_argument(
        "--only_read",
        action="store_true",
        help="Read xlsx/xlsm files and stop after outputting raw_tables.txt",
    )
    args_parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Ignore cache and re-extract tables from XLSX files",
    )
    args_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Verbosity. Multiple `-v`s increase the log level. Can also be set on the command line by setting the environment variable `LOGURU_LEVEL`. Available levels are `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, and `CRITICAL`. Default is `SUCCESS`",
    )
    args = args_parser.parse_args(arg_list)
    if not isinstance(args.input, list) or len(args.input) < 1:
        print("ERROR: expected at least 1 input.")
        args_parser.print_help()
        sys.exit(-1)
    return args


def main(arg_list: None | list[str] = None) -> None:
    """Main entry point for the xl2times package.

    Returns
    -------
        None.
    """
    args = parse_args(arg_list)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
