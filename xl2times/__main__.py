import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
from pandas.core.frame import DataFrame
import pandas as pd
import pickle
from pathlib import Path
import os
import sys
import time
from typing import Dict, List

from xl2times import __file__ as xl2times_file_path
from xl2times.utils import max_workers
from . import datatypes, utils
from . import excel
from . import transforms

logger = utils.get_logger()


cache_dir = os.path.abspath(os.path.dirname(xl2times_file_path)) + "/.cache/"
os.makedirs(cache_dir, exist_ok=True)


def _read_xlsx_cached(filename: str) -> List[datatypes.EmbeddedXlTable]:
    """Extract EmbeddedXlTables from xlsx file (cached).

    Since excel.extract_tables is quite slow, we cache its results in `cache_dir`.
    Each file is named by the hash of the contents of an xlsx file, and contains
    a tuple (filename, modified timestamp, [EmbeddedXlTable]).
    """
    with open(filename, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")  # pyright: ignore
    hsh = digest.hexdigest()
    if os.path.isfile(cache_dir + hsh):
        fname1, _timestamp, tables = pickle.load(open(cache_dir + hsh, "rb"))
        # In the extremely unlikely event that we have a hash collision, also check that
        # the filename is the same:
        # TODO check modified timestamp also matches
        if filename == fname1:
            logger.info(f"Using cached data for {filename} from {cache_dir + hsh}")
            return tables
    # Write extracted data to cache:
    tables = excel.extract_tables(filename)
    pickle.dump((filename, "TODO ModifiedTime", tables), open(cache_dir + hsh, "wb"))
    logger.info(f"Saved cache for {filename} to {cache_dir + hsh}")
    return excel.extract_tables(filename)


def convert_xl_to_times(
    input_files: List[str],
    output_dir: str,
    config: datatypes.Config,
    model: datatypes.TimesModel,
    no_cache: bool,
    verbose: bool = False,
    stop_after_read: bool = False,
) -> Dict[str, DataFrame]:
    start_time = datetime.now()
    with ProcessPoolExecutor(max_workers) as executor:
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
        transforms.process_regions,
        transforms.process_time_periods,
        transforms.remove_exreg_cols,
        transforms.generate_dummy_processes,
        transforms.process_time_slices,
        transforms.process_transform_table_variants,
        transforms.process_transform_tables,
        transforms.process_tradelinks,
        transforms.process_processes,
        transforms.process_topology,
        transforms.process_flexible_import_tables,  # slow
        transforms.process_user_constraint_tables,
        transforms.process_commodity_emissions,
        transforms.process_commodities,
        transforms.process_transform_availability,
        transforms.fill_in_missing_values,
        transforms.generate_uc_properties,
        transforms.expand_rows_parallel,  # slow
        transforms.remove_invalid_values,
        transforms.capitalise_some_values,
        transforms.apply_fixups,
        transforms.generate_commodity_groups,
        transforms.fill_in_missing_pcgs,
        transforms.generate_trade,
        transforms.include_tables_source,
        transforms.merge_tables,
        transforms.complete_processes,
        transforms.apply_more_fixups,
        transforms.process_units,
        transforms.process_years,
        transforms.complete_commodity_groups,
        transforms.process_uc_wildcards,
        transforms.process_wildcards,
        transforms.convert_aliases,
        transforms.fix_topology,
        transforms.resolve_remaining_cgs,
        transforms.complete_dictionary,
        transforms.convert_to_string,
        lambda config, tables, model: dump_tables(
            tables, os.path.join(output_dir, "merged_tables.txt")
        ),
        lambda config, tables, model: produce_times_tables(config, tables),
    ]

    input = raw_tables
    output = {}
    for transform in transform_list:
        start_time = time.time()
        output = transform(config, input, model)
        end_time = time.time()
        sep = "\n\n" + "=" * 80 + "\n" if verbose else ""
        logger.info(
            f"{sep}transform {transform.__code__.co_name} took {end_time - start_time:.2f} seconds"
        )
        if verbose:
            if isinstance(output, list):
                for table in sorted(
                    output, key=lambda t: (t.tag, t.filename, t.sheetname, t.range)
                ):
                    logger.info(table)
            elif isinstance(output, dict):
                for tag, df in output.items():
                    df_str = df.to_csv(index=False, lineterminator="\n")
                    logger.info(f"{tag}\n{df_str}{df.shape}\n")
        input = output
    assert isinstance(output, dict)

    logger.info(
        f"Conversion complete, {len(output)} tables produced,"
        f" {sum(df.shape[0] for df in output.values())} rows"
    )

    return output


def write_csv_tables(tables: Dict[str, DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".csv"):
            os.remove(os.path.join(output_dir, item))
    for tablename, df in tables.items():
        df.to_csv(os.path.join(output_dir, tablename + "_output.csv"), index=False)


def read_csv_tables(input_dir: str) -> Dict[str, DataFrame]:
    result = {}
    for filename in os.listdir(input_dir):
        result[filename.split(".")[0]] = pd.read_csv(
            os.path.join(input_dir, filename), dtype=str
        )
    return result


def compare(
    data: Dict[str, DataFrame], ground_truth: Dict[str, DataFrame], output_dir: str
) -> str:
    logger.info(
        f"Ground truth contains {len(ground_truth)} tables,"
        f" {sum(df.shape[0] for _, df in ground_truth.items())} rows"
    )

    missing = set(ground_truth.keys()) - set(data.keys())
    missing_str = ", ".join(
        [f"{x} ({ground_truth[x].shape[0]})" for x in sorted(missing)]
    )
    if len(missing) > 0:
        logger.warning(f"Missing {len(missing)} tables: {missing_str}")

    additional_tables = set(data.keys()) - set(ground_truth.keys())
    additional_str = ", ".join(
        [f"{x} ({data[x].shape[0]})" for x in sorted(additional_tables)]
    )
    if len(additional_tables) > 0:
        logger.warning(f"{len(additional_tables)} additional tables: {additional_str}")
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
                logger.warning(
                    f"Table {table_name} header incorrect, was"
                    f" {data_cols}, should be {transformed_gt_cols}"
                )

            # both are in string form so can be compared without any issues
            gt_rows = set(tuple(row) for row in gt_table.to_numpy().tolist())
            data_rows = set(tuple(row) for row in data_table.to_numpy().tolist())
            total_gt_rows += len(gt_rows)
            total_correct_rows += len(gt_rows.intersection(data_rows))
            additional = data_rows - gt_rows
            total_additional_rows += len(additional)
            missing = gt_rows - data_rows
            if len(additional) != 0 or len(missing) != 0:
                logger.warning(
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
        f"{total_correct_rows / total_gt_rows :.1%} of ground truth rows present"
        f" in output ({total_correct_rows}/{total_gt_rows})"
        f", {total_additional_rows} additional rows"
    )

    logger.info(result)
    return result


def produce_times_tables(
    config: datatypes.Config, input: Dict[str, DataFrame]
) -> Dict[str, DataFrame]:
    logger.info(
        f"produce_times_tables: {len(input)} tables incoming,"
        f" {sum(len(value) for (_, value) in input.items())} rows"
    )
    result = {}
    used_tables = set()
    for mapping in config.times_xl_maps:
        if mapping.xl_name not in input:
            logger.warning(
                f"Cannot produce table {mapping.times_name} because"
                f" {mapping.xl_name} does not exist"
            )
        else:
            used_tables.add(mapping.xl_name)
            df = input[mapping.xl_name].copy()
            # Filter rows according to filter_rows mapping:
            for filter_col, filter_val in mapping.filter_rows.items():
                if filter_col not in df.columns:
                    logger.warning(
                        f"Cannot produce table {mapping.times_name} because"
                        f" {mapping.xl_name} does not contain column {filter_col}"
                    )
                    # TODO break this loop and continue outer loop?
                filter = set(x.lower() for x in {filter_val})
                i = df[filter_col].str.lower().isin(filter)
                df = df.loc[i, :]
            # TODO find the correct tech group
            if "techgroup" in mapping.xl_cols:
                df["techgroup"] = df["techname"]
            if not all(c in df.columns for c in mapping.xl_cols):
                missing = set(mapping.xl_cols) - set(df.columns)
                logger.warning(
                    f"Cannot produce table {mapping.times_name} because"
                    f" {mapping.xl_name} does not contain the required columns"
                    f" - {', '.join(missing)}"
                )
            else:
                # Excel columns can be duplicated into multiple Times columns
                for times_col, xl_col in mapping.col_map.items():
                    df[times_col] = df[xl_col]
                cols_to_drop = [x for x in df.columns if x not in mapping.times_cols]
                df.drop(columns=cols_to_drop, inplace=True)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                # TODO this is a hack. Use pd.StringDtype() so that notna() is sufficient
                i = (
                    df[mapping.times_cols[-1]].notna()
                    & (df != "None").all(axis=1)
                    & (df != "nan").all(axis=1)
                    & (df != "").all(axis=1)
                )
                df = df.loc[i, mapping.times_cols]
                # Drop tables that are empty after filtering and dropping Nones:
                if len(df) == 0:
                    continue
                result[mapping.times_name] = df

    unused_tables = set(input.keys()) - used_tables
    if len(unused_tables) > 0:
        logger.warning(
            f"{len(unused_tables)} unused tables: {', '.join(sorted(unused_tables))}"
        )

    return result


def write_dd_files(
    tables: Dict[str, DataFrame], config: datatypes.Config, output_dir: str
):
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
        "output.dd": [t for t in config.dd_table_order if t != "ALL_TS"],
    }

    for fname, tablenames in tables_in_file.items():
        with open(os.path.join(output_dir, fname), "w") as fout:
            for tablename in [t for t in tablenames if t in tables]:
                df = tables[tablename]
                if tablename in sets:
                    fout.write(f"SET {tablename}\n/\n")
                    lines = convert_set(df)
                else:
                    fout.write(f"PARAMETER\n{tablename} ' '/\n")
                    lines = convert_parameter(tablename, df)
                fout.writelines(sorted(lines))
                fout.write("\n/;\n")
    pass


def strip_filename_prefix(table, prefix):
    if isinstance(table, datatypes.EmbeddedXlTable):
        if table.filename.startswith(prefix):
            table.filename = table.filename[len(prefix) + 1 :]
    return table


def dump_tables(tables: List, filename: str) -> List:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as text_file:
        for t in tables if isinstance(tables, List) else tables.items():
            if isinstance(t, datatypes.EmbeddedXlTable):
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
    """
    Runs the xl2times conversion.
    Args:
         args: pre-parsed command line arguments
    Returns:
         comparison with ground-truth string if `ground_truth_dir` is provided, else None.
    """
    config = datatypes.Config(
        "times_mapping.txt",
        "times-info.json",
        "times-sets.json",
        "veda-tags.json",
        "veda-attr-defaults.json",
        args.regions,
    )

    model = datatypes.TimesModel()

    if not isinstance(args.input, list) or len(args.input) < 1:
        logger.critical(f"expected at least 1 input. Got {args.input}")
        sys.exit(-1)
    elif len(args.input) == 1:
        assert os.path.isdir(args.input[0])
        input_files = [
            str(path)
            for path in Path(args.input[0]).rglob("*")
            if path.suffix in [".xlsx", ".xlsm"] and not path.name.startswith("~")
        ]
        logger.info(f"Loading {len(input_files)} files from {args.input[0]}")
    else:
        input_files = args.input

    model.files.update([Path(path).stem for path in input_files])

    if args.only_read:
        tables = convert_xl_to_times(
            input_files,
            args.output_dir,
            config,
            model,
            args.no_cache,
            verbose=args.verbose,
            stop_after_read=True,
        )
        sys.exit(0)

    tables = convert_xl_to_times(
        input_files, args.output_dir, config, model, args.no_cache, verbose=args.verbose
    )

    if args.dd:
        write_dd_files(tables, config, args.output_dir)
    else:
        write_csv_tables(tables, args.output_dir)

    if args.ground_truth_dir:
        ground_truth = read_csv_tables(args.ground_truth_dir)
        comparison = compare(tables, ground_truth, args.output_dir)
        return comparison
    else:
        return None


def parse_args(arg_list: None | list[str]) -> argparse.Namespace:
    """Parses command line arguments.

    Args:
        arg_list: List of command line arguments. Uses sys.argv (default argparse behaviour) if `None`.

    Returns:
        argparse.Namespace: Parsed arguments.
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
        action="store_true",
        help="Verbose mode: print tables after every transform",
    )
    args = args_parser.parse_args(arg_list)
    return args


def main(arg_list: None | list[str] = None) -> None:
    """Main entry point for the xl2times package
    Returns:
        None.
    """
    args = parse_args(arg_list)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
