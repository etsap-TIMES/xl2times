from pandas.core.frame import DataFrame
import pandas as pd
from dataclasses import replace
from typing import Dict, List
from itertools import groupby
import re
import os
from concurrent.futures import ProcessPoolExecutor
import time
from functools import reduce
import pickle
from . import datatypes
from . import excel
from . import transforms


def read_mappings(filename: str) -> List[datatypes.TimesXlMap]:
    """
    Function to load mappings from a text file between the excel sheets we use as input and
    the tables we give as output. The mappings have the following structure:

    OUTPUT_TABLE[DATAYEAR,VALUE] = ~TimePeriods(Year,B)

    where OUTPUT_TABLE is the name of the table we output and it includes a list of the
    different fields or column names it includes. On the other side, TimePeriods is the type
    of table that we will use as input to produce that table, and the arguments are the
    columns of that table to use to produce the output. The last argument can be of the
    form `Attribute:ATTRNAME` which means the output will be filtered to only the rows of
    the input table that have `ATTRNAME` in the Attribute column.

    The mappings are loaded into TimesXlMap objects. See the description of that class for more
    information of the different fields they contain.

    :param filename:        Name of the text file containing the mappings.
    :return:                List of mappings in TimesXlMap format.
    """
    mappings = []
    dropped = []
    with open(filename) as file:
        while True:
            line = file.readline().rstrip()
            if line == "":
                break
            (times, xl) = line.split(" = ")
            (times_name, times_cols_str) = list(filter(None, re.split("\[|\]", times)))
            (xl_name, xl_cols_str) = list(filter(None, re.split("\(|\)", xl)))
            times_cols = times_cols_str.split(",")
            xl_cols = xl_cols_str.split(",")
            filter_rows = False
            if xl_cols[-1].startswith("Attribute:"):
                xl_cols[-1] = xl_cols[-1].replace("Attribute:", "")
                filter_rows = True
            col_map = {}
            for index, value in enumerate(xl_cols):
                col_map[value] = times_cols[index]
            # Uppercase and validate tags:
            if xl_name.startswith("~"):
                xl_name = xl_name.upper()
                assert datatypes.Tag.has_tag(xl_name), f"Tag {xl_name} not found"
            entry = datatypes.TimesXlMap(
                times_name=times_name,
                times_cols=times_cols,
                xl_name=xl_name,
                xl_cols=xl_cols,
                col_map=col_map,
                filter_rows=filter_rows,
            )

            # TODO remove: Filter out mappings that are not yet finished
            if entry.xl_name != datatypes.Tag.todo and not any(
                c.startswith("TODO") for c in entry.xl_cols
            ):
                mappings.append(entry)
            else:
                dropped.append(line)

    if len(dropped) > 0:
        print(f"WARNING: Dropping {len(dropped)} mappings that are not yet complete")
    return mappings


def convert_xl_to_times(
    input_files: List[str],
    output_dir: str,
    mappings: List[datatypes.TimesXlMap],
    use_pkl: bool,
) -> Dict[str, DataFrame]:
    pickle_file = "raw_tables.pkl"
    if use_pkl and os.path.isfile(pickle_file):
        raw_tables = pickle.load(open(pickle_file, "rb"))
        print(f"WARNING: Using pickled data not xlsx")
    else:
        raw_tables = []

        use_pool = True
        if use_pool:
            with ProcessPoolExecutor() as executor:
                for result in executor.map(excel.extract_tables, input_files):
                    raw_tables.extend(result)
        else:
            for f in input_files:
                result = excel.extract_tables(f)
                raw_tables.extend(result)
        pickle.dump(raw_tables, open(pickle_file, "wb"))

    print(
        f"Extracted {len(raw_tables)} tables,"
        f" {sum(table.dataframe.shape[0] for table in raw_tables)} rows"
    )

    transform_list = [
        lambda tables: dump_tables(tables, os.path.join(output_dir, "raw_tables.txt")),
        transforms.normalize_tags_columns_attrs,
        transforms.remove_fill_tables,
        lambda tables: [transforms.remove_comment_rows(t) for t in tables],
        lambda tables: [transforms.remove_comment_cols(t) for t in tables],
        transforms.remove_tables_with_formulas,  # slow
        transforms.process_transform_insert,
        transforms.process_flexible_import_tables,  # slow
        transforms.process_user_constraint_tables,
        transforms.process_commodity_emissions,
        transforms.process_commodities,
        transforms.process_processes,
        transforms.process_transform_availability,
        transforms.fill_in_missing_values,
        transforms.process_time_slices,
        expand_rows_parallel,  # slow
        transforms.remove_invalid_values,
        transforms.process_time_periods,
        transforms.process_currencies,
        transforms.apply_fixups,
        transforms.extract_commodity_groups,
        transforms.merge_tables,
        transforms.process_years,
        transforms.process_wildcards,
        convert_to_string,
        lambda tables: dump_tables(
            tables, os.path.join(output_dir, "merged_tables.txt")
        ),
        lambda tables: produce_times_tables(tables, mappings),
    ]

    results = []
    input = raw_tables
    for transform in transform_list:
        start_time = time.time()
        output = transform(input)
        end_time = time.time()
        print(
            f"transform {transform.__code__.co_name} took {end_time-start_time:.2f} seconds"
        )
        results.append(output)
        input = output

    print(
        f"Conversion complete, {len(output)} tables produced,"
        f" {sum(df.shape[0] for tablename, df in output.items())} rows"
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
):
    print(
        f"Ground truth contains {len(ground_truth)} tables,"
        f" {sum(df.shape[0] for tablename, df in ground_truth.items())} rows"
    )

    missing = set(ground_truth.keys()) - set(data.keys())
    missing_str = ", ".join(
        [f"{x} ({ground_truth[x].shape[0]})" for x in sorted(missing)]
    )
    if len(missing) > 0:
        print(f"WARNING: Missing {len(missing)} tables: {missing_str}")

    total_gt_rows = 0
    total_correct_rows = 0
    for table_name, gt_table in sorted(
        ground_truth.items(), reverse=True, key=lambda t: len(t[1].values)
    ):
        total_gt_rows += len(gt_table.values)
        if table_name in data:
            data_table = data[table_name]

            # Remove .integer suffix added to duplicate column names by CSV reader (mangle_dupe_cols=False not supported)
            transformed_gt_cols = [col.split(".")[0] for col in gt_table.columns]

            if transformed_gt_cols != list(data[table_name].columns):
                print(
                    f"WARNING: Table {table_name} header incorrect, was"
                    f" {data_table.columns.values}, should be {transformed_gt_cols}"
                )
            else:
                # both are in string form so can be compared without any issues
                gt_rows = set(tuple(row) for row in gt_table.values.tolist())
                data_rows = set(tuple(row) for row in data_table.values.tolist())
                total_correct_rows += len(gt_rows.intersection(data_rows))
                additional = data_rows - gt_rows
                missing = gt_rows - data_rows
                if len(additional) != 0 or len(missing) != 0:
                    print(
                        f"WARNING: Table {table_name} ({data_table.shape[0]} rows,"
                        f" {gt_table.shape[0]} GT rows) contains {len(additional)}"
                        f" additional rows and is missing {len(missing)} rows"
                    )
                    DataFrame(additional).to_csv(
                        os.path.join(output_dir, table_name + "_additional.csv"),
                        index=False,
                    )
                    DataFrame(missing).to_csv(
                        os.path.join(output_dir, table_name + "_missing.csv"),
                        index=False,
                    )

    print(
        f"{total_correct_rows / total_gt_rows :.1%} of ground truth rows present"
        f" in output ({total_correct_rows}/{total_gt_rows})"
    )


def convert_to_string(input: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    output = {}
    for key, value in input.items():
        output[key] = value.applymap(
            lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
        )
    return output


def produce_times_tables(
    input: Dict[str, DataFrame], mappings: List[datatypes.TimesXlMap]
) -> Dict[str, DataFrame]:
    print(
        f"produce_times_tables: {len(input)} tables incoming,"
        f" {sum(len(value) for (_, value) in input.items())} rows"
    )
    result = {}
    used_tables = set()
    for mapping in mappings:
        if not mapping.xl_name in input:
            print(
                f"WARNING: Cannot produce table {mapping.times_name} because input table"
                f" {mapping.xl_name} does not exist"
            )
        else:
            used_tables.add(mapping.xl_name)
            df = input[mapping.xl_name].copy()
            if mapping.filter_rows:
                # Select just the rows where the attribute value matches the last mapping column or output table name
                if not "Attribute" in df.columns:
                    print(
                        f"WARNING: Cannot produce table {mapping.times_name} because input"
                        f" table {mapping.xl_name} does not contain an Attribute column"
                    )
                else:
                    colname = mapping.xl_cols[-1]
                    filter = set(x.lower() for x in {colname, mapping.times_name})
                    i = df["Attribute"].str.lower().isin(filter)
                    df = df.loc[i, :]
                    if colname not in df.columns:
                        df = df.rename(columns={"VALUE": colname})
            # TODO find the correct tech group
            if "TechGroup" in mapping.xl_cols:
                df["TechGroup"] = df["TechName"]
            if not all(c in df.columns for c in mapping.xl_cols):
                missing = set(mapping.xl_cols) - set(df.columns)
                print(
                    f"WARNING: Cannot produce table {mapping.times_name} because input"
                    f" table {mapping.xl_name} does not contain the required columns"
                    f" - {', '.join(missing)}"
                )
            else:
                cols_to_drop = [x for x in df.columns if not x in mapping.xl_cols]
                df.drop(columns=cols_to_drop, inplace=True)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.rename(columns=mapping.col_map, inplace=True)
                i = df[mapping.times_cols[-1]].notna()
                df = df.loc[i, mapping.times_cols]
                result[mapping.times_name] = df

    unused_tables = set(input.keys()) - used_tables
    if len(unused_tables) > 0:
        print(
            f"WARNING: {len(unused_tables)} unused tables: {', '.join(sorted(unused_tables))}"
        )

    return result


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


def expand_rows_parallel(
    tables: List[datatypes.EmbeddedXlTable],
) -> List[datatypes.EmbeddedXlTable]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(transforms.expand_rows, tables))


def write_csv_tables(tables: Dict[str, DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".csv"):
            os.remove(os.path.join(output_dir, item))
    for tablename, df in tables.items():
        df.to_csv(os.path.join(output_dir, tablename + "_output.csv"), index=False)
