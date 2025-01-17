import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def parse_parameter_values_from_file(
    path: Path,
) -> tuple[dict[str, list], dict[str, set]]:
    """Parse a `*.dd` file and extract the sets and parameters.

    There are parameters and sets, and each has a slightly different format.
    `*.dd` files have data of the following form:

    PARAMETER
    PARAM_NAME ' '/
    attr_1 attr_2
    ...

    /

    SET SET_NAME
    /
    attr_1.attr_2.attr_3
    ...

    /
    """
    data = list(open(path))
    data = [line.rstrip() for line in data]

    param_value_dict: dict[str, list] = dict()
    set_data_dict: dict[str, set] = dict()
    index = 0
    while index < len(data):
        if data[index].startswith("PARAMETER"):
            # We expect the parameter name on the next line.
            index += 1
            while data[index].strip() == "":
                index += 1

            param_name = data[index].replace(
                " ' '/", ""
            )  # param_name is followed by this pattern

            index += 1
            param_data = []
            # Parse values until a line with / is encountered.
            while not data[index].startswith("/") and data[index] != "":
                line = data[index]

                # Either "value" for a scalar, or "key value" for an array.
                # So value is always the last word, or only token
                split_point = line.rfind(" ")
                if split_point == -1:
                    # if only one word
                    attributes, value = [], line
                else:
                    attributes, value = line[:split_point], line[split_point + 1 :]
                    attributes = attributes.split(".")
                    attributes = [a if " " in a else a.strip("'") for a in attributes]

                param_data.append([*attributes, value])

                index += 1

            param_value_dict[param_name] = param_data

        if data[index].startswith("SET"):
            # See https://www.gams.com/latest/docs/UG_SetDefinition.html
            # This can only parse a subset of the allowed representations
            _, name = data[index].split(" ")
            index += 1
            while data[index].strip() == "":
                index += 1

            assert data[index].startswith("/")
            index += 1

            set_data = set()
            while not data[index].startswith("/") and data[index] != "":
                parts = [[]]
                for word in data[index].split("'"):
                    if word != "":
                        if word.isspace():
                            parts.append([])
                        elif word.endswith(" "):
                            parts[-1].append(word.strip())
                            parts.append([])
                        else:
                            parts[-1].append(word)
                words = ["".join(part) for part in parts]
                attributes = words[0].split(".")
                if len(words) == 1:
                    set_data.add(tuple([*attributes]))
                elif len(words) == 2:
                    text = words[1]
                    set_data.add(tuple([*attributes, text]))
                else:
                    raise ValueError(
                        f"Unexpected number of spaces in set value setting: {data[index]}"
                    )

                index += 1

            if name in set_data_dict:
                set_data_dict[name].update(set_data)
            else:
                set_data_dict[name] = set_data

        index += 1

    return param_value_dict, set_data_dict


def save_data_with_headers(
    param_data_dict: dict[str, pd.DataFrame | list[str]],
    headers_data: dict[str, list[str]],
    save_dir: str,
) -> None:
    """Saves data (with headers) to the provided directory.

    Note that the header and data dictionaries are assumed to be parallel dictionaries

    Parameters
    ----------
    param_data_dict
        Dictionary containing key=param_name and val=dataframe for parameters or List[str] for sets
    headers_data
        Dictionary containing key=param_name and val=dataframes
    save_dir
        Path to folder in which to save the tabular data files
    """
    for param_name, param_data in param_data_dict.items():
        try:
            columns = headers_data[param_name]
        except KeyError:
            raise ValueError(
                f"Could not find mapping for {param_name} in mapping file."
            )
        for row in param_data:
            if len(row) != len(columns):
                raise ValueError(
                    f"Mismatched number of columns for param {param_name} between data ({len(row)}) and mapping ({len(columns)})"
                )
        df = pd.DataFrame(
            data=np.asarray(param_data)[:, 0 : len(columns)], columns=columns
        )
        df.to_csv(os.path.join(save_dir, param_name + ".csv"), index=False)

    return


def generate_headers_by_attr() -> dict[str, list[str]]:
    with open("xl2times/config/times-info.json") as f:
        attributes = json.load(f)

    headers_by_attr = {}

    for attr in attributes:
        if attr["gams-cat"] == "parameter":
            headers_by_attr[attr["name"]] = [*attr["indexes"], "VALUE"]
        else:
            headers_by_attr[attr["name"]] = attr["indexes"]

    return headers_by_attr


def convert_dd_to_tabular(
    dd_files: list[Path], output_dir: str, headers_by_attr: dict[str, list[str]]
) -> None:

    all_sets = defaultdict(list)
    all_parameters = defaultdict(list)
    for path in dd_files:
        logger.info(f"Processing path: {path}")
        local_param_values, local_sets = parse_parameter_values_from_file(path)

        # merge params from file into global collection
        for param, data in local_param_values.items():
            all_parameters[param].extend(data)

        for set_name, data in local_sets.items():
            all_sets[set_name].extend(data)

    use_subfolders = False
    if use_subfolders:
        param_path = os.path.join(output_dir, "params")
        set_path = os.path.join(output_dir, "sets")
    else:
        param_path = output_dir
        set_path = output_dir
    os.makedirs(param_path, exist_ok=True)
    os.makedirs(set_path, exist_ok=True)

    # Extract headers with key=param_name and value=List[attributes]
    lines = list(open("xl2times/config/times_mapping.txt"))
    headers_data = headers_by_attr
    # The following will overwrite data obtained from headers_by_attr
    # TODO: Remove once migration is done?
    for line in lines:
        ln = line.strip()
        if ln != "":
            param_name = ln.split("[")[0]
            attributes = ln.split("[")[1].split("]")[0].split(",")
            headers_data[param_name] = [*attributes]

    save_data_with_headers(all_parameters, headers_data, param_path)
    save_data_with_headers(all_sets, headers_data, set_path)

    return


def main(arg_list: None | list[str] = None):
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "input_dir", type=str, help="Input directory containing .dd files."
    )
    args_parser.add_argument(
        "output_dir", type=str, help="Output directory to save the .csv files in."
    )
    args_parser.add_argument(
        "--include_files",
        type=str,
        nargs="*",
        help="List of .dd file stems to process.",
    )
    args = args_parser.parse_args(arg_list)
    dd_files = [p for p in Path(args.input_dir).glob("*.dd")]
    if args.include_files:
        valid_stems = {stem.lower() for stem in args.include_files}
        dd_files = [p for p in dd_files if p.stem.lower() in valid_stems]
    convert_dd_to_tabular(dd_files, args.output_dir, generate_headers_by_attr())


if __name__ == "__main__":
    main(sys.argv[1:])
