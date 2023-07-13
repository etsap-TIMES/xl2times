from pathlib import Path
import argparse
import os.path
import sys
import times_reader

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "input",
        nargs="*",
        help="Either an input directory, or a list of input xlsx files to process",
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
    args_parser.add_argument("--use_pkl", action="store_true")
    args = args_parser.parse_args()

    mappings = times_reader.read_mappings("times_mapping.txt")

    if not isinstance(args.input, list) or len(args.input) < 1:
        print(f"ERROR: expected at least 1 input. Got {args.input}")
        sys.exit(1)
    elif len(args.input) == 1:
        assert os.path.isdir(args.input[0])
        input_files = [
            str(path)
            for path in Path(args.input[0]).rglob("*.xlsx")
            if not path.name.startswith("~")
        ]
        print(f"Loading {len(input_files)} files from {args.input[0]}")
    else:
        input_files = args.input

    tables = times_reader.convert_xl_to_times(
        input_files, args.output_dir, mappings, args.use_pkl
    )

    if args.dd:
        times_reader.write_dd_files(tables, mappings, args.output_dir)
    else:
        times_reader.write_csv_tables(tables, args.output_dir)

    if args.ground_truth_dir:
        ground_truth = times_reader.read_csv_tables(args.ground_truth_dir)
        times_reader.compare(tables, ground_truth, args.output_dir)
