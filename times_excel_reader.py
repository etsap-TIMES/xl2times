from pathlib import Path
import argparse
import times_reader

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "input_dir", type=str, help="Input directory containing xlsx files"
    )
    args_parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    args_parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default="ground_truth",
        help="Ground truth directory to compare with output",
    )
    args_parser.add_argument("--use_pkl", action="store_true")
    args = args_parser.parse_args()

    mappings = times_reader.read_mappings("times_mapping.txt")

    input_files = [
        str(path)
        for path in Path(args.input_dir).rglob("*.xlsx")
        if not path.name.startswith("~")
    ]
    print(f"Loading {len(input_files)} files from {args.input_dir}")

    tables = times_reader.convert_xl_to_times(
        input_files, args.output_dir, mappings, args.use_pkl
    )

    times_reader.write_csv_tables(tables, args.output_dir)

    if args.ground_truth_dir:
        ground_truth = times_reader.read_csv_tables(args.ground_truth_dir)
        times_reader.compare(tables, ground_truth, args.output_dir)
