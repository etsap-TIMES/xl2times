import argparse
import git
import os
from os import path
import pandas as pd
from re import match
import shutil
import subprocess
import sys
from tabulate import tabulate
import time
from typing import Tuple

import pickle


def run_benchmark(
    benchmarks_folder: str, benchmark_name: str, skip_csv: bool = False
) -> Tuple[float, float, int, int]:
    xl_folder = path.join(benchmarks_folder, "xlsx", benchmark_name)
    dd_folder = path.join(benchmarks_folder, "dd", benchmark_name)
    csv_folder = path.join(benchmarks_folder, "csv", benchmark_name)
    out_folder = path.join(benchmarks_folder, "out", benchmark_name)

    # First convert ground truth DD to csv
    if not skip_csv:
        shutil.rmtree(csv_folder, ignore_errors=True)
        res = subprocess.run(
            [
                "python",
                "utils/dd_to_csv.py",
                dd_folder,
                csv_folder,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if res.returncode != 0:
            # Remove partial outputs
            shutil.rmtree(csv_folder, ignore_errors=True)
            print(res.stdout)
            print(f"ERROR: dd_to_csv failed on {benchmark_name}")
            sys.exit(1)
    elif not path.exists(csv_folder):
        print(f"ERROR: --skip_csv is true but {csv_folder} does not exist")
        sys.exit(1)

    # Then run the tool
    args = [
        xl_folder,
        "--output_dir",
        out_folder,
        "--ground_truth_dir",
        csv_folder,
    ]
    start = time.time()
    res = subprocess.run(
        ["python", "times_excel_reader.py"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    runtime = time.time() - start
    with open(path.join(benchmarks_folder, "out", f"{benchmark_name}.out"), "w") as f:
        f.write(res.stdout)

    if res.returncode == 0:
        lastline = res.stdout.splitlines()[-1]
        m = match(
            r"(\d+\.\d)\% of ground truth rows present in output \((\d+)/(\d+)\)"
            r", (\d+) additional rows",
            lastline,
        )
        if not m:
            print(f"ERROR: could not parse output of run:\n{lastline}")
            sys.exit(1)
        # return (runtime, accuracy, num_correct_rows, num_additional_rows)
        return (runtime, float(m.groups()[0]), int(m.groups()[1]), int(m.groups()[3]))
    else:
        print(res.stdout)
        print(f"ERROR: tool failed on {benchmark_name}")
        sys.exit(1)


def run_all_benchmarks(benchmarks_folder, skip_csv=False):
    # Each benchmark is a directory in the benchmarks/xlsx/ folder:
    benchmarks = next(os.walk(path.join(benchmarks_folder, "xlsx")))[1]
    benchmarks = [b for b in sorted(benchmarks) if b[0] != "."]

    print("Running benchmarks", end="", flush=True)
    results = []
    headers = ["Benchmark", "Time (s)", "Accuracy", "Correct Rows", "Additional Rows"]
    for benchmark_name in benchmarks:
        result = run_benchmark(benchmarks_folder, benchmark_name, skip_csv)
        results.append((benchmark_name, *result))
        print(".", end="", flush=True)
    print("\n\n" + tabulate(results, headers, floatfmt=".1f") + "\n")
    pickle.dump(results, open("temp.pkl", "wb"))

    # The rest of this script checks regressions against main
    # so skip it if we're already on main
    repo = git.Repo(".")
    origin = repo.remotes.origin
    origin.fetch("main")
    if "main" not in repo.heads:
        repo.create_head("main", origin.refs.main).set_tracking_branch(origin.refs.main)
    try:
        mybranch = repo.active_branch
    except TypeError:  # If we're not on a branch (like on CI), create one:
        mybranch = repo.create_head("mybranch")

    if mybranch.name == "main":
        print("Skipping regression tests as we're on main branch. Goodbye!")
        sys.exit(0)

    if repo.is_dirty():
        print("ERROR: your working directory is not clean. Aborting.")
        sys.exit(1)

    # Re-run benchmarks on main
    repo.heads.main.checkout()
    print("Running benchmarks on main", end="", flush=True)
    results_main = []
    for benchmark_name in benchmarks:
        result = run_benchmark(benchmarks_folder, benchmark_name, skip_csv=True)
        results_main.append((benchmark_name, *result))
        print(".", end="", flush=True)
    print("\n\n" + tabulate(results_main, headers, floatfmt=".1f") + "\n")

    # Checkout back to branch
    mybranch.checkout()

    # Compare results
    main_headers = ["Benchmark"] + ["M " + h for h in headers[1:]]
    df = pd.merge(
        pd.DataFrame(results_main, columns=main_headers),
        pd.DataFrame(results, columns=headers),
        on="Benchmark",
        how="outer",
    )
    if df.isna().values.any():
        print("ERROR: number of benchmarks changed")
        sys.exit(1)
    accu_regressions = df[df["Correct Rows"] < df["M Correct Rows"]]["Benchmark"]
    addi_regressions = df[df["Additional Rows"] > df["M Additional Rows"]]["Benchmark"]
    time_regressions = df[df["Time (s)"] > 2 * df["M Time (s)"]]["Benchmark"]

    if len(accu_regressions) + len(addi_regressions) + len(time_regressions) > 0:
        if not accu_regressions.empty:
            print(f"ERROR: correct rows regressed on: {', '.join(accu_regressions)}")
        if not addi_regressions.empty:
            print(f"ERROR: additional rows regressed on: {', '.join(accu_regressions)}")
        if not time_regressions.empty:
            print(f"ERROR: runtime regressed on: {', '.join(time_regressions)}")
        sys.exit(1)
    # TODO also check if any new tables are missing?

    print("No regressions. You're awesome!")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "benchmarks_folder",
        type=str,
        help="Benchmarks directory. Assumes subdirectories `xlsx` and `dd` containing one"
        " folder per benchmark of inputs and outputs (respectively)",
    )
    args_parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run a single benchmark instead of all benchmarks",
    )
    args_parser.add_argument(
        "--skip_csv",
        action="store_true",
        default=False,
        help="Skip generating csv versions of ground truth DD files",
    )
    args = args_parser.parse_args()

    if args.run is not None:
        runtime, _, _, _ = run_benchmark(
            args.benchmarks_folder, args.run, args.skip_csv
        )
        print(f"Ran {args.run} in {runtime:.2f}s")
    else:
        run_all_benchmarks(args.benchmarks_folder, args.skip_csv)
