import argparse
import git
import os
from os import path
import shutil
import subprocess
import sys
from tabulate import tabulate
import time


def run_benchmark(benchmarks_folder, benchmark_name, skip_csv=False):
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
        lastline = res.stdout.splitlines()[-1].split(" ")
        accuracy = lastline[0] + " " + lastline[-1]
        return (runtime, accuracy)
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
    headers = ["Demo", "Time (s)", "Result"]
    for benchmark_name in benchmarks:
        runtime, accuracy = run_benchmark(benchmarks_folder, benchmark_name, skip_csv)
        results.append((benchmark_name, runtime, accuracy))
        print(".", end="", flush=True)
    print("\n\n" + tabulate(results, headers, floatfmt=".2f") + "\n")

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
        results_main.append(
            (benchmark_name, *run_benchmark(benchmarks_folder, benchmark_name))
        )
        print(".", end="", flush=True)
    print("\n\n" + tabulate(results_main, headers, floatfmt=".2f") + "\n")

    # Checkout back to branch
    mybranch.checkout()

    # Compare results
    accuracy = {b: float(r.split("%")[0]) for b, _, r in results}
    accuracy_main = {b: float(r.split("%")[0]) for b, _, r in results_main}
    if set(accuracy.keys()) != set(accuracy_main.keys()):
        print("ERROR: number of benchmarks changed")
        sys.exit(1)
    accu_regressions = [b for b in accuracy if accuracy[b] < accuracy_main[b]]

    times = {b: t for b, t, _ in results}
    times_main = {b: t for b, t, _ in results_main}
    time_regressions = [b for b in times if times[b] > 2 * times_main[b]]

    if len(accu_regressions + time_regressions) > 0:
        if accu_regressions:
            print(f"ERROR: accuracy regressed on: {', '.join(accu_regressions)}")
        if time_regressions:
            print(f"ERROR: runtime regressed on: {', '.join(time_regressions)}")
        sys.exit(1)
    # TODO also check if any table is missing more rows, and
    # check if any new tables are missing?

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
        runtime, _ = run_benchmark(args.benchmarks_folder, args.run, args.skip_csv)
        print(f"Ran {args.run} in {runtime:.2f}s")
    else:
        run_all_benchmarks(args.benchmarks_folder, args.skip_csv)
