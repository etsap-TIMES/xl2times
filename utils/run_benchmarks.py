import argparse
import git
from os import path, symlink
import pandas as pd
from re import match
import shutil
import subprocess
import sys
from tabulate import tabulate
import time
from typing import Any, Tuple
import yaml


def parse_result(lastline):
    m = match(
        r"(\d+\.\d)\% of ground truth rows present in output \((\d+)/(\d+)\)"
        r", (\d+) additional rows",
        lastline,
    )
    if not m:
        print(f"ERROR: could not parse output of run:\n{lastline}")
        sys.exit(1)
    # return (accuracy, num_correct_rows, num_additional_rows)
    return (float(m.groups()[0]), int(m.groups()[1]), int(m.groups()[3]))


def run_benchmark_dd(
    benchmarks_folder: str,
    benchmark: Any,
    times_folder: str,
    out_folder: str = "out",
    verbose: bool = False,
) -> Tuple[float, float, int, int]:
    xl_folder = path.join(benchmarks_folder, "xlsx", benchmark["input_folder"])
    dd_folder = path.join(benchmarks_folder, "dd", benchmark["dd_folder"])
    csv_folder = path.join(benchmarks_folder, "csv", benchmark["name"])
    out_folder = path.join(benchmarks_folder, out_folder, benchmark["name"])

    # First run the tool and generate DD output
    args = [
        "--output_dir",
        out_folder,
        "--dd",
    ]
    if "inputs" in benchmark:
        args.extend((path.join(xl_folder, b) for b in benchmark["inputs"]))
    else:
        args.append(xl_folder)
    start = time.time()
    res = subprocess.run(
        ["python", "times_excel_reader.py"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    runtime = time.time() - start

    with open(path.join(out_folder, "stdout"), "w") as f:
        f.write(res.stdout)
    if verbose:
        line = "-" * 80
        print(f"\n{line}\n{benchmark['name']}\n{line}\n\n{res.stdout}")
        print(res.stderr if res.stderr is not None else "")

    if res.returncode != 0:
        print(res.stdout)
        print(f"ERROR: tool failed on {benchmark['name']}")
        sys.exit(1)

    # Copy GAMS scaffolding
    scaffolding_folder = path.join(
        path.dirname(path.realpath(__file__)), "..", "gams_scaffold"
    )
    shutil.copytree(scaffolding_folder, out_folder, dirs_exist_ok=True)
    # Create link to TIMES source TODO get path as arg
    if not path.exists(path.join(out_folder, "source")):
        symlink(times_folder, path.join(out_folder, "source"), True)

    # Run GAMS
    res = subprocess.run(
        ["gams", "runmodel"],
        cwd=out_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr if res.stderr is not None else "")
        print(f"ERROR: GAMS failed on {benchmark['name']}")
        sys.exit(1)
    if "error" in res.stdout.lower():
        print(res.stdout)
        print(f"ERROR: GAMS errored on {benchmark['name']}")
        return (runtime, "Error running GAMS")

    # Run GAMS on ground truth:
    shutil.copytree(scaffolding_folder, dd_folder, dirs_exist_ok=True)
    # Modify batinclude files according to benchmarks.yml's `dd_files`
    scenario_run = open(path.join(dd_folder, "scenario.run")).readlines()
    with open(path.join(dd_folder, "scenario.run"), "w") as f:
        for line in scenario_run:
            if line.strip() == "$BATINCLUDE output.dd":
                for file in benchmark["dd_files"]:
                    f.write(f"$BATINCLUDE {file}.dd\n")
                continue
            f.write(line)
    # TODO also get milestone years from benchmarks.yml
    # Create link to TIMES source TODO get path as arg
    if not path.exists(path.join(dd_folder, "source")):
        symlink(times_folder, path.join(dd_folder, "source"), True)
    res = subprocess.run(
        ["gams", "runmodel"],
        cwd=dd_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr if res.stderr is not None else "")
        print(f"ERROR: GAMS failed on {benchmark['name']} ground truth")
        sys.exit(1)
    if "error" in res.stdout.lower():
        print(res.stdout)
        print(f"ERROR: GAMS errored on {benchmark['name']}")
        return (runtime, "Error running GAMS on ground truth")

    # Run gdxdiff to compare
    res = subprocess.run(
        [
            "gdxdiff",
            path.join(dd_folder, "scenario.gdx"),
            path.join(out_folder, "scenario.gdx"),
            path.join(out_folder, "diffile.gdx"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr if res.stderr is not None else "")
        print(f"ERROR: gdxdiff failed on {benchmark['name']}")
        return (runtime, "Different")

    return (runtime, "OK")


def run_benchmark(
    benchmarks_folder: str,
    benchmark: Any,
    skip_csv: bool = False,
    out_folder: str = "out",
    verbose: bool = False,
) -> Tuple[float, float, int, int]:
    xl_folder = path.join(benchmarks_folder, "xlsx", benchmark["input_folder"])
    dd_folder = path.join(benchmarks_folder, "dd", benchmark["dd_folder"])
    csv_folder = path.join(benchmarks_folder, "csv", benchmark["name"])
    out_folder = path.join(benchmarks_folder, out_folder, benchmark["name"])

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
            print(f"ERROR: dd_to_csv failed on {benchmark['name']}")
            sys.exit(1)
    elif not path.exists(csv_folder):
        print(f"ERROR: --skip_csv is true but {csv_folder} does not exist")
        sys.exit(1)

    # Then run the tool
    args = [
        "--output_dir",
        out_folder,
        "--ground_truth_dir",
        csv_folder,
    ]
    if "inputs" in benchmark:
        args.extend((path.join(xl_folder, b) for b in benchmark["inputs"]))
    else:
        args.append(xl_folder)
    start = time.time()
    res = subprocess.run(
        ["python", "times_excel_reader.py"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    runtime = time.time() - start

    with open(path.join(out_folder, "stdout"), "w") as f:
        f.write(res.stdout)
    if verbose:
        line = "-" * 80
        print(f"\n{line}\n{benchmark['name']}\n{line}\n\n{res.stdout}")
        print(res.stderr if res.stderr is not None else "")

    if res.returncode == 0:
        (accuracy, num_correct, num_additional) = parse_result(
            res.stdout.splitlines()[-1]
        )
        return (runtime, accuracy, num_correct, num_additional)
    else:
        print(res.stdout)
        print(f"ERROR: tool failed on {benchmark['name']}")
        sys.exit(1)


def run_all_benchmarks(
    benchmarks_folder: str,
    benchmarks: list,
    skip_csv=False,
    skip_main=False,
    verbose=False,
):
    print("Running benchmarks", end="", flush=True)
    results = []
    headers = ["Benchmark", "Time (s)", "Accuracy", "Correct Rows", "Additional Rows"]
    for benchmark in benchmarks:
        result = run_benchmark(
            benchmarks_folder, benchmark, skip_csv=skip_csv, verbose=verbose
        )
        results.append((benchmark["name"], *result))
        print(".", end="", flush=True)
    print("\n\n" + tabulate(results, headers, floatfmt=".1f") + "\n")

    # The rest of this script checks regressions against main
    # so skip it if we're already on main
    repo = git.Repo(".")  # pyright: ignore
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

    if skip_main:
        results_main = []
        for benchmark in benchmarks:
            with open(
                path.join(benchmarks_folder, "out-main", benchmark["name"], "stdout"),
                "r",
            ) as f:
                result = parse_result(f.readlines()[-1])
            # Use a fake runtime
            results_main.append((benchmark["name"], 999, *result))
        print(
            f"Skipped running on main. Using results from {path.join(benchmarks_folder, 'out-main')}"
        )

    else:
        if repo.is_dirty():
            print("ERROR: your working directory is not clean. Aborting.")
            sys.exit(1)

        # Re-run benchmarks on main
        repo.heads.main.checkout()
        print("Running benchmarks on main", end="", flush=True)
        results_main = []
        for benchmark in benchmarks:
            result = run_benchmark(
                benchmarks_folder,
                benchmark,
                skip_csv=True,
                out_folder="out-main",
                verbose=verbose,
            )
            results_main.append((benchmark["name"], *result))
            print(".", end="", flush=True)

    # Print table with combined results to make comparison easier
    combined_results = [
        (
            f"{b:<20}",
            f"{t0:5.1f} {t:5.1f}",
            f"{a0:5.1f} {a:5.1f}",
            f"{c0:6d} {c:6d}",
            f"{d0:6d} {d:6d}",
        )
        for ((b, t, a, c, d), (_, t0, a0, c0, d0)) in zip(results, results_main)
    ]
    print("\n\n" + tabulate(combined_results, headers, stralign="right") + "\n")

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
        print(f"ERROR: number of benchmarks changed:\n{df}")
        sys.exit(1)
    accu_regressions = df[df["Correct Rows"] < df["M Correct Rows"]]["Benchmark"]
    addi_regressions = df[df["Additional Rows"] > df["M Additional Rows"]]["Benchmark"]
    time_regressions = df[df["Time (s)"] > 2 * df["M Time (s)"]]["Benchmark"]

    runtime_change = df["Time (s)"].sum() - df["M Time (s)"].sum()
    print(f"Change in runtime: {runtime_change:+.2f}")
    correct_change = df["Correct Rows"].sum() - df["M Correct Rows"].sum()
    print(f"Change in correct rows: {correct_change:+d}")
    additional_change = df["Additional Rows"].sum() - df["M Additional Rows"].sum()
    print(f"Change in additional rows: {additional_change:+d}")

    if len(accu_regressions) + len(addi_regressions) + len(time_regressions) > 0:
        print()
        if not accu_regressions.empty:
            print(f"ERROR: correct rows regressed on: {', '.join(accu_regressions)}")
        if not addi_regressions.empty:
            print(f"ERROR: additional rows regressed on: {', '.join(addi_regressions)}")
        if not time_regressions.empty:
            print(f"ERROR: runtime regressed on: {', '.join(time_regressions)}")
        sys.exit(1)
    # TODO also check if any new tables are missing?

    print("No regressions. You're awesome!")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "benchmarks_yaml",
        type=str,
        help="Benchmarks specification file",
    )
    args_parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run a single benchmark instead of all benchmarks",
    )
    args_parser.add_argument(
        "--dd",
        action="store_true",
        default=False,
        help="Generate DD files, and use GAMS to compare with ground truth",
    )
    args_parser.add_argument(
        "--times_dir",
        type=str,
        default=None,
        help="Absolute path to TIMES_model. Required if using --dd",
    )
    args_parser.add_argument(
        "--skip_csv",
        action="store_true",
        default=False,
        help="Skip generating csv versions of ground truth DD files",
    )
    args_parser.add_argument(
        "--skip_main",
        action="store_true",
        default=False,
        help="Skip running tool on main and reuse existing result files",
    )
    args_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print output of run on each benchmark",
    )
    args = args_parser.parse_args()

    spec = yaml.safe_load(open(args.benchmarks_yaml))
    benchmarks_folder = spec["benchmarks_folder"]
    benchmark_names = [b["name"] for b in spec["benchmarks"]]
    if len(set(benchmark_names)) != len(benchmark_names):
        print("ERROR: Found duplicate name in benchmarks YAML file")
        sys.exit(1)

    if args.dd and args.times_dir is None:
        print("ERROR: --times_model is required when using --dd")
        sys.exit(1)

    if args.run is not None:
        benchmark = next((b for b in spec["benchmarks"] if b["name"] == args.run), None)
        if benchmark is None:
            print(f"ERROR: could not find {args.run} in {args.benchmarks_yaml}")
            sys.exit(1)

        if args.dd:
            runtime, _, _, _ = run_benchmark_dd(
                benchmarks_folder,
                benchmark,
                args.times_dir,
                verbose=args.verbose,
            )
        else:
            runtime, _, _, _ = run_benchmark(
                benchmarks_folder,
                benchmark,
                skip_csv=args.skip_csv,
                verbose=args.verbose,
            )
        print(f"Ran {args.run} in {runtime:.2f}s")
    else:
        if args.dd:
            print("Running benchmarks", end="", flush=True)
            results = []
            headers = ["Benchmark", "Time (s)", "Result"]
            for benchmark in [b for b in spec["benchmarks"] if "dd_files" in b]:
                result = run_benchmark_dd(
                    benchmarks_folder,
                    benchmark,
                    args.times_dir,
                    verbose=args.verbose,
                )
                results.append((benchmark["name"], *result))
                print(".", end="", flush=True)
            print("\n\n" + tabulate(results, headers, floatfmt=".1f") + "\n")
        else:
            run_all_benchmarks(
                benchmarks_folder,
                spec["benchmarks"],
                args.skip_csv,
                args.skip_main,
                args.verbose,
            )
