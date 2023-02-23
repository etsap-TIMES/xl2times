import os
from os import path
import shutil
import subprocess
import sys
from tabulate import tabulate
import time


def run_benchmark(benchmarks_folder, benchmark_name):
    xl_folder = path.join(benchmarks_folder, "xlsx", benchmark_name)
    dd_folder = path.join(benchmarks_folder, "dd", benchmark_name)
    csv_folder = path.join(benchmarks_folder, "csv", benchmark_name)
    out_folder = path.join(benchmarks_folder, "out", benchmark_name)

    # First convert ground truth DD to csv, if we haven't already
    if not path.exists(csv_folder):
        res = subprocess.run(
            [
                "python",
                "utils/dd_to_csv.py",
                dd_folder,
                csv_folder,
            ],
            capture_output=True,
            text=True,
        )
        if not res.returncode == 0:
            # Remove partial outputs so that next run retries
            shutil.rmtree(csv_folder, ignore_errors=True)
            return (0.0, "FAIL dd_to_csv: " + res.stderr.splitlines()[-1])

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
        return (runtime, f"FAIL: {res.stderr.splitlines()[-1]}")


if __name__ == "__main__":
    assert len(sys.argv) == 2
    benchmarks_folder = sys.argv[1]

    # Each benchmark is a directory in the benchmarks/xlsx/ folder:
    benchmarks = next(os.walk(path.join(benchmarks_folder, "xlsx")))[1]
    benchmarks = [b for b in sorted(benchmarks) if b[0] != "."]

    print("Running benchmarks", end="", flush=True)
    results = []
    headers = ["Demo", "Time (s)", "Result"]
    for benchmark_name in benchmarks:
        results.append(
            (benchmark_name, *run_benchmark(benchmarks_folder, benchmark_name))
        )
        print(".", end="", flush=True)
    print("\n\n" + tabulate(results, headers, floatfmt=".2f") + "\n")

    # Check if git status is clean
    res = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if res.returncode != 0 or len(res.stdout.strip()) != 0:
        print(res.stdout)
        print(res.stderr)
        print("ERROR: your working directory is not clean. Aborting.")
        sys.exit(1)

    # Checkout main branch
    res = subprocess.run(["git", "checkout", "main"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        print("ERROR: failed to checkout main branch. Aborting.")
        sys.exit(1)

    # Re-run benchmarks
    print("Running benchmarks on main", end="", flush=True)
    results_main = []
    for benchmark_name in benchmarks:
        results_main.append(
            (benchmark_name, *run_benchmark(benchmarks_folder, benchmark_name))
        )
        print(".", end="", flush=True)
    print("\n\n" + tabulate(results_main, headers, floatfmt=".2f") + "\n")

    # Checkout back to branch
    res = subprocess.run(["git", "checkout", "-"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        print("WARNING: failed to checkout branch. You're still on the main branch.")

    # Compare results
    accuracy = {b: float(r.split("%")[0]) for b, _, r in results}
    accuracy_main = {b: float(r.split("%")[0]) for b, _, r in results_main}
    if set(accuracy.keys()) != set(accuracy_main.keys()):
        print("ERROR: number of benchmarks changed")
        sys.exit(1)
    accu_regressions = [b for b in accuracy if accuracy[b] < accuracy_main[b]]

    times = {b: t for b, t, _ in results}
    times_main = {b: t for b, t, _ in results_main}
    time_regressions = [b for b in times if times[b] > 1.1 * times_main[b]]

    if len(accu_regressions + time_regressions) > 0:
        if accu_regressions:
            print(f"ERROR: accuracy regressed on: {', '.join(accu_regressions)}")
        if time_regressions:
            print(f"ERROR: runtime regressed on: {', '.join(time_regressions)}")
        sys.exit(1)
    # TODO also check if any table is missing more rows, and
    # check if any new tables are missing?

    print("All good. You're awesome!")
