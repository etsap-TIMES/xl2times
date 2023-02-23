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

    print("Running benchmarks")
    results = []
    for benchmark_name in benchmarks:
        results.append(
            (benchmark_name, *run_benchmark(benchmarks_folder, benchmark_name))
        )
        print(".", end="", flush=True)
    print("\n" + tabulate(results, headers=["Demo", "Time", "Result"]) + "\n")
    # TODO exit(1) if any benchmark run failed
