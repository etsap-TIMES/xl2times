import os
import shutil
import subprocess
from tabulate import tabulate
import time

# check WD
# mkdir demos-csv


def run_demo_model(demo_num):
    xl_folder = f"demos-xlsx/DemoS_{demo_num:03d}"
    dd_folder = f"demos-dd/DD-DemoS_{demo_num:03d}/"
    csv_folder = f"demos-csv/DemoS_{demo_num:03d}"
    out_folder = f"demos-out/DemoS_{demo_num:03d}"

    # First convert ground truth DD to csv, if we haven't already
    if not os.path.exists(csv_folder):
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
            shutil.rmtree(csv_folder)
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
        ["python", "times_excel_reader.py"] + args, capture_output=True, text=True
    )
    runtime = time.time() - start

    if res.returncode == 0:
        lastline = res.stdout.splitlines()[-1].split(" ")
        accuracy = lastline[0] + " " + lastline[-1]
        return (runtime, accuracy)
    else:
        return (runtime, f"FAIL: {res.stderr.splitlines()[-1]}")


if __name__ == "__main__":
    results = []
    for i in range(1, 13):
        results.append((i, *run_demo_model(i)))
        print(".", end="", flush=True)
    print("\n" + tabulate(results, headers=["Demo", "Time", "Result"]) + "\n")
