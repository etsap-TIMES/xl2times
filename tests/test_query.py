"""
Some utils and tools to test different approaches to transforms.query and benchmark performance.

Right now this file doesn't test anything, but if we go with a new approach to queries, we can use the code here to test that the new algorithm produces the same results as the old one.

Notes
-----
Check on benchmarks how large table and query_df are
    table: (350143, 20), updates: (217, 16)
    table: (350043, 20), updates: (12276, 16)
    N * Q = 350043 * 12276 = 4,297,127,868  -- might take too long!

Okay, there are queries from TFM_UPD etc that use the disjunctive lists.. but these can be exploded in the query DF?
    max N = 362319, Q = 214613
    N * Q = 362319 * 214613 = 77,758,367,547 -- way too much! 77GB, needs batching. Not sure how long it will take

Test the boolean mask method on random DFs of similar size?
    20% slower than iterating query? :(

Pickle the actual table and queries used by GEO, and benchmark on that.
    It's too slow because exploding 217 queries leads to 6e4 queries!

Bool mask to do: (abandoned bool mask because it's too slow)
    Explode queries containing lists of possible values
    Modify the boolean mask method to keep track of which query resulted in which row
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

# Create a global random number generator
rng = np.random.default_rng(42)


def query(
    table: DataFrame,
    process: str | list[str] | None,
    commodity: str | list[str] | None,
    attribute: str | None,
    region: str | list[str] | None,
    year: int | list | None,
    limtype: str | list[str] | None,
    val: int | float | None,
    module: str | list[str] | None,
) -> pd.Index:
    query_fields = {
        "process": process,
        "commodity": commodity,
        "attribute": attribute,
        "region": region,
        "year": year,
        "limtype": limtype,
        "value": val,
        "module_name": module,
    }

    def is_missing(field):
        return pd.isna(field) if not isinstance(field, list) else False

    qs = [
        f"{k} in {v if isinstance(v, list) else [v]}"
        for k, v in query_fields.items()
        if not is_missing(v)
    ]

    if not qs:  # If no conditions, return all rows
        return table.index
    query_str = " and ".join(qs)
    row_idx = table.query(query_str).index
    return row_idx


def query_boolmask(table: pd.DataFrame, filters: pd.DataFrame) -> pd.Index:
    """
    Return the index of `table` rows that match *any* of the filter rows in `filters`,
    where NaN in filters means “match anything here”.

    - `table`: big DataFrame which has all columns of `filters.columns`
    - `filters`: small DataFrame, each row a “partial” spec. NaN = wildcard.
    """
    # Convert columns to categorical for faster comparisons
    categorical_cols = [col for col in filters.columns if col != "value"]
    for col in categorical_cols:
        table[col] = pd.Categorical(table[col])
        filters[col] = pd.Categorical(
            filters[col], categories=table[col].cat.categories
        )

    # number of table rows N, number of filters Q
    N, Q = len(table), len(filters)  # noqa: N806
    print(f"N: {N}, Q: {Q}")
    # start with all True mask of shape (N, Q)
    mask = np.ones((N, Q), dtype=bool)

    for col in filters.columns:
        desired = filters[col].to_numpy()
        actual = table[col].to_numpy()

        # Get mask of which filter values are wildcards (NA)
        wild = pd.isna(desired)

        if not wild.all():  # If all filters are NA, this filter matches every row
            # Create equality mask only for non-NA filter values
            non_wild_mask = ~wild
            eq = np.ones((N, Q), dtype=bool)
            non_wild_desired = desired[non_wild_mask]
            eq[:, non_wild_mask] = actual[:, None] == non_wild_desired[None, :]

            # accumulate
            mask &= eq

        # early exit if nothing can match any filter any more
        if not mask.any():
            return table.index[mask.any(axis=1)]

    # rows matching at least one filter
    hits = mask.any(axis=1)
    return table.index[hits]


def process_chunk(queries, table):
    """Process a chunk of queries using the given table."""
    results = set()
    for q in queries:
        results.update(query(table, **q))
    return results


if __name__ == "__main__":
    import pickle
    import sys
    import time
    from itertools import product
    from pathlib import Path

    import numpy as np

    # Helper function to generate random data
    def generate_random_data(n_rows):
        processes = [f"PROC_{i}" for i in range(10)]
        commodities = [f"COM_{i}" for i in range(10)]
        attributes = ["ATT_1", "ATT_2", "ATT_3"]
        regions = ["REG_A", "REG_B", "REG_C"]
        years = [2020, 2025, 2030, 2035]
        limtypes = ["LO", "UP", "FX"]
        modules = ["MOD_1", "MOD_2"]

        data = {
            "process": rng.choice(processes, n_rows),
            "commodity": rng.choice(commodities, n_rows),
            "attribute": rng.choice(attributes, n_rows),
            "region": rng.choice(regions, n_rows),
            "year": rng.choice(years, n_rows),
            "limtype": rng.choice(limtypes, n_rows),
            "value": rng.uniform(0, 100, n_rows),
            "module_name": rng.choice(modules, n_rows),
            "extra_col_1": rng.random(n_rows),
            "extra_col_2": rng.random(n_rows),
        }
        return pd.DataFrame(data)

    # Generate random queries
    def generate_random_queries(n_queries, table):
        queries = []
        while len(queries) < n_queries:
            query_dict = {
                "process": (
                    rng.choice(table["process"].unique())
                    if rng.random() > 0.5
                    else None
                ),
                "commodity": (
                    rng.choice(table["commodity"].unique())
                    if rng.random() > 0.5
                    else None
                ),
                "attribute": (
                    rng.choice(table["attribute"].unique())
                    if rng.random() > 0.5
                    else None
                ),
                "region": (
                    rng.choice(table["region"].unique()) if rng.random() > 0.8 else None
                ),
                "year": (
                    int(rng.choice(table["year"].unique()))
                    if rng.random() > 0.5
                    else None
                ),
                "limtype": (
                    rng.choice(table["limtype"].unique())
                    if rng.random() > 0.5
                    else None
                ),
                "val": (float(rng.uniform(0, 100)) if rng.random() > 0.9 else None),
                "module": (
                    rng.choice(table["module_name"].unique())
                    if rng.random() > 0.5
                    else None
                ),
            }
            if any(v is not None for _, v in query_dict.items()):
                queries.append(query_dict)
        return queries

    def queries_to_df(queries):
        # Convert list of query dicts to DataFrame, replacing None with pd.NA
        # First, explode queries that have lists of values in any column
        queries_exploded = []
        cols = queries[0].keys()
        for q in queries:
            # Get all possible combinations of list values
            values = [v if isinstance(v, list) else [v] for _, v in q.items()]

            # Create a dict for each combination
            for combination in product(*values):
                queries_exploded.append(dict(zip(cols, combination)))

        df = pd.DataFrame(queries_exploded)
        # Rename columns to match table
        df = df.rename(columns={"module": "module_name", "val": "value"})
        # Replace None with pd.NA
        return df.replace({None: pd.NA})

    # Use pickled data from GEO:
    table_query_file = Path("tfm_mig_queries-12276.pkl")
    # table_query_file = Path("tfm_mig_queries-217.pkl")
    with table_query_file.open("rb") as f:
        geo_table, queries_raw = pickle.load(f)
    queries_raw = queries_raw.head(1000)

    # Convert the raw queries into the expected format:
    geo_queries = []
    for _, row in queries_raw.iterrows():
        q = {
            "process": row.get("process", None),
            "commodity": row.get("commodity", None),
            "attribute": row.get("attribute", None),
            "region": row.get("region", None),
            "year": row.get("year", None),
            "limtype": row.get("limtype", None),
            "val": row.get("value", None),
            "module": None,
        }
        geo_queries.append(q)
    geo_queries_df = queries_to_df(geo_queries)

    # Compare results between old and new methods
    print("\nComparing query methods on GEO dataset:")

    # Test old method
    print("\nTesting original query method...")
    start_time = time.time()
    old_results = set()
    for q in geo_queries:
        old_results.update(query(geo_table, **q))
    old_total_time = time.time() - start_time

    # Test process pool method with different numbers of workers
    print("\nTesting process pool method with different worker counts...")

    # Get number of CPUs for scaling worker counts
    import multiprocessing

    n_cpus = multiprocessing.cpu_count()
    worker_counts = set([2, 4, 8, n_cpus, n_cpus * 2])

    results = []
    for n_workers in sorted(worker_counts):
        print(f"\nTesting with {n_workers} workers...")
        start_time = time.time()
        pool_results = set()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            actual_n_workers = executor._max_workers  # pyright: ignore
            # Split queries into chunks based on worker count
            chunk_size = max(1, len(geo_queries) // actual_n_workers)
            chunks = [
                geo_queries[i : i + chunk_size]
                for i in range(0, len(geo_queries), chunk_size)
            ]

            # Submit all chunks to the executor and collect results
            futures = [
                executor.submit(process_chunk, chunk, geo_table) for chunk in chunks
            ]
            for future in futures:
                pool_results.update(future.result())

        total_time = time.time() - start_time
        speedup = old_total_time / total_time
        results.append(
            {
                "workers": n_workers,
                "time": total_time,
                "speedup": speedup,
                "matches": len(pool_results),
                "correct": pool_results == old_results,
            }
        )

    print("\nPerformance comparison:")
    print("Original method:")
    print(f"Total time: {old_total_time:.2f} seconds")
    print(f"Total unique matches: {len(old_results)}")
    print("\nParallel method results:")
    print(f"{'Workers':>8} {'Time (s)':>10} {'Speedup':>10} {'Correct':>8}")
    print("-" * 40)
    for r in results:
        print(
            f"{r['workers']:8d} {r['time']:10.2f} {r['speedup']:10.2f} {r['correct']:8}"
        )

    sys.exit(0)

    # Test with small dataset first
    print("Testing with small dataset:")
    small_table = generate_random_data(10)
    print(f"Table shape: {small_table.shape}")
    small_queries = generate_random_queries(2, small_table)
    small_queries_df = queries_to_df(small_queries)

    # Compare results between old and new methods
    print("\nComparing query methods on small dataset:")

    # Get combined results from old method
    old_results = set()
    for q in small_queries:
        old_results.update(query(small_table, **q))

    # Get results from new method for all queries at once
    new_results = set(query_boolmask(small_table, small_queries_df))

    print(f"Old method total unique matches: {len(old_results)}")
    print(f"New method total matches: {len(new_results)}")
    print(f"Results identical: {old_results == new_results}")
    if old_results != new_results:
        print("Differences:")
        print(f"In old but not new: {old_results - new_results}")
        print(f"In new but not old: {new_results - old_results}")

    # Test with large dataset for performance comparison
    print("\nTesting with large dataset:")
    large_table = generate_random_data(350_000)
    print(f"Table shape: {large_table.shape}")
    large_queries = generate_random_queries(100, large_table)
    large_queries_df = queries_to_df(large_queries)

    # Test old method
    print("\nTesting original query method...")
    start_time = time.time()
    old_results = set()
    for q in large_queries:
        old_results.update(query(large_table, **q))
    old_total_time = time.time() - start_time

    # Test new method
    print("Testing new boolmask method...")
    start_time = time.time()
    new_results = set(query_boolmask(large_table, large_queries_df))
    new_total_time = time.time() - start_time

    print("\nPerformance comparison:")
    print("Original method:")
    print(f"Total time: {old_total_time:.2f} seconds")
    print(f"Total unique matches: {len(old_results)}")

    print("\nBoolmask method:")
    print(f"Total time: {new_total_time:.2f} seconds")
    print(f"Total matches: {len(new_results)}")
    print(f"Results identical: {old_results == new_results}")
    print(f"Speed improvement: {old_total_time / new_total_time:.1f}x")
