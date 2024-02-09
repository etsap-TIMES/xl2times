import pickle
from datetime import datetime
from pathlib import Path
from timeit import timeit

import pandas as pd
import sys
from loguru import logger
from tqdm import tqdm

from xl2times import datatypes, transforms
from xl2times.transforms import (
    _process_comm_groups_looped,
    _process_comm_groups_vectorised,
    _count_comm_group_vectorised,
    _count_comm_group_looped,
)

pd.set_option(
    "display.max_rows",
    20,
    "display.max_columns",
    20,
    "display.width",
    300,
    "display.max_colwidth",
    75,
    "display.precision",
    3,
)


class TestTransforms:
    def test_generate_commodity_groups(self):
        """
        Full run, all regions:
            Vectorised version took 0.021999 seconds
            looped version took 966.653371 seconds
            43958x speedup!
        """
        # data extracted immediately before the original for loops
        comm_groups = pd.read_parquet(
            "tests/data/comm_groups_austimes_test_data.parquet"
        )

        # filter data so test runs faster
        comm_groups = comm_groups.query("region in ['ACT', 'NSW']")

        t1 = comm_groups.copy()
        t2 = comm_groups.copy()

        s0 = datetime.now()
        _count_comm_group_vectorised(t2)
        s1 = datetime.now()
        _count_comm_group_looped(t1)
        s2 = datetime.now()

        logger.debug(
            f"_count_comm_group_vectorised took {(s1 - s0).total_seconds()} seconds, _count_comm_group_looped took "
            f"{(s2 - s1).total_seconds()} seconds, {(s2 - s1).microseconds / (s1 - s0).microseconds :1f}x speedup"
        )
        assert t1.equals(t2), "Looped and vectorised versions should be equal"

    def test_default_pcg_vectorised(self):
        """Tests the default primary commodity group identification logic in vectorised form.
        Full run:
            Looped version took 1107.66 seconds
            Vectorised version took 62.85 seconds
        """

        # data extracted immediately before the original for loops
        comm_groups = pd.read_parquet("tests/data/austimes_pcg_test_data.parquet")

        t1 = comm_groups[
            (comm_groups["process"] == "EN_CSP15-Q8")
            & (comm_groups["region"].isin(["ACT"]))
        ]  # smaller test set
        cg_loop = _process_comm_groups_looped(
            t1.copy(), transforms.csets_ordered_for_pcg
        )
        cg_vec = _process_comm_groups_vectorised(
            t1.copy(), transforms.csets_ordered_for_pcg
        )
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t2 = comm_groups[
            (comm_groups["process"] == "EN_CSP15-Q8")
            & (comm_groups["region"].isin(["ACT", "NSW"]))
        ]  # larger test set
        cg_loop = _process_comm_groups_looped(
            t2.copy(), transforms.csets_ordered_for_pcg
        )
        cg_vec = _process_comm_groups_vectorised(
            t2.copy(), transforms.csets_ordered_for_pcg
        )
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t3 = comm_groups[
            (comm_groups["process"] == "TC_Age_Care")
            & (comm_groups["region"].isin(["ACT", "NSW"]))
        ]  # all processes, 2 regions
        cg_loop = _process_comm_groups_looped(
            t3.copy(), transforms.csets_ordered_for_pcg
        )
        cg_vec = _process_comm_groups_vectorised(
            t3.copy(), transforms.csets_ordered_for_pcg
        )

        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t4 = comm_groups[
            (comm_groups["process"].isin(["TC_Age_Care", "EN_CSP15-Q8"]))
            & (comm_groups["region"].isin(["ACT", "NT"]))
        ]
        cg_loop = _process_comm_groups_looped(
            t4.copy(), transforms.csets_ordered_for_pcg
        )
        cg_vec = _process_comm_groups_vectorised(
            t4.copy(), transforms.csets_ordered_for_pcg
        )
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t5 = comm_groups[(comm_groups["region"].isin(["ACT", "NT", "NSW"]))]
        cg_loop = _process_comm_groups_looped(
            t5.copy(), transforms.csets_ordered_for_pcg
        )
        cg_vec = _process_comm_groups_vectorised(
            t5.copy(), transforms.csets_ordered_for_pcg
        )
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        # Benchmark:
        # t6 = comm_groups # for full run
        t6 = t5  # for smaller run
        t_looped = timeit(
            lambda: _process_comm_groups_looped(
                t6.copy(), transforms.csets_ordered_for_pcg
            ),
            number=1,
        )
        t_vector = timeit(
            lambda: _process_comm_groups_vectorised(
                t6.copy(), transforms.csets_ordered_for_pcg
            ),
            number=1,
        )
        logger.info(
            f"Looped version took {t_looped:.2f} seconds, vectorised version took {t_vector:.2f} seconds"
        )


if __name__ == "__main__":
    TestTransforms().test_default_pcg_vectorised()
