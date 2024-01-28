import pandas as pd
import sys
from loguru import logger
from tqdm import tqdm

from xl2times import datatypes, transforms
from xl2times.transforms import pcg_looped, pcg_vectorised

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


class TestXL2Times:
    def test_default_pcg_vectorised(self):
        """Tests the default primary commodity group identification logic in vectorised form."""
        comm_groups = pd.read_parquet("tests/data/austimes_pcg_test_data.pq")
        regions = comm_groups["region"].unique()

        t1 = comm_groups[
            (comm_groups["process"] == "EN_CSP15-Q8")
            & (comm_groups["region"].isin(["ACT"]))
        ]  # smaller test set
        cg_loop = pcg_looped(t1.copy(), transforms.csets_ordered_for_pcg)
        cg_vec = pcg_vectorised(t1.copy(), transforms.csets_ordered_for_pcg)
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t2 = comm_groups[
            (comm_groups["process"] == "EN_CSP15-Q8")
            & (comm_groups["region"].isin(["ACT", "NSW"]))
        ]  # larger test set
        cg_loop = pcg_looped(t2.copy(), transforms.csets_ordered_for_pcg)
        cg_vec = pcg_vectorised(t2.copy(), transforms.csets_ordered_for_pcg)
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t3 = comm_groups[
            (comm_groups["process"] == "TC_Age_Care")
            & (comm_groups["region"].isin(["ACT", "NSW"]))
        ]  # all processes, 2 regions
        cg_loop = pcg_looped(t3.copy(), transforms.csets_ordered_for_pcg)
        cg_vec = pcg_vectorised(t3.copy(), transforms.csets_ordered_for_pcg)
        cg_loop.fillna(False) == cg_vec.fillna(False)
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t4 = comm_groups[
            (comm_groups["process"].isin(["TC_Age_Care", "EN_CSP15-Q8"]))
            & (comm_groups["region"].isin(["ACT", "NT"]))
        ]
        cg_loop = pcg_looped(t4.copy(), transforms.csets_ordered_for_pcg)
        cg_vec = pcg_vectorised(t4.copy(), transforms.csets_ordered_for_pcg)
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"

        t5 = comm_groups[(comm_groups["region"].isin(["ACT", "NT"]))]
        cg_loop = pcg_looped(t5.copy(), transforms.csets_ordered_for_pcg)
        cg_vec = pcg_vectorised(t5.copy(), transforms.csets_ordered_for_pcg)
        assert cg_loop.fillna(False).equals(
            cg_vec.fillna(False)
        ), "Looped and vectorised versions should be equal"
