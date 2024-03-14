from datetime import datetime

import pandas as pd
from pandas import DataFrame

from xl2times import datatypes, transforms, utils
from xl2times.transforms import (
    _count_comm_group_vectorised,
    _match_wildcards,
    _process_comm_groups_vectorised,
    commodity_map,
    expand_rows,
    get_matching_commodities,
    get_matching_processes,
    process_map,
)

logger = utils.get_logger()

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


def _match_uc_wildcards_old(
    df: pd.DataFrame, dictionary: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Old version of the process_uc_wildcards matching logic, for comparison with the
    new vectorised version.

    TODO remove this function once validated.
    """

    def make_str(df):
        if df is not None and len(df) != 0:
            list_from_df = df.iloc[:, 0].unique()
            return ",".join(list_from_df)
        else:
            return None

    df["process"] = df.apply(
        lambda row: make_str(get_matching_processes(row, dictionary)), axis=1
    )
    df["commodity"] = df.apply(
        lambda row: make_str(get_matching_commodities(row, dictionary)), axis=1
    )

    query_columns = transforms.process_map.keys() | transforms.commodity_map.keys()
    cols_to_drop = [col for col in df.columns if col in query_columns]

    df = expand_rows(
        query_columns,
        datatypes.EmbeddedXlTable(
            tag="",
            uc_sets={},
            sheetname="",
            range="",
            filename="",
            dataframe=df.drop(columns=cols_to_drop),
        ),
    ).dataframe
    return df


class TestTransforms:
    def test_explode_process_commodity_cols(self):
        df = DataFrame(
            {
                "process": ["a", "b", ["c", "d"]],
                "commodity": [["v", "w", "x"], "y", "z"],
            }
        )
        df2 = transforms.explode_process_commodity_cols(
            None, {"name": df.copy()}, None  # pyright: ignore
        )
        correct = DataFrame(
            {
                "process": ["a", "a", "a", "b", "c", "d"],
                "commodity": ["v", "w", "x", "y", "z", "z"],
            }
        )
        assert df2["name"].equals(correct)

    def test_uc_wildcards(self):
        """Tests logic that matches wildcards in the process_uc_wildcards transform .

        Results on Ireland model:
            Old method took 0:00:08.42 seconds
            New method took 0:00:00.18 seconds, speedup: 46.5x
        """
        import pickle

        df_in = pd.read_parquet("tests/data/process_uc_wildcards_ireland_data.parquet")
        with open("tests/data/process_uc_wildcards_ireland_dict.pkl", "rb") as f:
            dictionary = pickle.load(f)
        df = df_in.copy()

        t0 = datetime.now()

        # optimised functions
        df_new = _match_wildcards(
            df, process_map, dictionary, get_matching_processes, "process"
        )
        df_new = _match_wildcards(
            df_new, commodity_map, dictionary, get_matching_commodities, "commodity"
        )

        t1 = datetime.now()

        # Unoptimised function
        df_old = _match_uc_wildcards_old(df, dictionary)

        t2 = datetime.now()

        logger.info(f"Old method took {t2 - t1} seconds")
        logger.info(
            f"New method took {t1 - t0} seconds, speedup: {((t2 - t1) / (t1 - t0)):.1f}x"
        )

        # unit tests
        assert df_new is not None and not df_new.empty
        assert (
            df_new.shape[0] >= df_in.shape[0]
        ), "should have more rows after processing uc_wildcards"
        assert (
            df_new.shape[1] < df_in.shape[1]
        ), "should have fewer columns after processing uc_wildcards"
        assert "process" in df_new.columns, "should have added process column"
        assert "commodity" in df_new.columns, "should have added commodity column"

        # consistency checks with old method
        assert len(set(df_new.columns).symmetric_difference(set(df_old.columns))) == 0

    def test_generate_commodity_groups(self):
        """Tests that the _count_comm_group_vectorised function works as expected.

        Full austimes run:
            Vectorised version took 0.021999 seconds
            looped version took 966.653371 seconds
            43958x speedup
        """
        # data extracted immediately before the original for loops
        comm_groups = pd.read_parquet(
            "tests/data/comm_groups_austimes_test_data.parquet"
        ).drop(columns=["commoditygroup"])

        # filter data so test runs faster
        comm_groups = comm_groups.query("region in ['ACT', 'NSW']")

        comm_groups2 = comm_groups.copy()
        _count_comm_group_vectorised(comm_groups2)
        assert comm_groups2.drop(columns=["commoditygroup"]).equals(comm_groups)
        assert comm_groups2.shape == (comm_groups.shape[0], comm_groups.shape[1] + 1)

    def test_default_pcg_vectorised(self):
        """Tests the default primary commodity group identification logic runs
        correctly.

        Full austimes run:
            Looped version took 1107.66 seconds
            Vectorised version took 62.85 seconds
        """
        # data extracted immediately before the original for loops
        comm_groups = pd.read_parquet("tests/data/austimes_pcg_test_data.parquet")

        comm_groups = comm_groups[(comm_groups["region"].isin(["ACT", "NT"]))]
        comm_groups2 = _process_comm_groups_vectorised(
            comm_groups.copy(), transforms.csets_ordered_for_pcg
        )
        assert comm_groups2 is not None and not comm_groups2.empty
        assert comm_groups2.shape == (comm_groups.shape[0], comm_groups.shape[1] + 1)
        assert comm_groups2.drop(columns=["DefaultVedaPCG"]).equals(comm_groups)


if __name__ == "__main__":
    # TestTransforms().test_default_pcg_vectorised()
    TestTransforms().test_uc_wildcards()
