import gzip
import pickle
from datetime import datetime

import pandas as pd
from pandasql import sqldf
from tqdm import tqdm

from xl2times import transforms, utils, datatypes
from xl2times.transforms import (
    _process_comm_groups_vectorised,
    _count_comm_group_vectorised,
    expand_rows,
    get_matching_commodities,
    get_matching_processes,
    _match_wildcards,
    process_map,
    commodity_map,
    query,
    eval_and_update,
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
    """Old version of the process_uc_wildcards matching logic, for comparison with the new vectorised version.
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


def _eval_updates_old(table: pd.DataFrame, updates: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Evaluate the update formulas in updates on matching rows in table, returning updated rows.
    """
    new_tables = []

    # TODO perf: collect all updates and go through FI_T only once?
    for _, row in tqdm(
        updates.iterrows(),
        total=len(updates),
        desc=f"Applying transformations from {datatypes.Tag.tfm_upd.value}",
    ):
        rows_to_update = query(
            table,
            row["process"],
            row["commodity"],
            row["attribute"],
            row["region"],
            row["year"],
        )

        if not any(rows_to_update):
            logger.info(f"A {datatypes.Tag.tfm_upd.value} row generated no records.")
            continue

        new_rows = table.loc[rows_to_update].copy()
        new_rows["source_filename"] = row["source_filename"]
        eval_and_update(new_rows, rows_to_update, row["value"])
        new_tables.append(new_rows)
    return new_tables


class TestTransforms:
    def test_eval_updates(self):
        """
        Austimes performance:
            Finished eval_updates() in 0:01:57.146817
            Finished eval_updates_old() in 0:05:29.668738

        Ireland performance:
            Finished eval_updates() in 0:00:02.955761
            Finished eval_updates_old() in 0:00:03.532933

        """
        table = pickle.load(gzip.open("tests/data/tmp_fi_t_ireland.pkl.gz", "rb"))
        updates = pickle.load(gzip.open("tests/data/tmp_tfm_upd_ireland.pkl.gz", "rb"))

        # warmup sqldf
        sqldf("select * from updates limit 1")

        t0 = datetime.now()
        updates_new = transforms._eval_updates(table, updates).sort_values(
            by=["process", "commodity", "attribute", "region", "year"]
        )
        t1 = datetime.now()
        updates_old = _eval_updates_old(table, updates)
        updates_old = pd.concat(updates_old).sort_values(
            by=["process", "commodity", "attribute", "region", "year"]
        )
        t2 = datetime.now()

        logger.info(f"_eval_updates_old() took {t2 - t1} seconds")
        logger.info(
            f"_eval_updates() took {t1 - t0} seconds, a speedup of {((t2 - t1) / (t1 - t0)):.1f}x"
        )

        assert updates_new.shape == updates_old.shape, "shapes should be equal"
        assert (updates_new.dtypes == updates_old.dtypes).all()
        assert (
            (
                updates_old.fillna("")
                .reset_index(drop=True)
                .astype(float, errors="ignore")
                .round(3)
                .drop(columns="value")
                == updates_new.fillna("")
                .reset_index(drop=True)
                .astype(float, errors="ignore")
                .round(3)
                .drop(columns="value")
            )
            .all()
            .all()
        ), "non-value columns should be equal"
        assert (
            updates_old["value"]
            .reset_index(drop=True)
            .astype(float, errors="ignore")
            .round(3)
            == updates_new["value"]
            .reset_index(drop=True)
            .astype(float, errors="ignore")
            .round(3)
        ).all(), "value columns should be equal"

    def test_process_wildcards(self):
        with gzip.open("tests/data/process_wildcards_test_data.pkl.gz", "rb") as f:
            table = pd.read_pickle(f)
        with gzip.open("tests/data/process_wildcards_test_model.pkl.gz", "rb") as f:
            model = pd.read_pickle(f)
        t0 = datetime.now()
        result = transforms.process_wildcards(None, table, model)  # pyright: ignore
        logger.info(f"process_wildcards() took {datetime.now() - t0} seconds")

    def test_uc_wildcards(self):
        """
        Tests logic that matches wildcards in the process_uc_wildcards transform .

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
        assert df_new.fillna(-1).equals(
            df_old.fillna(-1)
        ), "Dataframes should be equal (ignoring Nones and NaNs)"

    def test_generate_commodity_groups(self):
        """
        Tests that the _count_comm_group_vectorised function works as expected.
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
        """Tests the default primary commodity group identification logic runs correctly.
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
    # TestTransforms().test_uc_wildcards()
    TestTransforms().test_eval_updates()
