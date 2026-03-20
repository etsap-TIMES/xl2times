import pandas as pd
from pandas import DataFrame

from xl2times import transforms, utils
from xl2times.datatypes import (
    Config,
    EmbeddedXlTable,
    TimesModel,
)
from xl2times.transforms import (
    _count_comm_group_vectorised,
    _match_wildcards,
    _process_comm_groups_vectorised,
    commodity_map,
    process_map,
)

utils.setup_logger(None)

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 75)
pd.set_option("display.precision", 3)

config = Config(
    mapping_file="times_mapping.txt",
    times_info_file="times-info.json",
    times_sets_file="times-sets.json",
    veda_tags_file="veda-tags.json",
    veda_attr_defaults_file="veda-attr-defaults.json",
    regions="",
    include_dummy_imports=False,
    case=None,
)

model = TimesModel()


class TestTransforms:
    def test_explode_process_commodity_cols(self):
        df = DataFrame(
            {
                "process": ["a", "b", ["c", "d"]],
                "commodity": [["v", "w", "x"], "y", "z"],
            }
        )
        df2 = transforms.explode_process_commodity_cols(
            None,  # pyright: ignore
            {"name": df.copy()},
            None,  # pyright: ignore
        )
        correct = DataFrame(
            {
                "process": ["a", "a", "a", "b", "c", "d"],
                "commodity": ["v", "w", "x", "y", "z", "z"],
            }
        )
        assert df2["name"].equals(correct)

    def test_uc_wildcards(self):
        """Tests logic that matches wildcards in the process_uc_wildcards transform."""
        import pickle

        df_in = pd.read_parquet("tests/data/process_uc_wildcards_ireland_data.parquet")
        with open("tests/data/process_uc_wildcards_ireland_dict.pkl", "rb") as f:
            dictionary = pickle.load(f)
        df = df_in.copy()

        for result_col, item_map in {
            "process": process_map,
            "commodity": commodity_map,
        }.items():
            df = _match_wildcards(df, item_map, dictionary, result_col)

        # unit tests
        assert df is not None and not df.empty
        assert (
            df.shape[0] >= df_in.shape[0]
        ), "should have more rows after processing uc_wildcards"
        assert (
            df.shape[1] < df_in.shape[1]
        ), "should have fewer columns after processing uc_wildcards"
        assert "process" in df.columns, "should have added process column"
        assert "commodity" in df.columns, "should have added commodity column"

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

    def test_harmonise_tradelinks(self):
        """Tests that the harmonise_tradelinks transform runs without error."""
        cols = ["COFFEE", "ECU", "EUR", "BRA"]
        data = [
            ["ECU", pd.NA, 1, "1.0"],
            ["EUR", "2", pd.NA, "COFFEE-TRD"],
            ["BRA", 0, pd.NA, 0],
        ]
        tables = [
            EmbeddedXlTable(
                tag="~TRADELINKS",
                uc_sets=dict(),
                sheetname="Uni_trades",
                range="",
                filename="",
                dataframe=DataFrame(data=data, columns=cols),
            ),
            EmbeddedXlTable(
                tag="~TRADELINKS",
                uc_sets=dict(),
                sheetname="Bi_trades",
                range="",
                filename="",
                dataframe=DataFrame(data=data, columns=cols),
            ),
            EmbeddedXlTable(
                tag="~TRADELINKS",
                uc_sets=dict(),
                sheetname="trades",
                range="",
                filename="",
                dataframe=DataFrame(data=data, columns=cols),
            ),
        ]

        expected = {
            "Uni_trades": {
                "tag": "~TRADELINKS_DINS",
                "processes": {
                    "TU_COFFEE_ECU_EUR_01",
                    "TU_COFFEE_ECU_BRA_01",
                    "TU_COFFEE_EUR_ECU_01",
                    "COFFEE-TRD",
                },
            },
            "Bi_trades": {
                "tag": "~TRADELINKS_DINS",
                "processes": {
                    "TB_COFFEE_ECU_EUR_01",
                    "TB_COFFEE_ECU_BRA_01",
                    "COFFEE-TRD",
                },
            },
            "trades": {
                "tag": "~TRADELINKS_DINS",
                "processes": {
                    "TB_COFFEE_ECU_EUR_01",
                    "TU_COFFEE_ECU_BRA_01",
                    "COFFEE-TRD",
                },
            },
        }

        transformed_tables = transforms.harmonise_tradelinks(
            config=config, tables=tables, model=model
        )
        for table in transformed_tables:
            test = table.sheetname
            assert (
                table.tag == expected[test]["tag"]
            ), f"{test} should have expected tag"
            assert (
                set(table.dataframe["process"]) == expected[test]["processes"]
            ), f"{test} should have expected trade processes"


if __name__ == "__main__":
    # TestTransforms().test_default_pcg_vectorised()
    TestTransforms().test_uc_wildcards()
