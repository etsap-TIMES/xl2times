import pandas as pd
import pytest
from pandas import DataFrame

from xl2times import transforms, utils
from xl2times.datatypes import (
    Config,
    EmbeddedXlTable,
    Tag,
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


@pytest.fixture(scope="module")
def create_config() -> Config:
    """A fixture to create Config."""
    return Config(
        mapping_file="times_mapping.txt",
        times_info_file="times-info.json",
        times_sets_file="times-sets.json",
        veda_tags_file="veda-tags.json",
        veda_attr_defaults_file="veda-attr-defaults.json",
        regions="",
        include_dummy_imports=False,
        case=None,
    )


class TestTransforms:
    def test_process_time_periods_uses_active_milestone_year_definition(self):
        model = TimesModel()
        tables = [
            EmbeddedXlTable(
                tag=Tag.start_year,
                uc_sets={},
                sheetname="TimePeriods",
                range="B4",
                filename="SysSettings.xlsx",
                dataframe=DataFrame({"value": [2022]}),
            ),
            EmbeddedXlTable(
                tag=Tag.active_p_def,
                uc_sets={},
                sheetname="TimePeriods",
                range="B8",
                filename="SysSettings.xlsx",
                dataframe=DataFrame({"value": ["msy10_2055"]}),
            ),
            EmbeddedXlTable(
                tag=Tag.time_periods,
                uc_sets={},
                sheetname="TimePeriods",
                range="B12:C23",
                filename="SysSettings.xlsx",
                dataframe=DataFrame({"10p2050": [1, 1, 1, 5]}),
            ),
            EmbeddedXlTable(
                tag=Tag.milestoneyears,
                uc_sets={},
                sheetname="TimePeriods",
                range="E12:O33",
                filename="SysSettings.xlsx",
                dataframe=DataFrame(
                    {
                        "type": ["milestoneyear"] * 4,
                        "msy10_2055": [2022, 2023, 2024, 2025],
                    }
                ),
            ),
        ]

        transforms.process_time_periods(None, tables, model)

        assert model.time_periods[["year", "b", "e", "m"]].to_dict("records") == [
            {"year": 2022, "b": 2022, "e": 2022, "m": 2022},
            {"year": 2023, "b": 2023, "e": 2023, "m": 2023},
            {"year": 2024, "b": 2024, "e": 2024, "m": 2024},
            {"year": 2025, "b": 2025, "e": 2029, "m": 2025},
        ]

    def test_process_transform_availability_defaults_missing_value_to_one(self):
        table = EmbeddedXlTable(
            tag=Tag.tfm_ava,
            uc_sets={},
            sheetname="AVA",
            range="C4:E224",
            filename="SubRES_Wind-NREL-C_Trans.xlsx",
            dataframe=DataFrame({"pset_pn": ["*[_]AFG"], "region": ["OAS"]}),
        )

        [processed] = transforms.process_transform_availability(None, [table], None)

        assert processed.dataframe["value"].iloc[0] == 1

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

    def test_harmonise_tradelinks(self, create_config):
        """Tests that harmonise_tradelinks runs successfully and produces tables with expected tags and trade processes."""
        model = TimesModel()
        cols = ["COFFEE", "ECU", "EUR", "BRA"]
        data = [
            ["ECU", pd.NA, 1, "1.0"],
            ["EUR", "2", "0.0", "COFFEE-TRD"],
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
            config=create_config, tables=tables, model=model
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
