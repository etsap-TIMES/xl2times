import pandas as pd
import pytest
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
    _name_comm_groups_vectorised,
    _process_comm_groups_vectorised,
    commodity_map,
    process_map,
)

utils.setup_logger(None)


def _process_comm_groups_reference(
    comm_groups: DataFrame, csets_ordered_for_pcg: list[str]
) -> DataFrame:
    """Reference implementation of default PCG selection, used to check that the
    vectorised implementation produces identical output.

    This is a verbatim copy of the previous (groupby.apply-based) implementation of
    `transforms._process_comm_groups_vectorised`, including its quirk of flagging
    (io="IN", csets="DEM") rows whenever an OUT commodity group is selected as
    default (the `break` only exits the inner loop).
    """

    def _set_default_veda_pcg(group):
        """For a given [region, process] group, default group is set as the first cset
        in the `csets_ordered_for_pcg` list, which is an output, if one exists,
        otherwise the first input.
        """
        if not group["csets"].isin(csets_ordered_for_pcg).all():
            return group

        for io in ["OUT", "IN"]:
            for cset in csets_ordered_for_pcg:
                group.loc[
                    (group["io"] == io) & (group["csets"] == cset), "DefaultVedaPCG"
                ] = True
                if group["DefaultVedaPCG"].any():
                    break
        return group

    comm_groups["DefaultVedaPCG"] = None
    comm_groups_subset = comm_groups.groupby(
        ["region", "process"], sort=False, as_index=False
    ).apply(_set_default_veda_pcg)
    comm_groups_subset = comm_groups_subset.reset_index(
        level=0, drop=True
    ).sort_index()  # back to the original index and row order
    return comm_groups_subset


def _name_comm_groups_reference(comm_groups: DataFrame) -> DataFrame:
    """Reference implementation of commodity group naming (verbatim copy of the
    previous row-wise apply in `transforms.include_cgs_in_topology`).
    """

    def name_comm_group(df: pd.Series) -> str | None:
        """Generate the name of a commodity group based on the member count."""
        if df["commoditygroup"] > 1:
            return df["process"] + "_" + df["csets"] + df["io"][:1]
        elif df["commoditygroup"] == 1:
            return df["commodity"]
        else:
            return None

    comm_groups["commoditygroup"] = comm_groups.apply(name_comm_group, axis=1)
    return comm_groups


pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 75)
pd.set_option("display.precision", 3)


@pytest.fixture(scope="module")
def config() -> Config:
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

    def test_default_pcg_vectorised_matches_reference(self):
        """The vectorised default PCG selection must produce output identical to the
        previous groupby.apply implementation on real (austimes) data: same values,
        dtypes, row order and index.
        """
        comm_groups = pd.read_parquet("tests/data/austimes_pcg_test_data.parquet")
        # Subsample (region, process) groups so the (slow) reference implementation
        # keeps the test runtime reasonable, while retaining group diversity
        comm_groups = comm_groups[comm_groups["region"].isin(["ACT", "NT"])]
        processes = sorted(comm_groups["process"].unique())[::5]
        comm_groups = comm_groups[comm_groups["process"].isin(processes)]

        expected = _process_comm_groups_reference(
            comm_groups.copy(), transforms.csets_ordered_for_pcg
        )
        actual = _process_comm_groups_vectorised(
            comm_groups.copy(), transforms.csets_ordered_for_pcg
        )
        pd.testing.assert_frame_equal(actual, expected)
        assert actual.index.equals(expected.index)
        # both True rows and untouched (None) rows should exist in the test data
        assert (actual["DefaultVedaPCG"].values == True).any()  # noqa: E712
        assert actual["DefaultVedaPCG"].isna().any()

    def test_default_pcg_vectorised_edge_cases(self):
        """Synthetic edge cases for default PCG selection, including the preserved
        quirk of the original implementation: whenever an OUT commodity group wins,
        any (io="IN", csets="DEM") rows in the group are also flagged True.
        """
        # fmt: off
        rows = [
            # g1: OUT present -> (OUT, NRG) wins and the (IN, DEM) quirk triggers
            ("R1", "P1", "OUT", "C1", "NRG", True),
            ("R1", "P1", "IN", "C2", "DEM", True),  # quirk row
            ("R1", "P1", "IN", "C3", "NRG", None),
            # g2: only IN rows -> first cset present in order (MAT) wins
            ("R1", "P2", "IN", "C1", "MAT", True),
            ("R1", "P2", "IN", "C2", "NRG", None),
            ("R1", "P2", "IN", "C3", "FIN", None),
            # g3: a cset outside csets_ordered_for_pcg -> whole group untouched
            ("R1", "P3", "OUT", "C1", "NRG", None),
            ("R1", "P3", "IN", "C2", "XXX", None),
            # g4: io values outside IN/OUT are never flagged
            ("R1", "P4", "IN-A", "C1", "NRG", None),
            ("R1", "P4", "OUT-A", "C2", "DEM", None),
            # g5: OUT winner is not DEM; no (IN, DEM) rows so no quirk row
            ("R1", "P5", "OUT", "C1", "ENV", True),
            ("R1", "P5", "OUT", "C2", "FIN", None),
            ("R1", "P5", "IN", "C3", "MAT", None),
            # g6: OUT winner not DEM, plus quirk row and mixed io values
            ("R2", "P1", "OUT", "C1", "NRG", True),
            ("R2", "P1", "OUT", "C2", "ENV", None),
            ("R2", "P1", "IN", "C3", "DEM", True),  # quirk row
            ("R2", "P1", "IN", "C4", "DEM", True),  # quirk row
            ("R2", "P1", "IN-A", "C5", "DEM", None),
            # single-row groups
            ("R2", "P2", "OUT", "C1", "DEM", True),
            ("R2", "P3", "IN", "C1", "FIN", True),
            ("R2", "P4", "IN-A", "C1", "NRG", None),
            ("R2", "P5", "OUT", "C1", "XXX", None),
        ]
        # fmt: on
        comm_groups = DataFrame(
            [r[:5] for r in rows],
            columns=["region", "process", "io", "commodity", "csets"],
        )
        expected_flags = pd.Series([r[5] for r in rows], dtype=object)

        actual = _process_comm_groups_vectorised(
            comm_groups.copy(), transforms.csets_ordered_for_pcg
        )
        assert actual["DefaultVedaPCG"].equals(expected_flags)

        # and the reference implementation must agree
        expected = _process_comm_groups_reference(
            comm_groups.copy(), transforms.csets_ordered_for_pcg
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_name_comm_groups_vectorised_matches_reference(self):
        """The vectorised commodity group naming must produce output identical to the
        previous row-wise apply implementation on real (austimes) data.
        """
        comm_groups = pd.read_parquet(
            "tests/data/comm_groups_austimes_test_data.parquet"
        ).drop(columns=["commoditygroup"])
        comm_groups = comm_groups[comm_groups["region"].isin(["ACT", "NT"])]
        _count_comm_group_vectorised(comm_groups)

        expected = _name_comm_groups_reference(comm_groups.copy())
        actual = _name_comm_groups_vectorised(comm_groups.copy())
        pd.testing.assert_frame_equal(actual, expected)

    def test_name_comm_groups_vectorised_edge_cases(self):
        """Synthetic edge cases for commodity group naming."""
        comm_groups = DataFrame(
            {
                "region": ["R1"] * 4,
                "process": ["P1", "P1", "P1", "P2"],
                "io": ["OUT", "IN", "IN-A", "IN"],
                "commodity": ["C1", "C2", "C3", "C4"],
                "csets": ["NRG", "DEM", "NRG", "MAT"],
                "commoditygroup": [2, 1, 0, 3],
            }
        )
        expected_names = pd.Series(["P1_NRGO", "C2", None, "P2_MATI"], dtype=object)

        actual = _name_comm_groups_vectorised(comm_groups.copy())
        assert actual["commoditygroup"].equals(expected_names)

        expected = _name_comm_groups_reference(comm_groups.copy())
        pd.testing.assert_frame_equal(actual, expected)

    def test_harmonise_tradelinks(self, config):
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

        transformed_tables = transforms.harmonise_tradelinks(config, tables, model)
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
