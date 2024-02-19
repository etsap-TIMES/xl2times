import re
from datetime import datetime
from re import Pattern

import pandas as pd
from pandas.core.common import flatten

from xl2times import transforms, datatypes
from xl2times.transforms import (
    _process_comm_groups_vectorised,
    _count_comm_group_vectorised,
    get_matching_commodities,
    intersect,
    filter_by_pattern,
    expand_rows,
    query_columns,
)
from xl2times.utils import create_regexp, create_negative_regexp

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


def get_matching_processes(row, dictionary):
    matching_processes = None
    for col, key in [
        ("pset_pn", "processes_by_name"),
        ("pset_pd", "processes_by_desc"),
        ("pset_set", "processes_by_sets"),
        ("pset_ci", "processes_by_comm_in"),
        ("pset_co", "processes_by_comm_out"),
    ]:
        if row[col] is not None:
            matching_processes = intersect(
                matching_processes,
                filter_by_pattern(
                    dictionary[key], row[col].upper()
                ),  # 20% of runtime here.  Avoid regex if no wildcard chars in string?
            )
    if matching_processes is not None and any(matching_processes.duplicated()):
        raise ValueError("duplicated")
    return matching_processes


class TestTransforms:
    def test_uc_wildcards(self):
        import pickle

        dfo = pd.read_parquet("tests/data/process_uc_wildcards_austimes_data.parquet")
        with open("tests/data/process_uc_wildcards_austimes_dict.pkl", "rb") as f:
            dictionary = pickle.load(f)

        df = dfo.query("region in ['ACT']")

        # df = dfo.query("region in ['NSW']")
        # row count per region

        def make_str(df):
            if df is not None and len(df) != 0:
                list_from_df = df.iloc[:, 0].unique()  # 60% of runtime here
                return ",".join(list_from_df)
            else:
                return None

        wildcard_map = {
            "pset_pn": "processes_by_name",
            "pset_pd": "processes_by_desc",
            "pset_set": "processes_by_sets",
            "pset_ci": "processes_by_comm_in",
            "pset_co": "processes_by_comm_out",
        }

        commodity_map = {
            "cset_cn": "commodities_by_name",
            "cset_cd": "commodities_by_desc",
            "cset_set": "commodities_by_sets",
        }

        t0 = datetime.now()

        # This apply() just gets matches the wildcards of process names in the tables against the list of all process names
        # Then because there can be multiple matches per table row, `expand_rows` melts the result into long format.
        # We can probably do this a lot faster by building a list of all wildcard matches first (avoiding duplicate lookups) as a dataframe
        # and then doing an outer-join with the original dataframe.

        match_dfs = []
        for pname_short, pname_long in wildcard_map.items():
            wildcards = df[pname_short].dropna().unique()
            processes = dictionary[pname_long]
            matches = {
                w: filter_by_pattern(processes, wildcards[0].upper())[
                    "process"
                ].to_list()
                for w in wildcards
            }

            proc_match_df = pd.DataFrame(
                matches.items(), columns=["wildcard", "matches"]
            )
            proc_match_df["pname_short"] = pname_short
            match_dfs.append(proc_match_df)
        wildcard_matches = pd.concat(match_dfs).explode("matches")

        # now cross-join wildcard_matches with df
        df2 = df.copy()
        df2["process"] = None
        for pname_short in wildcard_map.keys():
            if pname_short in df2.columns and any(
                pname_short == wildcard_matches["pname_short"]
            ):
                wild = wildcard_matches[wildcard_matches["pname_short"] == pname_short]
                df2 = df2.merge(
                    wild, left_on=pname_short, right_on="wildcard", how="left"
                ).drop(columns=["wildcard", "pname_short"])
                # update process column with matches for pname_short rows
                df2["process"].update(df2["matches"])
                df2 = df2.drop(columns=["matches"])
        # df2 = df2.drop(columns=wildcard_map.keys())
        # df2 = df2.drop(columns=commodity_map.keys())

        df["process"] = df.apply(
            lambda row: make_str(get_matching_processes(row, dictionary)), axis=1
        )

        t1 = datetime.now()
        print(f"get_matching_processes took {t1 - t0} seconds")

        # df["commodity"] = df.apply(
        #     lambda row: make_str(get_matching_commodities(row, dictionary)), axis=1
        # )
        # t2 = datetime.now()
        # print(f"get_matching_commodities took {t2 - t1} seconds")

        cols_to_drop = [col for col in df.columns if col in query_columns]

        dfe = expand_rows(
            datatypes.EmbeddedXlTable(
                tag="",
                uc_sets={},
                sheetname="",
                range="",
                filename="",
                dataframe=df,  # .drop(columns=cols_to_drop),
            )
        ).dataframe
        assert len(set(dfe.columns).symmetric_difference(set(df2.columns))) == 0

        assert all(
            dfe.query(
                "`pset_ci`=='ELCHYD,ELCSOL,ELCWIN,Wind01,Solar01,Hydro01'"
            ).process
            == df2.query(
                "`pset_ci`=='ELCHYD,ELCSOL,ELCWIN,Wind01,Solar01,Hydro01'"
            ).process
        )
        # set column order the same
        df2 = df2[dfe.columns]

        assert (dfe.reset_index(drop=True) == df2.reset_index(drop=True)).all().all()

        print(dfe)

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
    TestTransforms().test_uc_wildcards()
