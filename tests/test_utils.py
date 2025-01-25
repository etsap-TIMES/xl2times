import pandas as pd

from xl2times import utils


class TestUtils:
    def test_explode(self):
        """Test that explode logics functions correctly."""
        input_df1 = pd.DataFrame(
            [
                ("PRC1", 100, pd.NA, 200),
                ("PRC2", 100, 150, pd.NA),
                ("PRC3", pd.NA, 150, 200),
            ],
            columns=["process", "act_bnd", "actbnd", "act_bnd"],
        )

        input_df2 = pd.DataFrame(
            [
                ("PRC1", 100, pd.NA),
                ("PRC2", 100, 150),
                ("PRC3", pd.NA, 150),
            ],
            columns=["process", "act_bnd", "actbnd"],
        )

        data_cols1 = ["act_bnd", "actbnd", "act_bnd"]
        data_cols2 = ["act_bnd", "actbnd"]

        correct_index1 = pd.RangeIndex(9)
        correct_index2 = pd.RangeIndex(6)

        correct_result1 = (
            pd.DataFrame(
                [
                    ("PRC1", 100),
                    ("PRC1", pd.NA),
                    ("PRC1", 200),
                    ("PRC2", 100),
                    ("PRC2", 150),
                    ("PRC2", pd.NA),
                    ("PRC3", pd.NA),
                    ("PRC3", 150),
                    ("PRC3", 200),
                ],
                columns=["process", "value"],
                index=correct_index1,
                dtype=object,
            ),
            pd.Series(
                [
                    "act_bnd",
                    "actbnd",
                    "act_bnd",
                    "act_bnd",
                    "actbnd",
                    "act_bnd",
                    "act_bnd",
                    "actbnd",
                    "act_bnd",
                ],
                index=correct_index1,
            ),
        )

        correct_result2 = (
            pd.DataFrame(
                [
                    ("PRC1", 100),
                    ("PRC1", pd.NA),
                    ("PRC2", 100),
                    ("PRC2", 150),
                    ("PRC3", pd.NA),
                    ("PRC3", 150),
                ],
                columns=["process", "value"],
                index=correct_index2,
                dtype=object,
            ),
            pd.Series(
                ["act_bnd", "actbnd", "act_bnd", "actbnd", "act_bnd", "actbnd"],
                index=correct_index2,
            ),
        )

        output1 = utils.explode(input_df1, data_cols1)
        output2 = utils.explode(input_df2, data_cols2)

        assert output1[0].equals(correct_result1[0]), "Dataframes should be equal"
        assert output1[1].equals(correct_result1[1]), "Series should be equal"
        assert output2[0].equals(correct_result2[0]), "Dataframes should be equal"
        assert output2[1].equals(correct_result2[1]), "Series should be equal"


if __name__ == "__main__":
    TestUtils().test_explode()
