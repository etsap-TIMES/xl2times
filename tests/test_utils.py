from xl2times import utils
import pandas as pd


class TestUtils:
    def test_explode(self):
        """
        Test that explode logics functions correctly
        """

        input_df1 = pd.DataFrame(
            [
                ("PRC1", 100, None, 200),
                ("PRC2", 100, 150, None),
                ("PRC3", None, 150, 200),
            ],
            columns=["process", "act_bnd", "actbnd", "act_bnd"],
        )

        input_df2 = pd.DataFrame(
            [
                ("PRC1", 100, None),
                ("PRC2", 100, 150),
                ("PRC3", None, 150),
            ],
            columns=["process", "act_bnd", "actbnd"],
        )

        data_cols1 = ["act_bnd", "actbnd", "act_bnd"]
        data_cols2 = ["act_bnd", "actbnd"]

        correct_result1 = pd.DataFrame(
            [
                ("PRC1", "act_bnd", 100),
                ("PRC1", "act_bnd", 200),
                ("PRC2", "act_bnd", 100),
                ("PRC2", "actbnd", 150),
                ("PRC3", "actbnd", 150),
                ("PRC3", "act_bnd", 200),
            ],
            columns=["process", "attribute", "value"],
        )

        correct_result2 = pd.DataFrame(
            [
                ("PRC1", "act_bnd", 100),
                ("PRC2", "act_bnd", 100),
                ("PRC2", "actbnd", 150),
                ("PRC3", "actbnd", 150),
            ],
            columns=["process", "attribute", "value"],
        )

        output_df1, _ = utils.explode(input_df1, data_cols1)
        output_df2, _ = utils.explode(input_df2, data_cols2)

        assert output_df1.equals(correct_result1), "Dataframes should be equal"
        assert output_df2.equals(correct_result2), "Dataframes should be equal"


if __name__ == "__main__":
    TestUtils().test_explode()
