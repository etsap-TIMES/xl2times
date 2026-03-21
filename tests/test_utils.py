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

    def test_create_regexp(self):
        """Test VEDA wildcard pattern to regex conversion."""
        # Basic wildcards
        assert utils.create_regexp("Elec*") == r"^Elec.*$"
        assert utils.create_regexp("Fuel?") == r"^Fuel.$"
        assert utils.create_regexp("Fuel_") == r"^Fuel.$"  # _ is equivalent to ?

        # Literal dots should be escaped
        assert utils.create_regexp("1.5MW") == r"^1\.5MW$"
        assert utils.create_regexp("coal.price") == r"^coal\.price$"

        # Literal underscore using [_]
        assert utils.create_regexp("Tech[_]1") == r"^Tech_1$"
        assert utils.create_regexp("Heat[_]Pump") == r"^Heat_Pump$"

        # Multiple wildcards
        assert utils.create_regexp("?_*") == r"^...*$"
        assert utils.create_regexp("Tech*[_]?") == r"^Tech.*_.$"

        # Comma-separated values
        assert utils.create_regexp("Coal,Gas") == r"^Coal$|^Gas$"
        assert utils.create_regexp("Coal,Gas,Oil") == r"^Coal$|^Gas$|^Oil$"

        # Patterns with spaces (should be stripped)
        assert utils.create_regexp(" Coal , Gas ") == r"^Coal$|^Gas$"

        # Mixed wildcards and commas
        assert utils.create_regexp("Heat*,Cool?") == r"^Heat.*$|^Cool.$"

        # Empty pattern matches everything
        assert utils.create_regexp("") == r".*"

        # Negative patterns are ignored by create_regexp
        assert utils.create_regexp("Heat*,-HeatPump") == r"^Heat.*$"
        assert utils.create_regexp("-Gas*,-Oil*") == r".*"

        # Complex combinations
        assert (
            utils.create_regexp("Tech[_]?,Fuel.*,1.5*")
            == r"^Tech_.$|^Fuel\..*$|^1\.5.*$"
        )

    def test_create_negative_regexp(self):
        """Test negative pattern handling."""
        # Basic negative patterns
        assert utils.create_negative_regexp("Heat*,-HeatPump") == r"^HeatPump$"
        assert utils.create_negative_regexp("-Gas*,-Oil*") == r"^Gas.*$|^Oil.*$"

        # Single negative with wildcards
        assert utils.create_negative_regexp("-Heat*") == r"^Heat.*$"
        assert utils.create_negative_regexp("-Cool?") == r"^Cool.$"

        # Multiple negatives with wildcards
        assert (
            utils.create_negative_regexp("-Heat*,-Cool?,-1.5MW")
            == r"^Heat.*$|^Cool.$|^1\.5MW$"
        )

        # Mixed positive and negative (positives are ignored)
        assert utils.create_negative_regexp("Coal,Gas,-Oil*") == r"^Oil.*$"

        # Pattern with spaces (should be stripped)
        assert utils.create_negative_regexp(" -Coal , -Gas ") == r"^Coal$|^Gas$"

        # Empty pattern or no negatives matches nothing
        assert utils.create_negative_regexp("") == r"^$"
        assert utils.create_negative_regexp("Coal,Gas") == r"^$"

        # Complex combinations
        assert (
            utils.create_negative_regexp("Tech*,-Tech[_]?,-Tech.Old")
            == r"^Tech_.$|^Tech\.Old$"
        )


if __name__ == "__main__":
    TestUtils().test_explode()
