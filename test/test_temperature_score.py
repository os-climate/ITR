import os
import unittest
import warnings

import pandas as pd

import ITR  # noqa F401
from ITR.configs import ColumnsConfig
from ITR.data.osc_units import Q_, asPintDataFrame, requantify_df_from_columns, ureg
from ITR.interfaces import EScope, ETimeFrames
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore

from utils import assert_pint_series_equal


class TestTemperatureScore(unittest.TestCase):
    """
    Test the reporting functionality. We'll use the Example data provider as the output of this provider is known in
    advance.
    """

    def setUp(self) -> None:
        """
        Create the provider and reporting instance which we'll use later on.
        :return:
        """
        self.temperature_score = TemperatureScore(time_frames=[ETimeFrames.LONG], scopes=EScope.get_result_scopes())
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "inputs",
                "data_test_temperature_score.csv",
            ),
            sep=";",
        )
        # Take care of INVESTMENT_VALUE
        requantify_df_from_columns(df, inplace=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # FIXME: should update CSV data to include a SCOPE column
            df[ColumnsConfig.SCOPE] = EScope.S1S2
            df.loc[df.company_name.eq("Company E"), ColumnsConfig.SCOPE] = EScope.S3
            df.loc[df.company_name.eq("Company AA"), ColumnsConfig.SCOPE] = EScope.S1S2
            df[ColumnsConfig.GHG_SCOPE12] = df[ColumnsConfig.GHG_SCOPE3].astype("pint[t CO2]")
            df[ColumnsConfig.GHG_SCOPE3] = df[ColumnsConfig.GHG_SCOPE3].astype("pint[t CO2]")
            for cumulative in [
                ColumnsConfig.CUMULATIVE_BUDGET,
                ColumnsConfig.CUMULATIVE_TARGET,
                ColumnsConfig.CUMULATIVE_TRAJECTORY,
            ]:
                df[cumulative] = df[cumulative].astype("pint[Mt CO2]")
            df[ColumnsConfig.BENCHMARK_GLOBAL_BUDGET] = df[ColumnsConfig.BENCHMARK_GLOBAL_BUDGET].astype("pint[Gt CO2]")
            df[ColumnsConfig.BENCHMARK_TEMP] = df[ColumnsConfig.BENCHMARK_TEMP].astype("pint[delta_degC]")
            for col in [
                ColumnsConfig.COMPANY_REVENUE,
                ColumnsConfig.COMPANY_MARKET_CAP,
                ColumnsConfig.COMPANY_ENTERPRISE_VALUE,
                ColumnsConfig.COMPANY_EV_PLUS_CASH,
                ColumnsConfig.COMPANY_TOTAL_ASSETS,
                ColumnsConfig.COMPANY_CASH_EQUIVALENTS,
            ]:
                df[col] = df[col].astype("pint[USD]")
            df[ColumnsConfig.INVESTMENT_VALUE] = df[ColumnsConfig.INVESTMENT_VALUE].astype("pint[USD]")
        self.data = df.set_index(["company_id"])

    def test_temp_score(self) -> None:
        """
        Test whether the temperature score is calculated as expected.

        :return:
        """
        exp_trce_mul = 0.0006004366812227075

        scores = self.temperature_score.calculate(self.data)

        exp_weight = 1.0
        exp_target_overshoot = 562.6345726 / 229.7868989
        exp_trajectory_overshoot = 528.250411 / 229.7868989
        exp_target_score = 1.5 + (exp_weight * 396.0 * (exp_target_overshoot - 1.0) * exp_trce_mul)
        exp_trajectory_score = 1.5 + (exp_weight * 396.0 * (exp_trajectory_overshoot - 1.0) * exp_trce_mul)
        exp_score = exp_target_score * 0.428571429 + exp_trajectory_score * (1 - 0.428571429)
        self.assertAlmostEqual(
            scores[(scores["company_name"] == "Company T") & (scores["scope"] == EScope.S1S2)][
                "temperature_score"
            ].iloc[0],
            Q_(exp_score, ureg.delta_degC),
            places=2,
            msg="The temp score was incorrect",
        )

        exp_weight = 1.0
        exp_target_overshoot = 699.9763453 / 223.2543241
        exp_trajectory_overshoot = 417.115686 / 223.2543241
        exp_target_score = 1.5 + (exp_weight * 396.0 * (exp_target_overshoot - 1.0) * exp_trce_mul)
        exp_trajectory_score = 1.5 + (exp_weight * 396.0 * (exp_trajectory_overshoot - 1.0) * exp_trce_mul)
        exp_score = exp_target_score * 0.428571429 + exp_trajectory_score * (1 - 0.428571429)
        self.assertAlmostEqual(
            scores[(scores["company_name"] == "Company E") & (scores["scope"] == EScope.S3)]["temperature_score"].iloc[
                0
            ],
            Q_(exp_score, ureg.delta_degC),
            places=2,
            msg="The fallback temp score was incorrect",
        )

        # aggregate S1S2S3 calculation has an extra post-processing step,
        # which overwrites call of call of TemperatureScore.get_score()
        self.assertAlmostEqual(
            scores[
                (scores["company_name"] == "Company AA")
                & (scores["time_frame"] == ETimeFrames.LONG)
                & (scores["scope"] == EScope.S1S2)
            ]["temperature_score"].iloc[0],
            Q_(1.79, ureg.delta_degC),
            places=2,
            msg="The aggregated fallback temp score was incorrect",
        )

    def test_temp_score_overwrite_tcre(self) -> None:
        """
        Test whether the temperature score is calculated as  when overwriting the Transient Climate Response cumulative Emissions (TCRE) control.

        :return:
        """
        exp_trce_mul = 0.0002729257641921397

        overwritten_temp_score = self.temperature_score
        orig_tcre = overwritten_temp_score.c.CONTROLS_CONFIG.tcre
        overwritten_temp_score.c.CONTROLS_CONFIG.tcre = Q_(1.0, ureg.delta_degC)
        scores = overwritten_temp_score.calculate(self.data)
        # We have to put this back as it can screw up other subsequent tests; unittests don't restore mutable default arguments
        overwritten_temp_score.c.CONTROLS_CONFIG.tcre = orig_tcre

        self.assertAlmostEqual(
            scores[(scores["company_name"] == "Company T") & (scores["scope"] == EScope.S1S2)][
                "temperature_score"
            ].iloc[0],
            Q_(1.65, ureg.delta_degC),
            places=2,
            msg="The temp score was incorrect",
        )

        exp_weight = 1.0
        exp_target_overshoot = 699.9763453 / 223.2543241
        exp_trajectory_overshoot = 417.115686 / 223.2543241
        exp_target_score = 1.5 + (exp_weight * 396.0 * (exp_target_overshoot - 1.0) * exp_trce_mul)
        exp_trajectory_score = 1.5 + (exp_weight * 396.0 * (exp_trajectory_overshoot - 1.0) * exp_trce_mul)
        exp_score = exp_target_score * 0.428571429 + exp_trajectory_score * (1 - 0.428571429)
        self.assertAlmostEqual(
            scores[(scores["company_name"] == "Company E") & (scores["scope"] == EScope.S3)]["temperature_score"].iloc[
                0
            ],
            Q_(exp_score, ureg.delta_degC),
            places=2,
            msg="The fallback temp score was incorrect",
        )

        # aggregate S1S2S3 calculation has an extra post-processing step,
        # which overwrites call of call of TemperatureScore.get_score()
        self.assertAlmostEqual(
            scores[
                (scores["company_name"] == "Company AA")
                & (scores["time_frame"] == ETimeFrames.LONG)
                & (scores["scope"] == EScope.S1S2)
            ]["temperature_score"].iloc[0],
            Q_(1.63, ureg.delta_degC),
            places=2,
            msg="The aggregated fallback temp score was incorrect",
        )

    def test_portfolio_aggregations(self):
        scores = self.temperature_score.calculate(self.data)
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(
            aggregations.long.S1S2.all.score,
            Q_(1.85956383, ureg.delta_degC),
            places=2,
            msg="Long WATS aggregation failed",
        )
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.TETS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(
            aggregations.long.S1S2.all.score,
            Q_(1.875, ureg.delta_degC),
            places=2,
            msg="Long TETS aggregation failed",
        )
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.MOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(
            aggregations.long.S1S2.all.score,
            Q_(1.869, ureg.delta_degC),
            places=2,
            msg="Long MOTS aggregation failed",
        )
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.EOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(
            aggregations.long.S1S2.all.score,
            Q_(1.840, ureg.delta_degC),
            places=2,
            msg="Long EOTS aggregation failed",
        )
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.ECOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(
            aggregations.long.S1S2.all.score,
            Q_(1.840, ureg.delta_degC),
            places=2,
            msg="Long ECOTS aggregation failed",
        )
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.AOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(
            aggregations.long.S1S2.all.score,
            Q_(1.869, ureg.delta_degC),
            places=2,
            msg="Long AOTS aggregation failed",
        )

    def test_filter_data(self):
        # This dataframe is not pure pint, but rather a hetergeoous mix of quantified and non-quantified data
        data = asPintDataFrame(
            pd.DataFrame(
                data=[
                    [ETimeFrames.LONG, EScope.S3, Q_(1, ureg.delta_degC)],
                    [ETimeFrames.MID, EScope.S1S2, Q_(2, ureg.delta_degC)],
                    [
                        ETimeFrames.MID,
                        EScope.S3,
                        Q_(3, ureg.delta_degC),
                    ],  # this should stay
                    [ETimeFrames.MID, EScope.S3, None],
                ],
                index=pd.Index(["id0", "id1", "id2", "id3"], name="company_id"),
                columns=["time_frame", "scope", "ghg_s3"],
            )
        )
        expected = asPintDataFrame(
            pd.DataFrame(
                data=[[ETimeFrames.MID, EScope.S3, Q_(3, ureg.delta_degC)]],
                index=pd.Index(["id2"], name="company_id"),
                columns=["time_frame", "scope", "ghg_s3"],
            )
        )
        time_frame = ETimeFrames.MID
        scope = EScope.S3

        filtered_data = data[data[self.temperature_score.c.COLS.SCOPE].eq(scope)]
        if scope == EScope.S3:
            na_s3 = filtered_data[self.temperature_score.c.COLS.GHG_SCOPE3].isna()
            filtered_data = filtered_data[~na_s3]
        filtered_data = filtered_data[filtered_data[self.temperature_score.c.COLS.TIME_FRAME].eq(time_frame)].copy()
        filtered_data[self.temperature_score.grouping] = filtered_data[self.temperature_score.grouping].fillna(
            "unknown"
        )

        for col in filtered_data.columns:
            if isinstance(filtered_data[col], object):
                pd.testing.assert_series_equal(filtered_data[col], expected[col])
            else:
                assert_pint_series_equal(self, filtered_data[col], expected[col])


if __name__ == "__main__":
    test = TestTemperatureScore()
    test.setUp()
    test.test_temp_score()
    test.test_portfolio_aggregations()
