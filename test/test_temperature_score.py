import os
import unittest

import pandas as pd

from ITR.configs import ColumnsConfig
from ITR.interfaces import ETimeFrames, EScope
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod


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
        self.data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "inputs",
                                             "data_test_temperature_score.csv"), sep=";")

    def test_temp_score(self) -> None:
        """
        Test whether the temperature score is calculated as expected.

        :return:
        """
        scores = self.temperature_score.calculate(self.data)
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company T") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], 1.82, places=2, msg="The temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company E") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], 1.84, places=2,
                               msg="The fallback temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company AA") &
                                   (scores["time_frame"] == ETimeFrames.LONG) &
                                   (scores["scope"] == EScope.S1S2S3)
                                   ]["temperature_score"].iloc[0], 1.79, places=5,
                               msg="The aggregated fallback temp score was incorrect")

    def test_temp_score_overwrite_tcre(self) -> None:
        """
        Test whether the temperature score is calculated as  when overwriting the Transient Climate Responsie cumulative Emissions (TCRE) control.

        :return:
        """
        overwritten_temp_score = self.temperature_score
        overwritten_temp_score.c.CONTROLS_CONFIG.tcre = 1.0
        scores = overwritten_temp_score.calculate(self.data)
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company T") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], 1.65, places=2, msg="The temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company E") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], 1.65, places=2,
                               msg="The fallback temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company AA") &
                                   (scores["time_frame"] == ETimeFrames.LONG) &
                                   (scores["scope"] == EScope.S1S2S3)
                                   ]["temperature_score"].iloc[0], 1.63, places=5,
                               msg="The aggregated fallback temp score was incorrect")

    def test_portfolio_aggregations(self):
        scores = self.temperature_score.calculate(self.data)
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, 1.857, places=2,
                               msg="Long WATS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.TETS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, 1.875, places=2,
                               msg="Long TETS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.MOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, 1.869, places=2,
                               msg="Long MOTS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.EOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, 1.840, places=2,
                               msg="Long EOTS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.ECOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, 1.840, places=2,
                               msg="Long ECOTS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.AOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, 1.869, places=2,
                               msg="Long AOTS aggregation failed")


if __name__ == "__main__":
    test = TestTemperatureScore()
    test.setUp()
    test.test_temp_score()
    test.test_portfolio_aggregations()
