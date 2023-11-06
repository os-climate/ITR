import unittest

import numpy as np
import pandas as pd
from utils import assert_pint_series_equal

import ITR
from ITR.configs import ColumnsConfig
from ITR.interfaces import EScope
from ITR.portfolio_aggregation import PortfolioAggregation, PortfolioAggregationMethod


class TestPortfolioAggregation(unittest.TestCase):
    """
    Test the interfaces.
    """

    def setUp(self) -> None:
        """ """
        self.data = pd.DataFrame()
        self.data.loc[:, ColumnsConfig.COMPANY_NAME] = [
            "Company A",
            "Company B",
            "Company C",
        ]
        self.data.loc[:, ColumnsConfig.COMPANY_REVENUE] = pd.Series([1.0, 2.0, 3.0], dtype="pint[USD]")
        self.data.loc[:, ColumnsConfig.COMPANY_MARKET_CAP] = pd.Series([1.0, 2.0, 3.0], dtype="pint[USD]")
        self.data.loc[:, ColumnsConfig.INVESTMENT_VALUE] = pd.Series([1.0, 2.0, 3.0], dtype="pint[USD]")
        self.data.loc[:, ColumnsConfig.SCOPE] = [
            EScope.S1S2,
            EScope.S1S2,
            EScope.S1S2S3,
        ]
        self.data.loc[:, ColumnsConfig.GHG_SCOPE12] = pd.Series([1.0, 2.0, 3.0], dtype="pint[t CO2]")
        self.data.loc[:, ColumnsConfig.GHG_SCOPE3] = pd.Series([1.0, 2.0, 3.0], dtype="pint[t CO2]")
        self.data.loc[:, ColumnsConfig.COMPANY_ENTERPRISE_VALUE] = pd.Series([1.0, 2.0, 3.0], dtype="pint[USD]")
        self.data.loc[:, ColumnsConfig.COMPANY_CASH_EQUIVALENTS] = pd.Series([1.0, 2.0, 3.0], dtype="pint[USD]")
        self.data.loc[:, ColumnsConfig.COMPANY_EV_PLUS_CASH] = pd.Series([1.0, 2.0, 3.0], dtype="pint[USD]")
        self.data.loc[:, ColumnsConfig.COMPANY_TOTAL_ASSETS] = pd.Series([1.0, 2.0, 3.0], dtype="pint[USD]")
        self.data.loc[:, ColumnsConfig.TEMPERATURE_SCORE] = pd.Series([1.0, 2.0, 3.0], dtype="pint[delta_degC]")

    def test_is_emissions_based(self):
        self.assertTrue(PortfolioAggregationMethod.is_emissions_based(PortfolioAggregationMethod.MOTS))
        self.assertTrue(PortfolioAggregationMethod.is_emissions_based(PortfolioAggregationMethod.EOTS))
        self.assertTrue(PortfolioAggregationMethod.is_emissions_based(PortfolioAggregationMethod.ECOTS))
        self.assertTrue(PortfolioAggregationMethod.is_emissions_based(PortfolioAggregationMethod.AOTS))
        self.assertTrue(PortfolioAggregationMethod.is_emissions_based(PortfolioAggregationMethod.ROTS))

        self.assertFalse(PortfolioAggregationMethod.is_emissions_based(PortfolioAggregationMethod.WATS))
        self.assertFalse(PortfolioAggregationMethod.is_emissions_based(PortfolioAggregationMethod.TETS))

    def test_get_value_column(self):
        self.assertEqual(
            PortfolioAggregationMethod.get_value_column(PortfolioAggregationMethod.MOTS, ColumnsConfig),
            ColumnsConfig.COMPANY_MARKET_CAP,
        )
        self.assertEqual(
            PortfolioAggregationMethod.get_value_column(PortfolioAggregationMethod.EOTS, ColumnsConfig),
            ColumnsConfig.COMPANY_ENTERPRISE_VALUE,
        )
        self.assertEqual(
            PortfolioAggregationMethod.get_value_column(PortfolioAggregationMethod.ECOTS, ColumnsConfig),
            ColumnsConfig.COMPANY_EV_PLUS_CASH,
        )
        self.assertEqual(
            PortfolioAggregationMethod.get_value_column(PortfolioAggregationMethod.AOTS, ColumnsConfig),
            ColumnsConfig.COMPANY_TOTAL_ASSETS,
        )
        self.assertEqual(
            PortfolioAggregationMethod.get_value_column(PortfolioAggregationMethod.ROTS, ColumnsConfig),
            ColumnsConfig.COMPANY_REVENUE,
        )
        self.assertEqual(
            PortfolioAggregationMethod.get_value_column(PortfolioAggregationMethod.WATS, ColumnsConfig),
            ColumnsConfig.COMPANY_MARKET_CAP,
        )
        self.assertEqual(
            PortfolioAggregationMethod.get_value_column(PortfolioAggregationMethod.TETS, ColumnsConfig),
            ColumnsConfig.COMPANY_MARKET_CAP,
        )

    def test_check_column(self):
        PortfolioAggregation()._check_column(data=self.data, column=ColumnsConfig.COMPANY_REVENUE)

        self.data.loc[0, ColumnsConfig.TEMPERATURE_SCORE] = np.nan
        # _check_column no longer raises an exceptiong for null or missing data, because
        # FOR AGGREGATION PURPOSES, such missing data is treated as zero (just like np.sum
        # treats missing data as zero, whereas np.add will create np.nan if either
        # addend is np.nan).  We no longer call _check_column for non-aggregating operations.
        return
        # with self.assertRaises(ValueError):
        #     PortfolioAggregation()._check_column(data=self.data, column=ColumnsConfig.TEMPERATURE_SCORE)

    def test_calculate_aggregate_score_WATS(self):
        pa_WATS = PortfolioAggregation()._calculate_aggregate_score(
            data=self.data,
            input_column=ColumnsConfig.TEMPERATURE_SCORE,
            portfolio_aggregation_method=PortfolioAggregationMethod.WATS,
        )
        expected = pd.Series([0.1666667, 0.6666667, 1.5], dtype="pint[delta_degC]")
        assert_pint_series_equal(self, pa_WATS, expected)

    def test_calculate_aggregate_score_TETS(self):
        pa_TETS = PortfolioAggregation()._calculate_aggregate_score(
            data=self.data,
            input_column=ColumnsConfig.TEMPERATURE_SCORE,
            portfolio_aggregation_method=PortfolioAggregationMethod.TETS,
        )
        expected = pd.Series([0.1111111, 0.4444444, 2.0], dtype="pint[delta_degC]")
        assert_pint_series_equal(self, pa_TETS, expected)

    def test_calculate_aggregate_score_ECOTS(self):
        pa_ECOTS = PortfolioAggregation()._calculate_aggregate_score(
            data=self.data,
            input_column=ColumnsConfig.TEMPERATURE_SCORE,
            portfolio_aggregation_method=PortfolioAggregationMethod.ECOTS,
        )
        expected = pd.Series([0.1111111, 0.4444444, 2.0], dtype="pint[delta_degC]")
        assert_pint_series_equal(self, pa_ECOTS, expected)

    def test_calculate_aggregate_score_MOTS(self):
        pa_MOTS = PortfolioAggregation()._calculate_aggregate_score(
            data=self.data,
            input_column=ColumnsConfig.TEMPERATURE_SCORE,
            portfolio_aggregation_method=PortfolioAggregationMethod.MOTS,
        )
        expected = pd.Series([0.1111111, 0.4444444, 2.0], dtype="pint[delta_degC]")
        assert_pint_series_equal(self, pa_MOTS, expected)

    def test_calculate_aggregate_score_EOTS(self):
        pa_EOTS = PortfolioAggregation()._calculate_aggregate_score(
            data=self.data,
            input_column=ColumnsConfig.TEMPERATURE_SCORE,
            portfolio_aggregation_method=PortfolioAggregationMethod.EOTS,
        )
        expected = pd.Series([0.1111111, 0.4444444, 2.0], dtype="pint[delta_degC]")
        assert_pint_series_equal(self, pa_EOTS, expected)

    def test_calculate_aggregate_score_AOTS(self):
        pa_AOTS = PortfolioAggregation()._calculate_aggregate_score(
            data=self.data,
            input_column=ColumnsConfig.TEMPERATURE_SCORE,
            portfolio_aggregation_method=PortfolioAggregationMethod.AOTS,
        )
        expected = pd.Series([0.11111111, 0.44444444, 2.0], dtype="pint[delta_degC]")
        assert_pint_series_equal(self, pa_AOTS, expected)

    def test_calculate_aggregate_score_ROTS(self):
        pa_ROTS = PortfolioAggregation()._calculate_aggregate_score(
            data=self.data,
            input_column=ColumnsConfig.TEMPERATURE_SCORE,
            portfolio_aggregation_method=PortfolioAggregationMethod.ROTS,
        )
        expected = pd.Series([0.1111111, 0.4444444, 2.0], dtype="pint[delta_degC]")
        assert_pint_series_equal(self, pa_ROTS, expected)


if __name__ == "__main__":
    test = TestPortfolioAggregation()
    test.setUp()
    test.test_calculate_aggregate_score_WATS()
