import os
import unittest

import pandas as pd

from ITR.configs import PortfolioAggregationConfig
from ITR.target_validation import TargetProtocol
from ITR.portfolio_aggregation import PortfolioAggregationMethod, PortfolioAggregation
from ITR.interfaces import IDataProviderTarget, IDataProviderCompany, EScope


class TestTargetValidation(unittest.TestCase):
    """
    Test the interfaces.
    """

    def setUp(self) -> None:
        """
        """
        self.target_protocol = TargetProtocol(config=PortfolioAggregationConfig)

    def test_process(self):
        target = [IDataProviderTarget(company_id="123",
                                      target_type="S3",
                                      scope=EScope.S3,
                                      coverage_s1=0.0,
                                      coverage_s2=0.0,
                                      coverage_s3=0.0,
                                      reduction_ambition=0.5,
                                      base_year=2020,
                                      base_year_ghg_s1=2030,
                                      base_year_ghg_s2=2030,
                                      base_year_ghg_s3=2030,
                                      start_year=2020,
                                      end_year=2030)]

        company = [IDataProviderCompany(company_name="Company A",
                                        company_id="123",
                                        isic="US0000000001",
                                        ghg_s1s2=10000,
                                        ghg_s3=10000,
                                        cumulative_budget=10000,
                                        cumulative_trajectory=10000,
                                        cumulative_target=10000,
                                        target_probability=0.5)]

        df = self.target_protocol.process(targets=target, companies=company)
        self.assertEqual(self.target_protocol.target_data.loc[:, "scope"].iloc[0], EScope.S3)
        self.assertEqual(self.target_protocol.target_data.loc[:, "base_year"].iloc[0], 2020)
        self.assertEqual(self.target_protocol.target_data.loc[:, "reduction_ambition"].iloc[0], 0.5)
        self.assertEqual(self.target_protocol.target_data.shape, (1, 16))





