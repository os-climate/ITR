import unittest
import json
import os
import pandas as pd
from numpy.testing import assert_array_equal

import ITR
from ITR.interfaces import EScope, ETimeFrames
from ITR.interfaces import ICompanyData, ICompanyEIProjectionsScopes, ICompanyEIProjections, ICompanyEIProjection
from ITR.interfaces import IProductionBenchmarkScopes, IEIBenchmarkScopes, PortfolioCompany

from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark

from ITR.data.data_warehouse import DataWarehouse
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod

from pint import Quantity
from ITR.data.osc_units import ureg, Q_, PA_, asPintSeries, asPintDataFrame

from utils import gen_company_data, DequantifyQuantity, assert_pint_series_equal

# For this test case, we prime the pump with known-aligned emissions intensities.
# We can then construct companies that have some passing resemplemnce to these, and then verify alignment/non-alignment
# as expected according to how we tweak them company by company.

class TestEIBenchmarks(unittest.TestCase):
    """
    Testdifferent flavours of emission intensity benchmarks
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        # All benchmarks use OECM Production for Production
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        # Each EI benchmark is particular to its own construction
        self.benchmark_EI_OECM_PC = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM_PC.json")
        self.benchmark_EI_OECM_S3 = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM_S3.json")
        self.benchmark_EI_TPI = os.path.join(self.root, "inputs", "json", "benchmark_EI_TPI_2_degrees.json")
        self.benchmark_EI_TPI_below_2 = os.path.join(self.root, "inputs", "json",
                                                     "benchmark_EI_TPI_below_2_degrees.json")
        # OECM Production-Centric (PC)
        with open(self.benchmark_EI_OECM_PC) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.OECM_EI_PC_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        # OECM (S3)
        with open(self.benchmark_EI_OECM_S3) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.OECM_EI_S3_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        # TPI
        with open(self.benchmark_EI_TPI) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.TPI_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        # TPI below 2
        with open(self.benchmark_EI_TPI_below_2) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.TPI_below_2_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        def gen_company_variation(company_name, company_id, region, sector,
                                  base_production,
                                  EI_df, ei_multiplier, ei_offset,
                                  ei_nz_year, ei_max_negative=None) -> ICompanyData:
            year_list = [2019, 2025, 2030, 2035, 2040, 2045, 2050]
            # the last slice(None) gives us all scopes to index against
            bm_ei = asPintDataFrame(EI_df.loc[(sector, region, slice(None)), year_list])

            # We set intensities to be the wonky things
            company_data = gen_company_data(company_name, company_id, region, sector,
                                            base_production,
                                            bm_ei * ei_multiplier + ei_offset,
                                            ei_nz_year, ei_max_negative)
            projected_intensities = company_data.projected_targets
            # We set targets to be the nicely aligned things
            # (which vary due to different sectors/regions/benchmarks)
            company_data = gen_company_data(company_name, company_id, region, sector,
                                            base_production,
                                            bm_ei,
                                            2051, ei_max_negative)
            company_data.projected_intensities = projected_intensities
            return company_data

        # Company AG is over-budget with its intensity projections, but OECM-aligned with their target projections
        company_ag = gen_company_variation('Company AG', 'US0079031078', 'North America', 'Electricity Utilities',
                                           Q_(9.9, "TWh"),
                                           self.OECM_EI_S3_bm._EI_df, 1.0, ei_offset = Q_(100, 'g CO2/kWh'),
                                           ei_nz_year = 2051, ei_max_negative = Q_(-1, 'g CO2/kWh'))

        company_ah = gen_company_variation('Company AH', 'US00724F1012', 'North America', 'Electricity Utilities',
                                           Q_(1.9, "TWh"),
                                           self.OECM_EI_S3_bm._EI_df, 1.5, ei_offset = Q_(0, 'g CO2/kWh'),
                                           ei_nz_year = 2031)

        company_ai = gen_company_variation('Company AI', 'FR0000125338', 'Europe', 'Electricity Utilities',
                                           Q_(4.9, "PJ"),
                                           self.OECM_EI_S3_bm._EI_df * 0.8, 1.0, ei_offset = Q_(0, 't CO2/MWh'),
                                           ei_nz_year = 2051)

        # print(json.dumps(company_ag.dict(), cls=DequantifyQuantity, indent=2))

        # load company data
        self.companies = [company_ag, company_ah, company_ai]
        self.base_company_data = BaseCompanyDataProvider(self.companies)

        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        self.OECM_S3_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.OECM_EI_S3_bm)
        self.TPI_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.TPI_EI_bm)
        self.TPI_below_2_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm,
                                                   self.TPI_below_2_EI_bm)

        self.company_ids = ["US0079031078",
                            "US00724F1012",
                            "FR0000125338"]

    def test_all_benchmarks(self):
        # Calculate Temp Scores
        oecm_PC_temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        oecm_S3_temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2S3],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        tpi_temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio = []
        for company in self.company_ids:
            portfolio.append(PortfolioCompany(
                company_name=company,
                company_id=company,
                investment_value=100,
                company_isin=company,
            )
            )
        # OECM S3
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.OECM_S3_warehouse, portfolio)
        scores = oecm_S3_temp_score.calculate(portfolio_data)
        agg_scores = oecm_S3_temp_score.aggregate_scores(scores)

        print(scores[['company_name','company_id', 'temperature_score', 'trajectory_score', 'trajectory_overshoot_ratio', 'target_score', 'target_overshoot_ratio']])

        # verify company scores:
        expected = pd.Series([1.44, 1.55, 1.86], dtype='pint[delta_degC]')
        assert_pint_series_equal(self, scores.temperature_score.values, expected, places=2)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2S3.all.score, Q_(1.62, ureg.delta_degC), places=2)

        # TPI
        # portfolio data
        breakpoint()
        portfolio_data = ITR.utils.get_data(self.TPI_warehouse, portfolio)
        scores = tpi_temp_score.calculate(portfolio_data)
        agg_scores = tpi_temp_score.aggregate_scores(scores)

        print(scores[['company_name','company_id', 'temperature_score', 'trajectory_score', 'trajectory_overshoot_ratio', 'target_score', 'target_overshoot_ratio']])

        # verify company scores:
        expected = pd.Series([1.64, 1.41, 1.31], dtype='pint[delta_degC]')
        assert_pint_series_equal(self, scores.temperature_score.values, expected, places=2)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1.all.score, Q_(1.45, ureg.delta_degC), places=2)

        # TPI below 2
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.TPI_below_2_warehouse, portfolio)
        scores = tpi_temp_score.calculate(portfolio_data)
        agg_scores = tpi_temp_score.aggregate_scores(scores)

        print(scores[['company_name','company_id', 'temperature_score', 'trajectory_score', 'trajectory_overshoot_ratio', 'target_score', 'target_overshoot_ratio']])

        # verify company scores:
        expected = pd.Series([1.52, 1.37, 1.58], dtype='pint[delta_degC]')
        assert_pint_series_equal(self, scores.temperature_score.values, expected, places=2)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1.all.score, Q_(1.49, ureg.delta_degC), places=2)

        # OECM PC -- This overwrites company data (which it should not)
        self.OECM_PC_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.OECM_EI_PC_bm)
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.OECM_PC_warehouse, portfolio)
        scores = oecm_PC_temp_score.calculate(portfolio_data)
        agg_scores = oecm_PC_temp_score.aggregate_scores(scores)

        print(scores[['company_name','company_id', 'temperature_score', 'trajectory_score', 'trajectory_overshoot_ratio', 'target_score', 'target_overshoot_ratio']])

        # verify company scores:
        expected = pd.Series([1.87, 1.55, 1.45], dtype='pint[delta_degC]')
        assert_pint_series_equal(self, scores.temperature_score.values, expected, places=2)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(1.63, ureg.delta_degC), places=2)

if __name__ == "__main__":
    test = TestEIBenchmarks()
    test.setUp()
    test.test_all_benchmarks()
