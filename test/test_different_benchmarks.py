import json
import unittest
import os
import pandas as pd
from numpy.testing import assert_array_equal
import ITR

from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark
from ITR.interfaces import ICompanyData, EScope, ETimeFrames, PortfolioCompany, IEIBenchmarkScopes, \
    IProductionBenchmarkScopes

from ITR.data.osc_units import ureg, Q_, PA_

class TestEIBenchmarks(unittest.TestCase):
    """
    Testdifferent flavours of emission intensity benchmarks
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_json = os.path.join(self.root, "inputs", "json", "fundamental_data.json")
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        self.benchmark_EI_OECM = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM.json")
        self.benchmark_EI_TPI = os.path.join(self.root, "inputs", "json", "benchmark_EI_TPI_2_degrees.json")
        self.benchmark_EI_TPI_below_2 = os.path.join(self.root, "inputs", "json",
                                                     "benchmark_EI_TPI_below_2_degrees.json")

        # load company data
        with open(self.company_json) as json_file:
            parsed_json = json.load(json_file)
        for company_data in parsed_json:
            company_data['emissions_metric'] = {'units':'t CO2'}
            if company_data['sector'] == 'Electricity Utilities':
                company_data['production_metric'] = {'units':'MWh'}
            elif company_data['sector'] == 'Steel':
                company_data['production_metric'] = {'units':'Fe_ton'}
        self.companies = [ICompanyData.parse_obj(company_data) for company_data in parsed_json]
        self.base_company_data = BaseCompanyDataProvider(self.companies)

        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        # load intensity benchmarks

        # OECM
        with open(self.benchmark_EI_OECM) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.OECM_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

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

        self.OECM_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.OECM_EI_bm)
        self.TPI_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.TPI_EI_bm)
        self.TPI_below_2_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm,
                                                   self.TPI_below_2_EI_bm)

        self.company_ids = ["US0079031078",
                            "US00724F1012",
                            "FR0000125338"]

    def test_all_benchmarks(self):
        # Calculate Temp Scores
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
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
        # OECM
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.OECM_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify company scores:
        expected = pd.Series([2.05, 2.22, 2.06], dtype='pint[delta_degC]')
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(2.11, ureg.delta_degC), places=2)

        # TPI
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.TPI_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify company scores:
        expected = pd.Series([2.35, 2.39, 2.22], dtype='pint[delta_degC]')
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(2.32, ureg.delta_degC), places=2)

        # TPI below 2
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.TPI_below_2_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify company scores:
        expected = pd.Series([2.11, 2.32, 2.35], dtype='pint[delta_degC]')
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(2.26, ureg.delta_degC), places=2)

if __name__ == "__main__":
    test = TestEIBenchmarks()
    test.setUp()
    test.test_all_benchmarks()
