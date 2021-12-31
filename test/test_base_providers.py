import json
import unittest
import os
import pandas as pd
from numpy.testing import assert_array_equal
import ITR

from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark
from ITR.interfaces import ICompanyData, EScope, ETimeFrames, PortfolioCompany, IEmissionIntensityBenchmarkScopes, \
    IProductionBenchmarkScopes, IYOYBenchmarkScopes

from ITR.data.osc_units import ureg, Q_, PA_

class TestBaseProvider(unittest.TestCase):
    """
    Test the Base provider
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_json = os.path.join(self.root, "inputs", "json", "fundamental_data.json")
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        self.benchmark_EI_json = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM.json")
        self.excel_data_path = os.path.join(self.root, "inputs", "test_data_company.xlsx")

        # load company data
        with open(self.company_json) as json_file:
            parsed_json = json.load(json_file)
        self.companies = [ICompanyData.parse_obj(company_data) for company_data in parsed_json]
        self.base_company_data = BaseCompanyDataProvider(self.companies)

        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IYOYBenchmarkScopes.parse_obj(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        # load intensity benchmarks
        with open(self.benchmark_EI_json) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEmissionIntensityBenchmarkScopes.parse_obj(parsed_json)
        self.base_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        self.base_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.base_EI_bm)
        self.company_ids = ["US0079031078",
                            "US00724F1012",
                            "FR0000125338"]
        self.company_info_at_base_year = pd.DataFrame(
            [[Q_(1.6982474347547, 't CO2/MWh'), Q_(1.04827859e+08, 'MWh'), 'Electricity Utilities', 'North America'],
             [Q_(0.476586931582279, 't CO2/MWh'), Q_(5.98937002e+08, 'MWh'), 'Electricity Utilities', 'North America'],
             [Q_(0.22457393169277, 't CO2/MWh'), Q_(1.22472003e+08, 'MWh'), 'Electricity Utilities', 'Europe']],
            index=self.company_ids,
            columns=[ColumnsConfig.BASE_EI, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR, ColumnsConfig.REGION])

    def test_temp_score_from_excel_data(self):
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
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.base_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify company scores:
        expected = [2.05, 2.22, 2.06]
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, 2.11, places=2)


    def test_get_benchmark(self):
        seq_index = pd.RangeIndex.from_range(range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))
        data = [pd.Series([1.698247435, 1.581691084, 1.386040647, 1.190390211, 0.994739774, 0.799089338,
                           0.782935186, 0.677935928, 0.572936671, 0.467937413, 0.362938156, 0.257938898,
                           0.233746281, 0.209553665, 0.185361048, 0.161168432, 0.136975815, 0.124810886,
                           0.112645956, 0.100481026, 0.088316097, 0.076151167, 0.062125588, 0.048100009,
                           0.034074431, 0.020048852, 0.006023273, 0.005843878, 0.005664482, 0.005485087,
                           0.005305691, 0.005126296
                           ], index=seq_index, dtype="pint[t CO2/MWh]"),
                pd.Series([0.476586932, 0.444131055, 0.389650913, 0.335170772, 0.28069063, 0.226210489,
                           0.22171226, 0.192474531, 0.163236802, 0.133999073, 0.104761344, 0.075523615,
                           0.068787023, 0.062050431, 0.055313839, 0.048577247, 0.041840655, 0.038453251,
                           0.035065847, 0.031678443, 0.028291039, 0.024903635, 0.020998121, 0.017092607,
                           0.013187093, 0.009281579, 0.005376065, 0.005326111, 0.005276157, 0.005226203,
                           0.005176249, 0.005126296
                           ], index=seq_index, dtype="pint[t CO2/MWh]"),
                pd.Series([0.224573932, 0.17975612, 0.163761501, 0.147766883, 0.131772265, 0.115777646,
                           0.099783028, 0.090628361, 0.081473693, 0.072319026, 0.063164359, 0.054009692,
                           0.050089853, 0.046170015, 0.042250176, 0.038330338, 0.034410499, 0.031104249,
                           0.027797999, 0.024491748, 0.021185498, 0.017879248, 0.016155615, 0.014431983,
                           0.012708351, 0.010984719, 0.009261087, 0.008488943, 0.007716798, 0.006944654,
                           0.00617251, 0.005400365
                           ], index=seq_index, dtype="pint[t CO2/MWh]")]
        expected_data = pd.concat(data, axis=1, ignore_index=True).T
        expected_data.index=self.company_ids


        pd.testing.assert_frame_equal(
            self.base_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year),
            expected_data)

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series([1.06866370e+08, 6.10584093e+08, 1.28474171e+08],
                                       index=self.company_ids,
                                       name=2025,
                                       dtype='pint[MWh]')
        print(self.base_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025]['US0079031078'])
        print(expected_data_2025['US0079031078'])
        pd.testing.assert_series_equal(
            self.base_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025],
            expected_data_2025, check_dtype=False)

    def test_get_cumulative_value(self):
        projected_ei = pd.DataFrame([[Q_(1.0, 't CO2/MWh'), Q_(2.0, 't CO2/MWh')], [Q_(3.0, 't CO2/MWh'), Q_(4.0, 't CO2/MWh')]], dtype='pint[t CO2/MWh]')
        projected_production = pd.DataFrame([[Q_(2.0, 'TWh'), Q_(4.0, 'TWh')], [Q_(6.0, 'TWh'), Q_(8.0, 'TWh')]], dtype='pint[TWh]')
        expected_data = pd.Series([10.0, 50.0],
                                    index=[0, 1],
                                    dtype='pint[Mt CO2]')
        print(self.base_warehouse._get_cumulative_emission(projected_emission_intensity=projected_ei,
                                                           projected_production=projected_production))
        print(f"expected_data = {expected_data}")
        pd.testing.assert_series_equal(
            self.base_warehouse._get_cumulative_emission(projected_emission_intensity=projected_ei,
                                                         projected_production=projected_production), expected_data)

    def test_get_company_data(self):
        company_1 = self.base_warehouse.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.base_warehouse.get_preprocessed_company_data(self.company_ids)[1]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AH")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US00724F1012")
        self.assertAlmostEqual(company_1.ghg_s1s2, Q_(104827858.636039, 'MWh'))    # These are apparently production numbers, not emissions numbers
        self.assertAlmostEqual(company_2.ghg_s1s2, Q_(598937001.892059, 'MWh'))    # These are apparently production numbers, not emissions numbers
        self.assertAlmostEqual(company_1.cumulative_budget, Q_(1362284467.0830, 't CO2'), places=4)
        self.assertAlmostEqual(company_2.cumulative_budget, Q_(2262242040.68059, 't CO2'), places=4)
        self.assertAlmostEqual(company_1.cumulative_target, Q_(3769096510.09909, 't CO2'), places=4)
        self.assertAlmostEqual(company_2.cumulative_target, Q_(5912426347.23670, 't CO2'), places=4)
        self.assertAlmostEqual(company_1.cumulative_trajectory, Q_(3745094638.52858, 't CO2'), places=4)
        self.assertAlmostEqual(company_2.cumulative_trajectory, Q_(8631481789.38558, 't CO2'), places=4)

    def test_get_value(self):
        expected_data = pd.Series([20248547997.0,
                                   276185899.0,
                                   10283015132.0],
                                  index=pd.Index(self.company_ids, name='company_id'),
                                  name='company_revenue')
        pd.testing.assert_series_equal(self.base_company_data.get_value(company_ids=self.company_ids,
                                                                        variable_name=ColumnsConfig.COMPANY_REVENUE),
                                       expected_data)
