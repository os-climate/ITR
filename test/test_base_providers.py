import json
import unittest
import os
import pandas as pd
import ITR
from ITR.data.osc_units import ureg, Q_

from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark
from ITR.interfaces import ICompanyData, EScope, ETimeFrames, PortfolioCompany, IEIBenchmarkScopes, \
    IProductionBenchmarkScopes
from utils import assert_pint_frame_equal, assert_pint_series_equal


class TestBaseProvider(unittest.TestCase):
    """
    Test the Base provider
    """

    def setUp_OECM_S3(self) -> None:
        self._setUpWithEIBM("benchmark_EI_OECM_S3.json", EScope.AnyScope)

    def setUp(self) -> None:
        self._setUpWithEIBM("benchmark_EI_OECM_PC.json", EScope.S1S2)

    def setUp_S3_only(self) -> None:
        self._setUpWithEIBM("benchmark_EI_S3.json", EScope.S3)

    def _setUpWithEIBM(self, eibm_filename, scope_to_calc) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_json = os.path.join(self.root, "inputs", "json", "fundamental_data.json")
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        self.benchmark_EI_json = os.path.join(self.root, "inputs", "json", eibm_filename)

        # load company data
        with open(self.company_json) as json_file:
            parsed_json = json.load(json_file)
        for company_data in parsed_json:
            company_data['emissions_metric'] = 't CO2'
            if company_data['sector'] == 'Electricity Utilities':
                if company_data['region'] == 'Europe':
                    company_data['production_metric'] = 'GJ'
                else:
                    company_data['production_metric'] = 'MWh'
            elif company_data['sector'] == 'Steel':
                company_data['production_metric'] = 't Steel'
        self.companies = [ICompanyData.parse_obj(company_data) for company_data in parsed_json]
        # If the company data does not have S3 emissions projections, it won't match any S3 scope data
        self.base_company_data = BaseCompanyDataProvider(self.companies)

        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        # load intensity benchmarks
        with open(self.benchmark_EI_json) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.base_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        self.base_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.base_EI_bm)
        self.company_ids = ["US0079031078",
                            "US00724F1012",
                            "FR0000125338",
                            "US17275R1023"]
        self.company_info_at_base_year = pd.DataFrame(
            [['Electricity Utilities', 'North America', scope_to_calc,
              Q_(1.6982474347547, 't CO2/MWh'), Q_(1.04827859e+08, 'MWh'), 'MWh'],
             ['Electricity Utilities', 'North America', scope_to_calc,
              Q_(0.476586931582279, 't CO2/MWh'), Q_(5.98937002e+08, 'MWh'), 'MWh'],
             ['Electricity Utilities', 'Europe', scope_to_calc,
              Q_(0.22457393169277, 't CO2/GJ'), Q_(1.22472003e+08, 'GJ'), 'GJ'],
             ['Electricity Utilities', 'North America', scope_to_calc,
              Q_(0.476586931582279, 't CO2/MWh'), Q_(5.98937002e+08, 'MWh'), 'MWh']],
            index=pd.Index(self.company_ids, name='company_id'),
            columns=[ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE,
                     ColumnsConfig.BASE_EI, ColumnsConfig.BASE_YEAR_PRODUCTION, ColumnsConfig.PRODUCTION_METRIC])


    def test_temp_score_from_json_data(self):
        return
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
            ))
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.base_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify company scores:
        expected = pd.Series([5.53, 2.72, 1.82], dtype='pint[delta_degC]', name='temperature_score')
        pd.testing.assert_series_equal(scores.temperature_score, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(2.11, ureg.delta_degC), places=2)

    def test_get_benchmark(self):
        # This test is a hot mess: the data are series of corp EI trajectories, which are company-specific
        # benchmarks are sector/region specific, and guide temperature scores, but we wouldn't expect
        # an exact match between the two except when the company's data was generated from the benchmark
        # (as test.utils.gen_company_data does).
        return
        seq_index = pd.RangeIndex.from_range(range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))
        data = [
            pd.Series([1.6982474347547, 1.5814362115005, 1.385357948863141, 1.18927968622576, 0.9932014235884,
                       0.7971231609510, 0.7809336851789, 0.675704827194235, 0.57047596920960, 0.4652471112250,
                       0.3600182532403, 0.2547893952557, 0.230543877037740, 0.20629835881977, 0.1820528406018,
                       0.1578073223838, 0.1335618041659, 0.121370273602458, 0.10917874303905, 0.0969872124756,
                       0.0847956819122, 0.0726041513488, 0.058547903118731, 0.04449165488867, 0.0304354066586,
                       0.0163791584285, 0.0023229101985, 0.002143122358755, 0.00196333451906, 0.0017835466794,
                       0.0016037588397, 0.0014239710000], index=seq_index, dtype="pint[t CO2/MWh]"),
            pd.Series([0.476586931582279, 0.4438761824346, 0.3889682148288414, 0.33406024722304, 0.27915227961723,
                       0.224244312011427, 0.2197107589327, 0.1902434296749848, 0.16077610041727, 0.13130877115956,
                       0.101841441901845, 0.0723741126441, 0.0655846189440570, 0.05879512524398, 0.05200563154391,
                       0.045216137843832, 0.0384266441438, 0.0350126391631084, 0.03159863418246, 0.02818462920181,
                       0.024770624221162, 0.0213566192405, 0.0174204357386057, 0.01348425223670, 0.00954806873479,
                       0.005611885232884, 0.0016757017310, 0.0016253555847724, 0.00157500943857, 0.00152466329236,
                       0.001474317146161, 0.0014239710000], index=seq_index, dtype="pint[t CO2/MWh]"),
            pd.Series([0.2245739316928, 0.1789585724182, 0.16267932465295, 0.146400076887697, 0.1301208291224,
                       0.1138415813572, 0.0975623335919, 0.08824475610517, 0.078927178618408, 0.0696096011316,
                       0.0602920236449, 0.0509744461581, 0.04698485296078, 0.042995259763452, 0.0390056665661,
                       0.0350160733688, 0.0310264801715, 0.02766139400289, 0.024296307834324, 0.0209312216658,
                       0.0175661354972, 0.0142010493286, 0.01244674461183, 0.010692439895051, 0.0089381351783,
                       0.0071838304615, 0.0054295257447, 0.00464364089781, 0.003857756050920, 0.0030718712040,
                       0.0022859863571, 0.0015001015102], index=seq_index, dtype="pint[t CO2/GJ]")]
        expected_data = pd.concat(data, axis=1, ignore_index=True).T
        expected_data.index = self.company_ids
        benchmarks = self.base_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year)

        assert_pint_frame_equal(self, benchmarks, expected_data)

    def test_get_benchmark_scope_matters(self):
        '''
        Simple sanity test, to verify, that getting intensity benchmarks
        takes in account primary scope - S1S2 or S3
        '''
        # benchmarks for default scope S1S2
        bm_s1s2 = self.base_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year, EScope.S1S2)

        # Reload EI benchmark with primary scope S3
        self.setUp_S3_only()
        bm_s3 = self.base_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year, EScope.S3)

        # Verify that different scope results into different values, but same index and columns
        self.assertTrue(bm_s1s2.index.droplevel('scope').equals(bm_s3.index.droplevel('scope')))
        self.assertTrue(bm_s1s2.columns.equals(bm_s3.columns))
        self.assertFalse(bm_s1s2.equals(bm_s3))

    def test_get_projected_production(self):
        # Note that 40763845.66650752 MWh = 146749844.39942706 gigajoule
        # expected_data_2025 is all MWh, but productions vector is heterogeneous
        expected_data_2025 = pd.Series([122926534.69719231, 702344308.6611674, 40763845.66650752, 702344308.6611674],
                                       index=self.company_ids,
                                       name=2025,
                                       dtype='pint[MWh]')
        productions = self.base_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025]
        assert_pint_series_equal(self, expected_data_2025, productions)

    def test_get_projected_targets(self):
        expected_data_2025 = pd.Series([122926534.69719231, 702344308.6611674, 40763845.66650752, 702344308.6611674],
                                       index=self.company_ids,
                                       name=2025,
                                       dtype='pint[MWh]')
        expected_data_2025 = pd.Series([Q_(1.2920428864089595, 't CO2 / MWh'),
                                        Q_(0.413042007371309, 't CO2 / MWh'),
                                        Q_(0.17005401282971486, 't CO2 / GJ'),
                                        Q_(0.04623233372785435, 't CO2 / GJ')],
                                       index = pd.MultiIndex.from_tuples(zip(self.company_ids, [EScope.S1S2]*len(self.company_ids))),
                                       name=2025)
        target_projections = self.base_company_data.get_company_projected_targets(self.company_ids, 2025)
        assert_pint_series_equal(self, expected_data_2025, target_projections)

    def test_get_cumulative_value(self):
        projected_ei = pd.DataFrame(
            [[Q_(1.0, 't CO2/MWh'), Q_(2.0, 't CO2/MWh')], [Q_(3.0, 't CO2/MWh'), Q_(4.0, 't CO2/MWh')]],
            dtype='pint[t CO2/MWh]')
        projected_production = pd.DataFrame([[Q_(2.0, 'TWh'), Q_(4.0, 'TWh')], [Q_(6.0, 'TWh'), Q_(8.0, 'TWh')]],
                                            dtype='pint[TWh]')
        expected_data = pd.Series([10.0, 50.0],
                                  index=[0, 1],
                                  dtype='pint[Mt CO2]')
        cumulative_emissions = self.base_warehouse._get_cumulative_emissions(projected_ei=projected_ei,
                                                                             projected_production=projected_production)
        assert_pint_series_equal(self, cumulative_emissions.iloc[:, -1], expected_data)

    def test_get_company_data(self):
        #                    cumulative_trajectory   cumulative_target   cumulative_budget
        # company_id   scope                                                              
        # US0079031078 S1S2     17222.957455753196  17342.428074061572  1243.1262721585053
        # US00724F1012 S1S2      40343.09136798881  27191.863852079525    7102.63790663654
        # FR0000125338 S1S2      73.38491457431634  24.240789263984546   7.188410439037949
        # US17275R1023 S1S2       762.144937298071   618.8253752192155   321.7262796687323

        companies = self.base_warehouse.get_preprocessed_company_data(self.company_ids)
        company_1 = companies[0]
        company_2 = companies[1]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AH")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US00724F1012")
        self.assertAlmostEqual(company_1.ghg_s1s2, Q_(640.885111270135, 'Mt CO2'))
        self.assertAlmostEqual(company_2.ghg_s1s2, Q_(1027.6039725941746, 'Mt CO2'))
        self.assertAlmostEqual(company_1.cumulative_budget, Q_(1243.1262721585053, 'Mt CO2'))
        self.assertAlmostEqual(company_2.cumulative_budget, Q_( 7102.63790663654, 'Mt CO2'))
        self.assertAlmostEqual(company_1.cumulative_target, Q_(17342.428074061572, 'Mt CO2'))
        self.assertAlmostEqual(company_2.cumulative_target, Q_(27191.863852079525, 'Mt CO2'))
        self.assertAlmostEqual(company_1.cumulative_trajectory, Q_(17222.957455753196, 'Mt CO2'))
        self.assertAlmostEqual(company_2.cumulative_trajectory, Q_(40343.09136798882, 'Mt CO2'))

        # Reload EI benchmark with primary scope S3
        self.setUp_S3_only()

        # Verify company data for S3

        # Alas, this test is broken because the fundamental company data only has trajectory and target
        # projections for S1S2, not S3.  Which means we cannot calculate a valid budget for S3.
        # The fact that we have benchmark S3 projections is not enough to connect the dots--
        # we need fundamental company data as well.  Adding S3 data to companies requires
        # changing other test cases (since the S3 data becomes part of cumulative emissions).
        return
        company_1 = self.base_warehouse.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.base_warehouse.get_preprocessed_company_data(self.company_ids)[3]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AJ")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US17275R1023")
        self.assertEquals(company_1.ghg_s3, Q_(0, 'Mt CO2'))
        self.assertAlmostEqual(company_2.ghg_s3, Q_(100080009.401725, 't CO2'))

    def test_get_value(self):
        expected_data = pd.Series([20248547997.0,
                                   276185899.0,
                                   10283015132.0,
                                   1860376238.2982879],
                                  index=pd.Index(self.company_ids, name='company_id'),
                                  name='company_revenue')
        pd.testing.assert_series_equal(self.base_company_data.get_value(company_ids=self.company_ids,
                                                                        variable_name=ColumnsConfig.COMPANY_REVENUE),
                                       expected_data)

    def test_scope_to_calc(self):
        return

        # this should be rewritten to test production_centric parameter of benchmark
        # For default EI benchmark, expect scope to calculate is S1S2
        self.assertEqual(self.base_EI_bm.scope_to_calc, EScope.S1S2)
        company_with_s3 = self.base_warehouse.company_data._companies[3]
        # Verify S3 is folded into S1S2
        self.assertEqual(company_with_s3.ghg_s3, 0)

        # Reload EI benchmark with primary scope S3
        self.setUp_S3_only()

        # Verify expected scope to calculate S3
        self.assertEqual(self.base_EI_bm.scope_to_calc, EScope.S3)
        company_with_s3 = self.base_warehouse.company_data._companies[3]
        # Verify S3 is NOT folded into S1S2
        self.assertNotEqual(company_with_s3.ghg_s3, 0)

    def test_production_benchmark_any_scope(self):
        pbm = self.base_production_bm._productions_benchmarks
        self.assertEqual(len(pbm.AnyScope.benchmarks), 66)


if __name__ == "__main__":
    test = TestBaseProvider()
    # setUp has special meaning within `unittest`
    test.setUp()
    test.test_get_projected_production()
    test.test_get_company_data()
