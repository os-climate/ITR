import json
import unittest
import os
import pandas as pd
import ITR

from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark
from ITR.interfaces import ICompanyData, EScope, ETimeFrames, PortfolioCompany, IEIBenchmarkScopes, \
    IProductionBenchmarkScopes
from ITR.data.osc_units import ureg, Q_
from utils import assert_pint_frame_equal, assert_pint_series_equal


class TestBaseProvider(unittest.TestCase):
    """
    Test the Base provider
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_json = os.path.join(self.root, "inputs", "json", "fundamental_data.json")
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        self.benchmark_EI_json = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM.json")

        # load company data
        with open(self.company_json) as json_file:
            parsed_json = json.load(json_file)
        for company_data in parsed_json:
            company_data['emissions_metric'] = {'units': 't CO2'}
            if company_data['sector'] == 'Electricity Utilities':
                company_data['production_metric'] = {'units': 'MWh'}
            elif company_data['sector'] == 'Steel':
                company_data['production_metric'] = {'units': 'Fe_ton'}
        self.companies = [ICompanyData.parse_obj(company_data) for company_data in parsed_json]
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
                            "FR0000125338"]
        self.company_info_at_base_year = pd.DataFrame(
            [[Q_(1.6982474347547, 't CO2/GJ'), Q_(1.04827859e+08, 'MWh'), {'units': 'MWh'}, 'Electricity Utilities',
              'North America'],
             [Q_(0.476586931582279, 't CO2/GJ'), Q_(5.98937002e+08, 'MWh'), {'units': 'MWh'}, 'Electricity Utilities',
              'North America'],
             [Q_(0.22457393169277, 't CO2/GJ'), Q_(1.22472003e+08, 'MWh'), {'units': 'MWh'}, 'Electricity Utilities',
              'Europe']],
            index=self.company_ids,
            columns=[ColumnsConfig.BASE_EI, ColumnsConfig.BASE_YEAR_PRODUCTION, ColumnsConfig.PRODUCTION_METRIC,
                     ColumnsConfig.SECTOR, ColumnsConfig.REGION])

    def test_temp_score_from_json_data(self):
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
        expected = pd.Series([2.05, 2.22, 2.06], dtype='pint[delta_degC]', name='temperature_score')
        pd.testing.assert_series_equal(scores.temperature_score, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(2.11, ureg.delta_degC), places=2)

    def test_get_benchmark(self):
        seq_index = pd.RangeIndex.from_range(range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))
        data = [
            pd.Series([1.6982474347547, 1.5814362115005, 1.385357948863141, 1.18927968622576, 0.9932014235884,
                       0.7971231609510, 0.7809336851789, 0.675704827194235, 0.57047596920960, 0.4652471112250,
                       0.3600182532403, 0.2547893952557, 0.230543877037740, 0.20629835881977, 0.1820528406018,
                       0.1578073223838, 0.1335618041659, 0.121370273602458, 0.10917874303905, 0.0969872124756,
                       0.0847956819122, 0.0726041513488, 0.058547903118731, 0.04449165488867, 0.0304354066586,
                       0.0163791584285, 0.0023229101985, 0.002143122358755, 0.00196333451906, 0.0017835466794,
                       0.0016037588397, 0.0014239710000], index=seq_index, dtype="pint[t CO2/GJ]"),
            pd.Series([0.476586931582279, 0.4438761824346, 0.3889682148288414, 0.33406024722304, 0.27915227961723,
                       0.224244312011427, 0.2197107589327, 0.1902434296749848, 0.16077610041727, 0.13130877115956,
                       0.101841441901845, 0.0723741126441, 0.0655846189440570, 0.05879512524398, 0.05200563154391,
                       0.045216137843832, 0.0384266441438, 0.0350126391631084, 0.03159863418246, 0.02818462920181,
                       0.024770624221162, 0.0213566192405, 0.0174204357386057, 0.01348425223670, 0.00954806873479,
                       0.005611885232884, 0.0016757017310, 0.0016253555847724, 0.00157500943857, 0.00152466329236,
                       0.001474317146161, 0.0014239710000], index=seq_index, dtype="pint[t CO2/GJ]"),
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

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series([106866369.91163988, 610584093.0081439, 128474170.5748834],
                                       index=self.company_ids,
                                       name=2025,
                                       dtype='pint[MWh]')
        productions = self.base_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025]
        assert_pint_series_equal(self, expected_data_2025, productions)

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
        assert_pint_series_equal(self, cumulative_emissions, expected_data)

    def test_get_company_data(self):
        company_1 = self.base_warehouse.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.base_warehouse.get_preprocessed_company_data(self.company_ids)[1]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AH")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US00724F1012")
        self.assertAlmostEqual(company_1.ghg_s1s2, Q_(104827858.636039, 't CO2'))
        self.assertAlmostEqual(company_2.ghg_s1s2, Q_(598937001.892059, 't CO2'))
        self.assertAlmostEqual(company_1.cumulative_budget, Q_(4904224081.498916, 't CO2'))
        self.assertAlmostEqual(company_2.cumulative_budget, Q_(8144071346.450123, 't CO2'))
        self.assertAlmostEqual(company_1.cumulative_target, Q_(13568747436.356716, 't CO2'))
        self.assertAlmostEqual(company_2.cumulative_target, Q_(21284734850.052108, 't CO2'))
        self.assertAlmostEqual(company_1.cumulative_trajectory, Q_(13482340698.702868, 't CO2'))
        self.assertAlmostEqual(company_2.cumulative_trajectory, Q_(31073334441.78807, 't CO2'))

    def test_get_value(self):
        expected_data = pd.Series([20248547997.0,
                                   276185899.0,
                                   10283015132.0],
                                  index=pd.Index(self.company_ids, name='company_id'),
                                  name='company_revenue')
        pd.testing.assert_series_equal(self.base_company_data.get_value(company_ids=self.company_ids,
                                                                        variable_name=ColumnsConfig.COMPANY_REVENUE),
                                       expected_data)


if __name__ == "__main__":
    test = TestBaseProvider()
    test.setUp()
    test.test_get_projected_production()
    test.test_get_company_data()
