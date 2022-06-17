import os
import unittest
import pandas as pd
from numpy.testing import assert_array_equal
import ITR

from ITR.data.excel import ExcelProviderCompany, ExcelProviderProductionBenchmark, ExcelProviderIntensityBenchmark
from ITR.data.data_warehouse import DataWarehouse
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.interfaces import EScope, ETimeFrames, PortfolioCompany
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod

from ITR.data.osc_units import ureg, Q_, PA_
from test_base_providers import assert_pint_frame_equal, assert_pint_series_equal


class TestExcelProvider(unittest.TestCase):
    """
    Test the excel provider
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_data_path = os.path.join(self.root, "inputs", "test_data_company.xlsx")
        self.sector_data_path = os.path.join(self.root, "inputs", "OECM_EI_and_production_benchmarks.xlsx")
        self.excel_company_data = ExcelProviderCompany(excel_path=self.company_data_path)
        self.excel_production_bm = ExcelProviderProductionBenchmark(excel_path=self.sector_data_path)
        self.excel_EI_bm = ExcelProviderIntensityBenchmark(excel_path=self.sector_data_path,
                                                           benchmark_temperature=Q_(1.5, ureg.delta_degC),
                                                           benchmark_global_budget=Q_(396, ureg('Gt CO2')),
                                                           is_AFOLU_included=False)
        self.excel_provider = DataWarehouse(self.excel_company_data, self.excel_production_bm, self.excel_EI_bm)
        # "US0079031078","US00724F1012","FR0000125338" are all Electricity Utilities
        self.company_ids = ["US0079031078",
                            "US00724F1012",
                            "FR0000125338"]
        self.company_info_at_base_year = pd.DataFrame(
            [[Q_(1.6982474347547, ureg('t CO2/GJ')), Q_(1.04827859e+08, 'MWh'), 'MWh', 'Electricity Utilities',
              'North America'],
             [Q_(0.476586931582279, ureg('t CO2/GJ')), Q_(5.98937002e+08, 'MWh'), 'MWh', 'Electricity Utilities',
              'North America'],
             [Q_(0.22457393169277, ureg('t CO2/GJ')), Q_(1.22472003e+08, 'MWh'), 'MWh', 'Electricity Utilities',
              'Europe']],
            index=self.company_ids,
            columns=[ColumnsConfig.BASE_EI, ColumnsConfig.BASE_YEAR_PRODUCTION, ColumnsConfig.PRODUCTION_METRIC,
                     ColumnsConfig.SECTOR, ColumnsConfig.REGION])

    def test_temp_score_from_excel_data(self):
        comids = ['US0079031078', 'US00724F1012', 'FR0000125338', 'US17275R1023', 'CH0198251305', 'US1266501006',
                  'FR0000120644', 'US24703L1035', 'TW0002308004', 'FR0000120321', 'CH0038863350', 'US8356993076',
                  'JP3401400001', 'US6541061031', 'GB0031274896', 'US6293775085', 'US7134481081', 'JP0000000001',
                  'NL0000000002', 'IT0000000003', 'SE0000000004', 'SE0000000005', 'NL0000000006', 'CN0000000007',
                  'CN0000000008', 'CN0000000009', 'BR0000000010', 'BR0000000011', 'BR0000000012', 'AR0000000013']

        # Calculate Temp Scores
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio = []
        for company in comids:
            portfolio.append(PortfolioCompany(
                company_name=company,
                company_id=company,
                investment_value=100,
                company_isin=company,
            )
            )
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.excel_provider, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify company scores:
        expected = pd.Series(
            [2.05, 2.22, 2.06, 2.01, 1.93, 1.78, 1.71, 1.34, 2.21, 2.69, 2.65, temp_score.fallback_score, 2.89,
             1.91, 2.16, 1.76, temp_score.fallback_score, temp_score.fallback_score, 1.47, 1.72, 1.76, 1.81,
             temp_score.fallback_score, 1.78, 1.84, temp_score.fallback_score, temp_score.fallback_score, 1.79,
             1.88, temp_score.fallback_score], dtype='pint[delta_degC]')
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(2.259, ureg.delta_degC), places=2)

    def test_get_projected_value(self):
        expected_data = pd.DataFrame([[0.47173539854, 0.47173539854, 0.44189682578, 0.41464110746,
                                       0.38996967250, 0.36806274561, 0.34913912031, 0.33330360054,
                                       0.32035733950, 0.30975889415, 0.30079767200, 0.29279514028,
                                       0.28518030797, 0.27746100833, 0.26917474321, 0.25988893074,
                                       0.24929343300, 0.23735178428, 0.22436718269, 0.21085780869,
                                       0.19734242153, 0.18420400050, 0.17166888467, 0.15984426596,
                                       0.14876299319, 0.13841672541, 0.12877635118, 0.11980346134,
                                       0.11145645720, 0.10369367159, 0.09647488879, 0.08976202499
                                       ],
                                      [0.13238525877, 0.13238525877, 0.12908211877, 0.12909858023,
                                       0.12953676930, 0.13032280984, 0.13131272139, 0.13230482791,
                                       0.13310462474, 0.13359849339, 0.13375903144, 0.13358296304,
                                       0.13302031772, 0.13192223791, 0.13001036827, 0.12689527083,
                                       0.12220115053, 0.11579686468, 0.10796520283, 0.09931320396,
                                       0.09049710319, 0.08200995414, 0.07413138016, 0.06697436519,
                                       0.06055023333, 0.05481812839, 0.04971518910, 0.04517281563,
                                       0.04112462705, 0.03750989666, 0.03427467036, 0.03137176364
                                       ],
                                      [0.06238164769, 0.07167027361, 0.07271651633, 0.07337797502,
                                       0.07402871633, 0.07463642054, 0.07515817014, 0.07555012074,
                                       0.07578426020, 0.07585578822, 0.07576864027, 0.07550958065,
                                       0.07502503448, 0.07420274359, 0.07286167398, 0.07077155342,
                                       0.06773480031, 0.06372033578, 0.05894234130, 0.05378239979,
                                       0.04862170773, 0.04372868191, 0.03924357390, 0.03521307488,
                                       0.03162986008, 0.02846065481, 0.02566255582, 0.02319145084,
                                       0.02100591678, 0.01906874251, 0.01734727738, 0.01581323733
                                       ]],
                                     index=self.company_ids,
                                     dtype='pint[t CO2/GJ]').astype('object')
        expected_data.columns = range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                      TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)
        trajectories = self.excel_company_data.get_company_projected_trajectories(self.company_ids)
        assert_pint_frame_equal(self, trajectories, expected_data)

    def test_get_benchmark(self):
        expected_data = pd.DataFrame([pd.Series([1.69824743475, 1.58143621150, 1.38535794886, 1.18927968623,
                                                 0.99320142359, 0.79712316095, 0.78093368518, 0.67570482719,
                                                 0.57047596921, 0.46524711122, 0.36001825324, 0.25478939526,
                                                 0.23054387704, 0.20629835882, 0.18205284060, 0.15780732238,
                                                 0.13356180417, 0.12137027360, 0.10917874304, 0.09698721248,
                                                 0.08479568191, 0.07260415135, 0.05854790312, 0.04449165489,
                                                 0.03043540666, 0.01637915843, 0.00232291020, 0.00214312236,
                                                 0.00196333452, 0.00178354668, 0.00160375884, 0.00142397100
                                                 ], name='US0079031078', dtype='pint[t CO2/GJ]'),
                                      pd.Series([0.47658693158, 0.44387618243, 0.38896821483, 0.33406024722,
                                                 0.27915227962, 0.22424431201, 0.21971075893, 0.19024342967,
                                                 0.16077610042, 0.13130877116, 0.10184144190, 0.07237411264,
                                                 0.06558461894, 0.05879512524, 0.05200563154, 0.04521613784,
                                                 0.03842664414, 0.03501263916, 0.03159863418, 0.02818462920,
                                                 0.02477062422, 0.02135661924, 0.01742043574, 0.01348425224,
                                                 0.00954806873, 0.00561188523, 0.00167570173, 0.00162535558,
                                                 0.00157500944, 0.00152466329, 0.00147431715, 0.00142397100
                                                 ], name='US00724F1012', dtype='pint[t CO2/GJ]'),
                                      pd.Series([0.22457393169, 0.17895857242, 0.16267932465, 0.14640007689,
                                                 0.13012082912, 0.11384158136, 0.09756233359, 0.08824475611,
                                                 0.07892717862, 0.06960960113, 0.06029202364, 0.05097444616,
                                                 0.04698485296, 0.04299525976, 0.03900566657, 0.03501607337,
                                                 0.03102648017, 0.02766139400, 0.02429630784, 0.02093122167,
                                                 0.01756613550, 0.01420104933, 0.01244674461, 0.01069243990,
                                                 0.00893813518, 0.00718383046, 0.00542952574, 0.00464364090,
                                                 0.00385775605, 0.00307187120, 0.00228598636, 0.00150010151],
                                                name='FR0000125338', dtype='pint[t CO2/GJ]')
                                      ],
                                     index=self.company_ids)
        expected_data.columns = list(range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                           TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))
        benchmarks = self.excel_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year)
        assert_pint_frame_equal(self, benchmarks, expected_data)

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series([106866369.91163988, 610584093.0081439, 128474170.5748834],
                                       index=self.company_ids,
                                       name=2025,
                                       dtype='pint[MWh]').astype('object')
        production = self.excel_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025]
        assert_pint_series_equal(self, production, expected_data_2025)

    def test_get_cumulative_value(self):
        projected_emission = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                                          dtype='pint[t CO2/GJ]')
        projected_production = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]],
                                            dtype='pint[GJ]')
        expected_data = pd.Series([10.0, 50.0], dtype='pint[t CO2]')
        emissions = self.excel_provider._get_cumulative_emissions(projected_ei=projected_emission,
                                                                  projected_production=projected_production)
        assert_pint_series_equal(self, emissions, expected_data)

    def test_get_company_data(self):
        # "US0079031078" and "US00724F1012" are both Electricity Utilities
        company_1 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[1]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AH")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US00724F1012")
        self.assertAlmostEqual(company_1.ghg_s1s2, Q_(104827858.636039, ureg('t CO2')))
        self.assertAlmostEqual(company_2.ghg_s1s2, Q_(598937001.892059, ureg('t CO2')))
        self.assertAlmostEqual(company_1.cumulative_budget, Q_(802170778.6532312, ureg('t CO2')))
        self.assertAlmostEqual(company_2.cumulative_budget, Q_(4746756343.422796, ureg('t CO2')))
        self.assertAlmostEqual(company_1.cumulative_target, Q_(2219403623.3851275, ureg('t CO2')))
        self.assertAlmostEqual(company_2.cumulative_target, Q_(12405766829.584078, ureg('t CO2')))
        self.assertAlmostEqual(company_1.cumulative_trajectory, Q_(2205270305.0716036, ureg('t CO2')))
        self.assertAlmostEqual(company_2.cumulative_trajectory, Q_(18111033302.421572, ureg('t CO2')))

    def test_get_value(self):
        expected_data = pd.Series([20248547997.0,
                                   276185899.0,
                                   10283015132.0],
                                  index=pd.Index(self.company_ids, name='company_id'),
                                  name='company_revenue')
        pd.testing.assert_series_equal(self.excel_company_data.get_value(company_ids=self.company_ids,
                                                                         variable_name=ColumnsConfig.COMPANY_REVENUE),
                                       expected_data)


if __name__ == "__main__":
    test = TestExcelProvider()
    test.setUp()
    test.test_temp_score_from_excel_data()
    test.test_get_company_data()
