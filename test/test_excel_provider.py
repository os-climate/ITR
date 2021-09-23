import os
import unittest

import pandas as pd
import numpy as np
import ITR
from ITR.data.excel import ExcelProviderCompany, ExcelProviderProductionBenchmark, ExcelProviderIntensistyBenchmark, \
    TabsConfig
from ITR.data.data_warehouse import DataWarehouse
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.interfaces import EScope, ETimeFrames, PortfolioCompany
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod


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
        self.excel_EI_bm = ExcelProviderIntensistyBenchmark(excel_path=self.sector_data_path, benchmark_temperature=1.5,
                                                            benchmark_global_budget=396, AFOLU_included=True)
        self.excel_provider = DataWarehouse(self.excel_company_data, self.excel_production_bm, self.excel_EI_bm)
        self.company_ids = ["US0079031078",
                            "US00724F1012",
                            "FR0000125338"]
        self.ghg = pd.DataFrame([[1.04827859e+08, 'Electricity Utilities', 'North America'],
                                 [5.98937002e+08, 'Electricity Utilities', 'North America'],
                                 [1.22472003e+08, 'Electricity Utilities', 'Europe']],
                                index=self.company_ids,
                                columns=[ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR, ColumnsConfig.REGION])

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
        portfolio_data = ITR.utils.get_data(self.excel_provider, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, 2.567, places=2)

    def test_unit_of_measure_correction(self):
        company_ids = self.company_ids + ["US6293775085"]
        projected_values = pd.DataFrame(np.ones((4, 32)),
                                        columns=range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                      TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1),
                                        index=company_ids)
        expected_data = pd.DataFrame(np.ones((4, 32)),
                                     columns=range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1),
                                     index=company_ids)
        expected_data.iloc[0:3, :] = 3.6
        pd.testing.assert_frame_equal(
            self.excel_company_data._unit_of_measure_correction(company_ids, projected_values),
            expected_data)

    def test_get_projected_value(self):
        expected_data = pd.DataFrame([[1.698247435, 1.698247435, 1.590828573, 1.492707987, 1.403890821, 1.325025884,
                                       1.256900833, 1.199892962, 1.153286422, 1.115132019, 1.082871619, 1.054062505,
                                       1.026649109, 0.99885963, 0.969029076, 0.935600151, 0.897456359, 0.854466423,
                                       0.807721858, 0.759088111, 0.710432718, 0.663134402, 0.618007985, 0.575439357,
                                       0.535546775, 0.498300211, 0.463594864, 0.431292461, 0.401243246, 0.373297218,
                                       0.347309599, 0.32314329],
                                      [0.476586932, 0.476586932, 0.464695628, 0.464754889, 0.466332369, 0.469162115,
                                       0.472725797, 0.47629738, 0.479176649, 0.480954576, 0.481532513, 0.480898667,
                                       0.478873144, 0.474920056, 0.468037326, 0.456822975, 0.439924142, 0.416868713,
                                       0.38867473, 0.357527534, 0.325789571, 0.295235835, 0.266872969, 0.241107715,
                                       0.21798084, 0.197345262, 0.178974681, 0.162622136, 0.148048657, 0.135035628,
                                       0.123388813, 0.112938349],
                                      [0.224573932, 0.258012985, 0.261779459, 0.26416071, 0.266503379, 0.268691114,
                                       0.270569413, 0.271980435, 0.272823337, 0.273080838, 0.272767105, 0.27183449,
                                       0.270090124, 0.267129877, 0.262302026, 0.254777592, 0.243845281, 0.229393209,
                                       0.212192429, 0.193616639, 0.175038148, 0.157423255, 0.141276866, 0.12676707,
                                       0.113867496, 0.102458357, 0.092385201, 0.083489223, 0.0756213, 0.068647473,
                                       0.062450199, 0.056927654]],
                                     columns=range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1),
                                     index=self.company_ids)
        pd.testing.assert_frame_equal(self.excel_company_data._get_projection(self.company_ids,
                                                                              TabsConfig.PROJECTED_EI),
                                      expected_data, check_names=False)

    def test_get_benchmark(self):
        expected_data = pd.DataFrame([[0.412593499, 0.384543001, 0.33745769, 0.290372378, 0.243287067, 0.196201755,
                                       0.192314091, 0.167044928, 0.141775765, 0.116506602, 0.091237439, 0.065968277,
                                       0.060146072, 0.054323867, 0.048501662, 0.042679458, 0.036857253, 0.033929636,
                                       0.03100202, 0.028074403, 0.025146786, 0.022219169, 0.018843768, 0.015468367,
                                       0.012092965, 0.008717564, 0.005342163, 0.005298989, 0.005255816, 0.005212642,
                                       0.005169469, 0.005126296
                                       ],
                                      [0.412593499, 0.384543001, 0.33745769, 0.290372378, 0.243287067, 0.196201755,
                                       0.192314091, 0.167044928, 0.141775765, 0.116506602, 0.091237439, 0.065968277,
                                       0.060146072, 0.054323867, 0.048501662, 0.042679458, 0.036857253, 0.033929636,
                                       0.03100202, 0.028074403, 0.025146786, 0.022219169, 0.018843768, 0.015468367,
                                       0.012092965, 0.008717564, 0.005342163, 0.005298989, 0.005255816, 0.005212642,
                                       0.005169469, 0.005126296
                                       ],
                                      [0.358814981, 0.286546823, 0.260755703, 0.234964582, 0.209173461, 0.18338234,
                                       0.15759122, 0.142829434, 0.128067648, 0.113305863, 0.098544077, 0.083782292,
                                       0.077461601, 0.071140911, 0.064820221, 0.058499531, 0.052178841, 0.046847554,
                                       0.041516267, 0.03618498, 0.030853693, 0.025522406, 0.022743071, 0.019963735,
                                       0.0171844, 0.014405065, 0.01162573, 0.010380657, 0.009135584, 0.007890511,
                                       0.006645438, 0.005400365]],
                                     index=self.company_ids,
                                     columns=range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))

        pd.testing.assert_frame_equal(
            self.excel_EI_bm.get_intensity_benchmarks(self.ghg),
            expected_data)

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series([1.06866370e+08, 6.10584093e+08, 1.28474171e+08],
                                       index=self.company_ids,
                                       name=2025)
        pd.testing.assert_series_equal(self.excel_production_bm.get_company_projected_production(self.ghg)[2025],
                                       expected_data_2025)

    def test_get_cumulative_value(self):
        projected_emission = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        projected_production = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]])
        expected_data = pd.Series([10.0, 50.0])
        pd.testing.assert_series_equal(
            self.excel_provider._get_cumulative_emission(projected_emission_intensity=projected_emission,
                                                         projected_production=projected_production), expected_data)

    def test_get_company_data(self):
        company_1 = self.excel_provider.get_company_aggregates(self.company_ids)[0]
        company_2 = self.excel_provider.get_company_aggregates(self.company_ids)[1]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AH")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US00724F1012")
        self.assertAlmostEqual(company_1.ghg_s1s2, 104827858.636039)
        self.assertAlmostEqual(company_2.ghg_s1s2, 598937001.892059)
        self.assertAlmostEqual(company_1.cumulative_budget, 345325664.840567, places=4)
        self.assertAlmostEqual(company_2.cumulative_budget, 1973028172.73122, places=4)
        self.assertAlmostEqual(company_1.cumulative_target, 3769096510.09909, places=4)
        self.assertAlmostEqual(company_2.cumulative_target, 5912426347.23670, places=4)
        self.assertAlmostEqual(company_1.cumulative_trajectory, 3745094638.52858, places=4)
        self.assertAlmostEqual(company_2.cumulative_trajectory, 8631481789.38558, places=4)

    def test_get_value(self):
        expected_data = pd.Series([20248547997.0,
                                   276185899.0,
                                   10283015132.0],
                                  index=pd.Index(self.company_ids, name='company_id'),
                                  name='company_revenue')
        pd.testing.assert_series_equal(self.excel_company_data.get_value(company_ids=self.company_ids,
                                                                         variable_name=ColumnsConfig.COMPANY_REVENUE),
                                       expected_data)
