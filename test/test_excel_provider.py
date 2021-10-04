import os
import unittest

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
import ITR
from ITR.data.excel import ExcelProviderCompany, ExcelProviderProductionBenchmark, ExcelProviderIntensityBenchmark, \
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
        self.excel_EI_bm = ExcelProviderIntensityBenchmark(excel_path=self.sector_data_path, benchmark_temperature=1.5,
                                                           benchmark_global_budget=396, is_AFOLU_included=False)
        self.excel_provider = DataWarehouse(self.excel_company_data, self.excel_production_bm, self.excel_EI_bm)
        self.company_ids = ["US0079031078",
                            "US00724F1012",
                            "FR0000125338"]
        self.company_info_at_base_year = pd.DataFrame(
            [[1.6982474347547, 1.04827859e+08, 'Electricity Utilities', 'North America'],
             [0.476586931582279, 5.98937002e+08, 'Electricity Utilities', 'North America'],
             [0.22457393169277, 1.22472003e+08, 'Electricity Utilities', 'Europe']],
            index=self.company_ids,
            columns=[ColumnsConfig.BASE_EI, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR, ColumnsConfig.REGION])

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
        expected = [2.05, 2.22, 2.06, 2.01, 1.93, 1.78, 1.71, 1.34, 2.21, 2.69, 2.65, temp_score.fallback_score, 2.89,
                    1.91, 2.16, 1.76, temp_score.fallback_score, temp_score.fallback_score, 1.47, 1.72, 1.76, 1.81,
                    temp_score.fallback_score, 1.78, 1.84, temp_score.fallback_score, temp_score.fallback_score, 1.74,
                    1.88, temp_score.fallback_score]
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, 2.259, places=2)

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
        expected_data = pd.DataFrame([[1.698247435, 1.581691084, 1.386040647, 1.190390211, 0.994739774, 0.799089338,
                                       0.782935186, 0.677935928, 0.572936671, 0.467937413, 0.362938156, 0.257938898,
                                       0.233746281, 0.209553665, 0.185361048, 0.161168432, 0.136975815, 0.124810886,
                                       0.112645956, 0.100481026, 0.088316097, 0.076151167, 0.062125588, 0.048100009,
                                       0.034074431, 0.020048852, 0.006023273, 0.005843878, 0.005664482, 0.005485087,
                                       0.005305691, 0.005126296
                                       ],
                                      [0.476586932, 0.444131055, 0.389650913, 0.335170772, 0.28069063, 0.226210489,
                                       0.22171226, 0.192474531, 0.163236802, 0.133999073, 0.104761344, 0.075523615,
                                       0.068787023, 0.062050431, 0.055313839, 0.048577247, 0.041840655, 0.038453251,
                                       0.035065847, 0.031678443, 0.028291039, 0.024903635, 0.020998121, 0.017092607,
                                       0.013187093, 0.009281579, 0.005376065, 0.005326111, 0.005276157, 0.005226203,
                                       0.005176249, 0.005126296
                                       ],
                                      [0.224573932, 0.17975612, 0.163761501, 0.147766883, 0.131772265, 0.115777646,
                                       0.099783028, 0.090628361, 0.081473693, 0.072319026, 0.063164359, 0.054009692,
                                       0.050089853, 0.046170015, 0.042250176, 0.038330338, 0.034410499, 0.031104249,
                                       0.027797999, 0.024491748, 0.021185498, 0.017879248, 0.016155615, 0.014431983,
                                       0.012708351, 0.010984719, 0.009261087, 0.008488943, 0.007716798, 0.006944654,
                                       0.00617251, 0.005400365]],
                                     index=self.company_ids,
                                     columns=range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))

        pd.testing.assert_frame_equal(
            self.excel_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year),
            expected_data)

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series([1.06866370e+08, 6.10584093e+08, 1.28474171e+08],
                                       index=self.company_ids,
                                       name=2025)
        pd.testing.assert_series_equal(
            self.excel_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025],
            expected_data_2025)

    def test_get_cumulative_value(self):
        projected_emission = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        projected_production = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]])
        expected_data = pd.Series([10.0, 50.0])
        pd.testing.assert_series_equal(
            self.excel_provider._get_cumulative_emission(projected_emission_intensity=projected_emission,
                                                         projected_production=projected_production), expected_data)

    def test_get_company_data(self):
        company_1 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[1]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AH")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US00724F1012")
        self.assertAlmostEqual(company_1.ghg_s1s2, 104827858.636039)
        self.assertAlmostEqual(company_2.ghg_s1s2, 598937001.892059)
        self.assertAlmostEqual(company_1.cumulative_budget, 1362284467.0830, places=4)
        self.assertAlmostEqual(company_2.cumulative_budget, 2262242040.68059, places=4)
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
