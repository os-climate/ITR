import os
import unittest

import pandas as pd

from numpy.testing import assert_array_equal
import ITR
from ITR.data.excel import ExcelProviderProductionBenchmark, ExcelProviderIntensityBenchmark
from ITR.data.template import TemplateProviderCompany
from ITR.data.data_warehouse import DataWarehouse
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.interfaces import EScope, ETimeFrames, PortfolioCompany
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod

from ITR.data.osc_units import ureg, Q_, PA_

class TestTemplateProvider(unittest.TestCase):
    """
    Test the excel provider
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_data_path = os.path.join(self.root, "inputs", "20220215 ITR Tool Sample Data.xlsx")
        self.sector_data_path = os.path.join(self.root, "inputs", "OECM_EI_and_production_benchmarks.xlsx")
        self.excel_production_bm = ExcelProviderProductionBenchmark(excel_path=self.sector_data_path)
        self.excel_EI_bm = ExcelProviderIntensityBenchmark(excel_path=self.sector_data_path, benchmark_temperature=Q_(1.5, ureg.delta_degC),
                                                           benchmark_global_budget=Q_(396, ureg('Gt CO2')), is_AFOLU_included=False)
        self.template_company_data = TemplateProviderCompany(excel_path=self.company_data_path)
        self.template_company_data._calculate_target_projections(production_bm=self.excel_production_bm, EI_bm=self.excel_EI_bm)
        self.excel_provider = DataWarehouse(self.template_company_data, self.excel_production_bm, self.excel_EI_bm)
        self.company_ids = ["US00130H1059", "US26441C2044", "KR7005490008"]
        # self.company_info_at_base_year = pd.DataFrame(
        #     [[Q_(1.6982474347547, ureg('t CO2/GJ')), Q_(1.04827859e+08, 'MWh'), 'MWh', 'Electricity Utilities', 'North America'],
        #      [Q_(0.476586931582279, ureg('t CO2/GJ')), Q_(5.98937002e+08, 'MWh'), 'MWh', 'Electricity Utilities', 'North America'],
        #      [Q_(0.22457393169277, ureg('t CO2/GJ')), Q_(1.22472003e+08, 'MWh'), 'MWh', 'Electricity Utilities', 'Europe']],
        #     index=self.company_ids,
        #     columns=[ColumnsConfig.BASE_EI, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.PRODUCTION_METRIC, ColumnsConfig.SECTOR, ColumnsConfig.REGION])

    def test_target_projections(self):
        comids = ['US00130H1059', 'US0185223007',
                  # 'US0138721065', 'US0158577090',
                  'US0188021085',
                  'US0236081024', 'US0255371017',
                  # 'US0298991011',
                  'US05351W1036',
                  # 'US05379B1070',
                  'US0921131092',
                  # 'CA1125851040',
                  'US1442851036', 'US1258961002', 'US2017231034',
                  'US18551QAA58', 'US2091151041', 'US2333311072', 'US25746U1097', 'US26441C2044',
                  'US29364G1031', 'US30034W1062',
                  'US30040W1080', 'US30161N1019', 'US3379321074',
                  'CA3495531079', 'US3737371050', 'US4198701009', 'US5526901096', 'US6703461052',
                  'US6362744095', 'US6680743050', 'US6708371033',
                  'US69331C1080',
                  'US69349H1077', 'KR7005490008',
                  ]
        
        for id in comids:
            print(target_projection(isin, data_target, data_emissions, data_prod))
        

    def test_temp_score(self):
        df_portfolio = pd.read_excel(self.company_data_path, sheet_name="Portfolio")
        companies = ITR.utils.dataframe_to_portfolio(df_portfolio)
        
        temperature_score = TemperatureScore(               
            time_frames = [ETimeFrames.LONG],     
            scopes=[EScope.S1S2],    
            aggregation_method=PortfolioAggregationMethod.WATS # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS.
        )
        amended_portfolio = temperature_score.calculate(data_warehouse=self.excel_provider, portfolio=companies)
        print(amended_portfolio[['company_name', 'time_frame', 'scope', 'temperature_score']])
        
    def test_temp_score_from_excel_data(self):
        comids = ['US00130H1059', 'US0185223007',
                  # 'US0138721065', 'US0158577090',
                  'US0188021085',
                  'US0236081024', 'US0255371017',]
        other_comids = [
                  # 'US0298991011',
                  'US05351W1036',
                  # 'US05379B1070',
                  'US0921131092',
                  # 'CA1125851040',
                  'US1442851036', 'US1258961002', 'US2017231034',
                  'US18551QAA58', 'US2091151041', 'US2333311072', 'US25746U1097', 'US26441C2044',
                  'US29364G1031', 'US30034W1062',
                  'US30040W1080', 'US30161N1019', 'US3379321074',
                  'CA3495531079', 'US3737371050', 'US4198701009', 'US5526901096', 'US6703461052',
                  'US6362744095', 'US6680743050', 'US6708371033',
                  # 'US6896481032',
                  'US69331C1080',
                  'US69349H1077', 'KR7005490008', # 'US69351T1060', 'US7234841010', 'US7365088472',
                  # 'US7445731067', 'US8581191009', 'US8168511090', 'US8425871071', 'CA87807B1076',
                  # 'US88031M1099', 'US8873991033', 'US9129091081', 'US92531L2079', 'US92840M1027',
                  # 'US92939U1060', 'US9818111026', 'US98389B1008'
                 ]
        
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
        expected = pd.Series([2.05, 2.22, 2.06, 2.01, 1.93, 1.78, 1.71, 1.34, 2.21, 2.69, 2.65, temp_score.fallback_score, 2.89,
                    1.91, 2.16, 1.76, temp_score.fallback_score, temp_score.fallback_score, 1.47, 1.72, 1.76, 1.81,
                    temp_score.fallback_score, 1.78, 1.84, temp_score.fallback_score, temp_score.fallback_score, 1.74,
                    1.88, temp_score.fallback_score], dtype='pint[delta_degC]')
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(2.259, ureg.delta_degC), places=2)

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
                                     index=self.company_ids,
                                     dtype='pint[t CO2/GJ]').astype('object')
        pd.testing.assert_frame_equal(self.excel_company_data.get_company_projected_trajectories(self.company_ids),
                                      expected_data, check_names=False)

    def test_get_benchmark(self):
        expected_data = pd.DataFrame([pd.Series([1.698247435, 1.581691084, 1.386040647, 1.190390211, 0.994739774, 0.799089338,
                                       0.782935186, 0.677935928, 0.572936671, 0.467937413, 0.362938156, 0.257938898,
                                       0.233746281, 0.209553665, 0.185361048, 0.161168432, 0.136975815, 0.124810886,
                                       0.112645956, 0.100481026, 0.088316097, 0.076151167, 0.062125588, 0.048100009,
                                       0.034074431, 0.020048852, 0.006023273, 0.005843878, 0.005664482, 0.005485087,
                                       0.005305691, 0.005126296
                                       ],name='US0079031078', dtype='pint[t CO2/GJ]'),
                                      pd.Series([0.476586932, 0.444131055, 0.389650913, 0.335170772, 0.28069063, 0.226210489,
                                       0.22171226, 0.192474531, 0.163236802, 0.133999073, 0.104761344, 0.075523615,
                                       0.068787023, 0.062050431, 0.055313839, 0.048577247, 0.041840655, 0.038453251,
                                       0.035065847, 0.031678443, 0.028291039, 0.024903635, 0.020998121, 0.017092607,
                                       0.013187093, 0.009281579, 0.005376065, 0.005326111, 0.005276157, 0.005226203,
                                       0.005176249, 0.005126296
                                       ],name='US00724F1012', dtype='pint[t CO2/GJ]'),
                                      pd.Series([0.224573932, 0.17975612, 0.163761501, 0.147766883, 0.131772265, 0.115777646,
                                       0.099783028, 0.090628361, 0.081473693, 0.072319026, 0.063164359, 0.054009692,
                                       0.050089853, 0.046170015, 0.042250176, 0.038330338, 0.034410499, 0.031104249,
                                       0.027797999, 0.024491748, 0.021185498, 0.017879248, 0.016155615, 0.014431983,
                                       0.012708351, 0.010984719, 0.009261087, 0.008488943, 0.007716798, 0.006944654,
                                       0.00617251, 0.005400365
                                       ],name='FR0000125338', dtype='pint[t CO2/GJ]')
                                     ],
                                     index=self.company_ids,
                                     columns=range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                   TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))
        pd.testing.assert_frame_equal(
            self.excel_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year),
            expected_data)

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series([1.06866370e+08, 6.10584093e+08, 1.28474171e+08],
                                       index=self.company_ids,
                                       name=2025,
                                       dtype='pint[MWh]').astype('object')
        pd.testing.assert_series_equal(
            self.excel_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025],
            expected_data_2025)

    def test_get_cumulative_value(self):
        projected_emission = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                                          dtype='pint[t CO2/GJ]')
        projected_production = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]],
                                            dtype='pint[GJ]')
        expected_data = pd.Series([10.0, 50.0], dtype='pint[Mt CO2]')
        pd.testing.assert_series_equal(
            self.excel_provider._get_cumulative_emission(projected_emission_intensity=projected_emission,
                                                         projected_production=projected_production), expected_data)

    def test_get_company_data(self):
        # "US0079031078" and "US00724F1012" are both Electricity Utilities
        company_1 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[1]
        self.assertEqual(company_1.company_name, "Company AG")
        self.assertEqual(company_2.company_name, "Company AH")
        self.assertEqual(company_1.company_id, "US0079031078")
        self.assertEqual(company_2.company_id, "US00724F1012")
        self.assertAlmostEqual(company_1.ghg_s1s2, Q_(104827858.636039, ureg('GJ')))
        self.assertAlmostEqual(company_2.ghg_s1s2, Q_(598937001.892059, ureg('GJ')))
        self.assertAlmostEqual(company_1.cumulative_budget, Q_(1362284467.0830, ureg('t CO2')), places=4)
        self.assertAlmostEqual(company_2.cumulative_budget, Q_(2262242040.68059, ureg('t CO2')), places=4)
        self.assertAlmostEqual(company_1.cumulative_target, Q_(3769096510.09909, ureg('t CO2')), places=4)
        self.assertAlmostEqual(company_2.cumulative_target, Q_(5912426347.23670, ureg('t CO2')), places=4)
        self.assertAlmostEqual(company_1.cumulative_trajectory, Q_(3745094638.52858, ureg('t CO2')), places=4)
        self.assertAlmostEqual(company_2.cumulative_trajectory, Q_(8631481789.38558, ureg('t CO2')), places=4)

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
    test = TestTemplateProvider()
    test.setUp()
    test.test_temp_score()
    test.get_target_projections()
