import os
import unittest

import pandas as pd
import numpy as np

from numpy.testing import assert_array_equal
import ITR
from ITR.data.base_providers import EITargetProjector
from ITR.data.excel import ExcelProviderProductionBenchmark, ExcelProviderIntensityBenchmark
from ITR.data.template import TemplateProviderCompany
from ITR.data.data_warehouse import DataWarehouse
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.interfaces import EScope, ETimeFrames, PortfolioCompany
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.data.osc_units import ureg, Q_
from utils import assert_pint_frame_equal
from test_base_providers import assert_pint_series_equal


class TestTemplateProvider(unittest.TestCase):
    """
    Test the excel template provider
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_data_path = os.path.join(self.root, "inputs", "20220215 ITR Tool Sample Data.xlsx")
        self.sector_data_path = os.path.join(self.root, "inputs", "OECM_EI_and_production_benchmarks.xlsx")
        self.excel_production_bm = ExcelProviderProductionBenchmark(excel_path=self.sector_data_path)
        self.excel_EI_bm = ExcelProviderIntensityBenchmark(excel_path=self.sector_data_path, benchmark_temperature=Q_(1.5, ureg.delta_degC),
                                                           benchmark_global_budget=Q_(396, ureg('Gt CO2')), is_AFOLU_included=False)
        self.template_company_data = TemplateProviderCompany(excel_path=self.company_data_path)
        self.excel_provider = DataWarehouse(self.template_company_data, self.excel_production_bm, self.excel_EI_bm)
        self.company_ids = ["US00130H1059", "US26441C2044", "KR7005490008"]
        self.company_info_at_base_year = pd.DataFrame(
            [[Q_(1.6982474347547, ureg('t CO2/GJ')), Q_(1.04827859e+08, 'MWh'), 'MWh', 'Electricity Utilities', 'North America'],
             [Q_(0.476586931582279, ureg('t CO2/GJ')), Q_(5.98937002e+08, 'MWh'), 'MWh', 'Electricity Utilities', 'North America'],
             [Q_(0.22457393169277, ureg('t CO2/GJ')), Q_(1.22472003e+08, 'MWh'), 'MWh', 'Electricity Utilities', 'Europe']],
            index=self.company_ids,
            columns=[ColumnsConfig.BASE_EI, ColumnsConfig.BASE_YEAR_PRODUCTION, ColumnsConfig.PRODUCTION_METRIC, ColumnsConfig.SECTOR, ColumnsConfig.REGION])

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
        company_data = self.template_company_data.get_company_data(comids)
        for c in company_data:
            company_sector_region_info = pd.DataFrame({
                ColumnsConfig.COMPANY_ID: [ c.company_id ],
                ColumnsConfig.BASE_YEAR_PRODUCTION: [ c.base_year_production ],
                ColumnsConfig.GHG_SCOPE12: [ c.ghg_s1s2 ],
                ColumnsConfig.SECTOR: [ c.sector ],
                ColumnsConfig.REGION: [ c.region ],
            }, index=[0])
            bm_production_data = (self.excel_production_bm.get_company_projected_production(company_sector_region_info)
                                  # We transpose the data so that we get a pd.Series that will accept the pint units as a whole (not element-by-element)
                                  .iloc[0].T
                                  .astype(f'pint[{str(c.base_year_production.units)}]'))
            print(f"{c.company_name}: {EITargetProjector().project_ei_targets(c.target_data, c.historic_data, bm_production_data).S1S2}")
        

    def test_temp_score(self):
        df_portfolio = pd.read_excel(self.company_data_path, sheet_name="Portfolio")
        # df_portfolio = df_portfolio[df_portfolio.company_id=='US00130H1059']
        companies = ITR.utils.dataframe_to_portfolio(df_portfolio)
        
        temperature_score = TemperatureScore(               
            time_frames=[ETimeFrames.LONG],
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
            ))
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
        company_ids = ["US00130H1059", "KR7005490008"]
        expected_data = pd.DataFrame([pd.Series(
            [605.1694925804982,574.1215117186019,555.4511355634547,537.3879182390771,519.9121149988865,
             503.0046231935541,486.6469613900634,470.8212491698162,455.5101875837016,440.6970402427642,
             426.36561502380243,412.5002463698985,399.08577816653377,386.10754717457115,373.5513670019951,
             361.4035125968896,349.65070524470207,338.28009805339593,327.2792619106238,316.6361718975723,
             306.3391941446274,296.3770731144923,286.7389192988567,277.4141973151697,268.39271439050435,
             259.66460921992547,251.22034118718275,243.05067993594568,235.14669528018078,227.49974744264256,
             220.10147761080776,212.94379879992977], name='US0079031078', dtype='pint[t CO2/GWh]'),
                                      pd.Series(
            [2.1951083625828733,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
             2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0], name='KR7005490008', dtype='pint[t CO2/Fe_ton]')],
                                     index=company_ids)
        expected_data.columns = range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                      TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)
        trajectories = self.template_company_data.get_company_projected_trajectories(company_ids)
        assert_pint_frame_equal(self, trajectories, expected_data)

    def test_get_benchmark(self):
        expected_data = pd.DataFrame([pd.Series([1.6982474347547,1.58143621150052,1.3853579488631413,1.1892796862257624,
                                                 0.9932014235883839,0.7971231609510052,0.7809336851788673,0.6757048271942354,
                                                 0.5704759692096036,0.46524711122497175,0.3600182532403398,0.2547893952557078,
                                                 0.23054387703774004,0.20629835881977232,0.1820528406018046,0.15780732238383688,
                                                 0.1335618041658692,0.12137027360245764,0.10917874303904605,0.09698721247563447,
                                                 0.08479568191222286,0.07260415134881128,0.058547903118739995,0.044491654888668734,
                                                 0.030435406658597484,0.016379158428526226,0.002322910198454965,0.0021431223587553565,
                                                 0.0019633345190557478,0.001783546679356139,0.0016037588396565301,0.0014239709999569286
                                                 ], name='US0079031078', dtype='pint[t CO2/GJ]'),
                                      pd.Series([0.476586931582279,0.4438761824346462,0.3889682148288414,0.33406024722303657,
                                                 0.2791522796172317,0.2242443120114269,0.21971075893269795,0.19024342967498475,
                                                 0.16077610041727156,0.13130877115955836,0.10184144190184515,0.07237411264413192,
                                                 0.06558461894405697,0.05879512524398202,0.05200563154390709,0.045216137843832147,
                                                 0.038426644143757224,0.03501263916310838,0.03159863418245954,0.028184629201810696,
                                                 0.024770624221161847,0.021356619240513006,0.017420435738605664,0.013484252236698325,
                                                 0.009548068734790988,0.005611885232883652,0.0016757017309763137,0.0016253555847724364,
                                                 0.001575009438568559,0.0015246632923646814,0.0014743171461608039,0.0014239709999569286
                                                 ], name='US00724F1012', dtype='pint[t CO2/GJ]'),
                                      pd.Series([0.22457393169277, 0.17895857241820134, 0.16267932465294896, 0.1464000768876966,
                                                 0.130120829122444, 0.11384158135719184, 0.09756233359193943, 0.0882447561051738,
                                                 0.078927178618408, 0.06960960113164248, 0.06029202364487683, 0.0509744461581112,
                                                 0.046984852960782, 0.04299525976345229, 0.03900566656612284, 0.0350160733687934,
                                                 0.031026480171464, 0.02766139400289410, 0.02429630783432425, 0.0209312216657544,
                                                 0.017566135497185, 0.01420104932861466, 0.01244674461183282, 0.0106924398950510,
                                                 0.008938135178269, 0.00718383046148729, 0.00542952574470545, 0.0046436408978128,
                                                 0.003857756050920, 0.00307187120402739, 0.00228598635713470, 0.0015001015102420],
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
        projected_emission = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], dtype='pint[t CO2/GJ]')
        projected_production = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]], dtype='pint[GJ]')
        expected_data = pd.Series([10.0, 50.0], dtype='pint[t CO2]')
        emissions = self.excel_provider._get_cumulative_emissions(projected_emission_intensity=projected_emission,
                                                                  projected_production=projected_production)
        assert_pint_series_equal(self, emissions, expected_data)

    def test_get_company_data(self):
        # "US0079031078" and "US00724F1012" are both Electricity Utilities
        company_1 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.excel_provider.get_preprocessed_company_data(self.company_ids)[1]
        self.assertEqual(company_1.company_name, "AES Corp.")
        self.assertEqual(company_2.company_name, "Duke Energy Corp.")
        self.assertEqual(company_1.company_id, "US00130H1059")
        self.assertEqual(company_2.company_id, "US26441C2044")
        self.assertAlmostEqual(company_1.ghg_s1s2, Q_(43215000.0, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.ghg_s1s2, Q_(82018839.2, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_1.cumulative_budget, Q_(47988154.144799985, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.cumulative_budget, Q_(673654041.4715265, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_1.cumulative_target, Q_(287877763.61957714, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.cumulative_target, Q_(1072738125.127108, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_1.cumulative_trajectory, Q_(1018535561.45581, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.cumulative_trajectory, Q_(2933704424.3851283, ureg('t CO2')), places=7)

    def test_get_value(self):
        expected_data = pd.Series([10189000000.0,
                                   25079000000.0,
                                   55955872344.1],
                                  index=pd.Index(self.company_ids, name='company_id'),
                                  name='company_revenue')
        pd.testing.assert_series_equal(self.template_company_data.get_value(company_ids=self.company_ids,
                                                                            variable_name=ColumnsConfig.COMPANY_REVENUE),
                                       expected_data)


if __name__ == "__main__":
    test = TestTemplateProvider()
    test.setUp()
    test.test_temp_score()
    test.test_target_projections()
