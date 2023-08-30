import os
import unittest

import pandas as pd

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
        self.data_warehouse = DataWarehouse(self.template_company_data, self.excel_production_bm, self.excel_EI_bm)
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
            projected_targets = EITargetProjector().project_ei_targets(c, bm_production_data).S1S2
            print(f"{c.company_name}: {projected_targets}")
        

    def test_temp_score(self):
        df_portfolio = pd.read_excel(self.company_data_path, sheet_name="Portfolio")
        # df_portfolio = df_portfolio[df_portfolio.company_id=='US00130H1059']
        portfolio = ITR.utils.dataframe_to_portfolio(df_portfolio)

        temperature_score = TemperatureScore(               
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],    
            aggregation_method=PortfolioAggregationMethod.WATS # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS.
        )

        portfolio_data = ITR.utils.get_data(self.data_warehouse, portfolio)

        amended_portfolio = temperature_score.calculate(data_warehouse=self.data_warehouse, data=portfolio_data, portfolio=portfolio)
        print(amended_portfolio[['company_name', 'time_frame', 'scope', 'temperature_score']])

    def _test_temp_score_from_excel_data(self):
        """
        DISABLED
        When running all tests in the /test directory, this test fails. When running all tests in
        test_template_provider.py, it passes. Indicates a state is saved somewhere(?). TODO: fix test.
        To enable test again, remove the leading '_' of the function name.  
        """
        excel_production_bm = ExcelProviderProductionBenchmark(excel_path=self.sector_data_path)
        excel_EI_bm = ExcelProviderIntensityBenchmark(excel_path=self.sector_data_path, benchmark_temperature=Q_(1.5, ureg.delta_degC),
                                                       benchmark_global_budget=Q_(396, ureg('Gt CO2')), is_AFOLU_included=False)
        template_company_data = TemplateProviderCompany(excel_path=self.company_data_path)
        data_warehouse = DataWarehouse(template_company_data, excel_production_bm, excel_EI_bm)
        comids = ['US00130H1059', 'US0185223007', 'US0188021085', 'US0236081024', 'US0255371017']

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
        portfolio_data = ITR.utils.get_data(data_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify company scores:
        expected = pd.Series([1.81, 1.87, 2.10, 2.18, 1.95], dtype='pint[delta_degC]')
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(1.982, ureg.delta_degC), places=2)

    def test_get_projected_value(self):
        company_ids = ["US00130H1059", "KR7005490008"]
        expected_data = pd.DataFrame([pd.Series(
            [605.1694925804982, 574.1215117186019, 555.4511355634547, 537.3879182390771, 519.9121149988865,
             503.0046231935541, 486.6469613900634, 470.8212491698162, 455.5101875837016, 440.6970402427642,
             426.36561502380243, 412.5002463698985, 399.08577816653377, 386.10754717457115, 373.5513670019951,
             361.4035125968896, 349.65070524470207, 338.28009805339593, 327.2792619106238, 316.6361718975723,
             306.3391941446274, 296.3770731144923, 286.7389192988567, 277.4141973151697, 268.39271439050435,
             259.66460921992547, 251.22034118718275, 243.05067993594568, 235.14669528018078, 227.49974744264256,
             220.10147761080776, 212.94379879992977], name='US0079031078', dtype='pint[t CO2/GWh]'),
                                      pd.Series(
            [2.1951083625828733, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
             2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], name='KR7005490008',
                                          dtype='pint[t CO2/Fe_ton]')],
                                     index=company_ids)
        expected_data.columns = range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                      TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)
        trajectories = self.template_company_data.get_company_projected_trajectories(company_ids)
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
        projected_emission = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], dtype='pint[t CO2/GJ]')
        projected_production = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]], dtype='pint[GJ]')
        expected_data = pd.Series([10.0, 50.0], dtype='pint[t CO2]')
        emissions = self.data_warehouse._get_cumulative_emissions(projected_ei=projected_emission,
                                                                  projected_production=projected_production)
        assert_pint_series_equal(self, emissions, expected_data)

    def test_get_company_data(self):
        # "US0079031078" and "US00724F1012" are both Electricity Utilities
        company_1 = self.data_warehouse.get_preprocessed_company_data(self.company_ids)[0]
        company_2 = self.data_warehouse.get_preprocessed_company_data(self.company_ids)[2]
        self.assertEqual(company_1.company_name, "AES Corp.")
        self.assertEqual(company_2.company_name, "POSCO")
        self.assertEqual(company_1.company_id, "US00130H1059")
        self.assertEqual(company_2.company_id, "KR7005490008")
        self.assertAlmostEqual(company_1.ghg_s1s2, Q_(43215000.0, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.ghg_s1s2, Q_(68874000.0, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_1.cumulative_budget, Q_(357340059.42291725, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.cumulative_budget, Q_(1221270599.1032874, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_1.cumulative_target, Q_(287877763.61957714, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.cumulative_target, Q_(1316305990.5630153, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_1.cumulative_trajectory, Q_(1127957483.2474423, ureg('t CO2')), places=7)
        self.assertAlmostEqual(company_2.cumulative_trajectory, Q_(2809084095.106841, ureg('t CO2')), places=7)

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
