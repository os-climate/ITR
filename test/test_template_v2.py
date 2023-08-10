import json
import os
import unittest
from numpy.testing import assert_array_equal
import pandas as pd

import ITR
from ITR.data.osc_units import ureg, Q_, M_, asPintSeries, requantify_df_from_columns
from ITR.configs import ColumnsConfig, TemperatureScoreConfig

from ITR.data.base_providers import EITargetProjector, BaseProviderProductionBenchmark, BaseProviderIntensityBenchmark
from ITR.data.template import TemplateProviderCompany
from ITR.data.data_warehouse import DataWarehouse
from ITR.interfaces import EScope, ETimeFrames, PortfolioCompany, IProductionBenchmarkScopes, IEIBenchmarkScopes
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from utils import assert_pint_series_equal, assert_pint_frame_equal


pd.options.display.width=999
pd.options.display.max_columns=99
pd.options.display.min_rows=30

class TestTemplateProviderV2(unittest.TestCase):
    """
    Test the excel template provider
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.company_data_path = os.path.join(self.root, "inputs", "20220927 ITR V2 Sample Data.xlsx")
        # self.company_data_path = os.path.join(self.root, "inputs", "20230106 ITR V2 Sample Data.xlsx")
        self.template_company_data = TemplateProviderCompany(excel_path=self.company_data_path)
        # load production benchmarks
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        # load intensity benchmarks
        self.benchmark_EI_json = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM_PC.json")
        with open(self.benchmark_EI_json) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.base_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        self.data_warehouse = DataWarehouse(self.template_company_data, self.base_production_bm, self.base_EI_bm)
        self.company_ids = ["US00130H1059", "US26441C2044", "KR7005490008"]
        self.company_info_at_base_year = self.template_company_data.get_company_intensity_and_production_at_base_year(self.company_ids)


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
        breakpoint()
        company_data = self.template_company_data.get_company_data(comids)
        company_dict = {
            field : [ getattr(c, field) for c in company_data ]
            for field in [ ColumnsConfig.BASE_YEAR_PRODUCTION, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR, ColumnsConfig.REGION ]
        }
        company_dict[ColumnsConfig.SCOPE] = [ EScope.AnyScope ] * len(company_data)
        company_index = [ c.company_id for c in company_data ]
        company_sector_region_info = pd.DataFrame(company_dict, pd.Index(company_index, name='company_id'))
        bm_production_data = self.base_production_bm.get_company_projected_production(company_sector_region_info)
        # FIXME: We should pre-compute some of these target projections and make them reference data

        # AES won't converge because S1S2 is netzero in 2040 and S3 data netzero in 2050, then merged together.
        # When we try to re-create, the merged S1S2+S3 historic data is projected against a 2040-only target.

        selected_company_ids = [ 'US00130H1059', 'US26441C2044', 'KR7005490008' ]

        expected_projections = {
            selected_company_ids[0]: ('S1S2', [ 0.602, 0.5385, 0.4816, 0.4307, 0.3852, 0.3445, 0.3081, 0.2754,
            0.2462, 0.1972, 0.1578, 0.1258, 0.0998, 0.0786, 0.061, 0.0464, 0.0342, 0.0238,
            0.015, 0.013, 0.0111, 0.0094, 0.0078, 0.0063, 0.0049, 0.0035, 0.0023, 0.0011,
                                   0.0 ]),
            selected_company_ids[1]: ('S1', [ 0.2987, 0.2744, 0.2519, 0.2311, 0.2118, 0.1939, 0.1774, 0.162,
            0.1477, 0.1344, 0.1221, 0.1106, 0.1, 0.0901, 0.0808, 0.0722, 0.0642, 0.0567,
            0.0497, 0.0432, 0.0371, 0.0314,  0.026,  0.021, 0.0163, 0.0118, 0.0077, 0.0037,
                                   -0.0 ]),
            selected_company_ids[2]: ('S1S2', [2.0046, 1.9561, 1.9088, 1.8626, 1.8176, 1.7736, 1.7307, 1.6888,
            1.648, 1.5723, 1.5001, 1.4313, 1.3656, 1.3029, 1.243, 1.186, 1.1315, 1.0796,
            1.03, 0.8003, 0.6178, 0.4724, 0.3561, 0.2627, 0.1873, 0.126, 0.0759, 0.0345,
                                   -0.0])
        }

        for c in company_data:
            if c.company_id not in selected_company_ids:
                continue
            scope, expected_projection = expected_projections[c.company_id]
            c_proj_targets = c.projected_targets[scope].projections
            if isinstance(c_proj_targets, pd.Series):
                c_proj_targets = c_proj_targets[c_proj_targets.index>=2022]
                breakpoint()
                assert [round(x, 4) for x in ITR.nominal_values(c_proj_targets.pint.m)] == expected_projection
            else:
                while c_proj_targets[0].year < 2022:
                    c_proj_targets = c_proj_targets[1:]
                    assert [ITR.nominal_values(round(x.value.m,4)) for x in c_proj_targets] == expected_projection
            

    def test_temp_score(self):
        df_portfolio = pd.read_excel(self.company_data_path, sheet_name="Portfolio")
        requantify_df_from_columns(df_portfolio, inplace=True)
        # df_portfolio = df_portfolio[df_portfolio.company_id=='US00130H1059']
        portfolio = ITR.utils.dataframe_to_portfolio(df_portfolio)

        temperature_score = TemperatureScore(               
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],    
            aggregation_method=PortfolioAggregationMethod.WATS # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS.
        )

        try:
            portfolio_data = ITR.utils.get_data(self.data_warehouse, portfolio)
        except RuntimeWarning:
            assert False

        amended_portfolio = temperature_score.calculate(data_warehouse=self.data_warehouse, data=portfolio_data, portfolio=portfolio)
        print(amended_portfolio[['company_name', 'time_frame', 'scope', 'temperature_score']])

    def test_get_projected_value(self):
        company_ids = ["US00130H1059", "KR7005490008"]
        expected_data = pd.DataFrame([pd.Series(
            [767.9424440147617,
             670.6945556381868, 673.1392212423145, 674.5304806701733, 676.0177499692604, 677.6037527866383, 679.2912949309276, 681.083266835849, 682.9826460976751, 684.9925000888088, 687.1159886497745,
             689.3563668619705, 691.7169879036096, 694.2013059913407, 696.8128794101235, 699.5553736340013, 702.4325645405033, 705.4483417214807, 708.606711893272, 711.9118024091774, 715.3678648773108,
             718.9792788869904, 722.7505558469256, 726.6863429385525, 730.791427187973, 735.0707396600562, 739.5293597783664, 744.172519774691, 749.0056092720603, 754.0341800052599, 759.2639506829616,
             764.7008119957226],
            name='US0079031078', dtype='pint[t CO2/GWh]'),
                                      pd.Series(
            [2.4575180887731207,
             2.4377593432586617, 2.46092577581102, 2.4843590497760357, 2.5080621555037244, 2.53203811697784, 2.5562899921939826, 2.580820873541958, 2.605633888192434, 2.6307321984879453, 2.6561190023382943,
             2.6817975336203976, 2.707771062582624, 2.7340428962536856, 2.7606163788561187, 2.7874948922244167, 2.8146818562278657, 2.8421807291981303, 2.869995008361647, 2.898128230276882, 2.926583971276502,
             2.955365847914516, 2.9844775174184477, 3.013922678146586, 3.043705070050382, 3.073828475142039, 3.1042967179673644, 3.135113666083935, 3.1662832305446384, 3.1978093663866503, 3.2296960731259103,
             3.2619473952571574],
            name='KR7005490008',
            dtype='pint[t CO2/(t Steel)]')],
                                     index=company_ids)
        expected_data.columns = range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                      TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)
        trajectories = self.template_company_data.get_company_projected_trajectories(company_ids)
        assert_pint_frame_equal(self, trajectories.loc[:, EScope.S1S2, :], expected_data, places=4)

    def test_get_benchmark(self):
        # This test is a hot mess: the data are series of corp EI trajectories, which are company-specific
        # benchmarks are sector/region specific, and guide temperature scores, but we wouldn't expect
        # an exact match between the two except when the company's data was generated from the benchmark
        # (as test.utils.gen_company_data does).
        expected_data = pd.DataFrame([pd.Series([0.392,
                                                 0.347, 0.307, 0.272, 0.24, 0.213, 0.188, 0.149, 0.114, 0.0827, 0.0551,
                                                 0.030800000000000004, 0.030400000000000003, 0.03, 0.0296, 0.0293, 0.0289, 0.025, 0.0211, 0.0174, 0.0137,
                                                 0.0101, 0.0094, 0.00875, 0.00815, 0.00758, 0.00706, 0.00678, 0.00651, 0.00626, 0.00601,
                                                 0.00577], name='US00130H1059', dtype='pint[t CO2e/MWh]'),
                                      pd.Series([0.392,
                                                 0.347, 0.307, 0.272, 0.24, 0.213, 0.188, 0.149, 0.114, 0.0827, 0.0551,
                                                 0.030800000000000004, 0.030400000000000003, 0.03, 0.0296, 0.0293, 0.0289, 0.025, 0.0211, 0.0174, 0.0137,
                                                 0.0101, 0.0094, 0.00875, 0.00815, 0.00758, 0.00706, 0.00678, 0.00651, 0.00626, 0.00601,
                                                 0.00577], name='US26441C2044', dtype='pint[t CO2e/MWh]'),
                                      pd.Series([1.653,
                                                 1.571, 1.494, 1.421, 1.351, 1.284, 1.221, 1.098, 0.988, 0.889, 0.799,
                                                 0.719, 0.647, 0.583, 0.525, 0.472, 0.42499999999999993, 0.377, 0.33399999999999996, 0.29699999999999993, 0.263,
                                                 0.233, 0.206, 0.182, 0.16, 0.141, 0.125, 0.107, 0.09, 0.0734, 0.0573,
                                                 0.0416], name='KR7005490008', dtype='pint[t CO2e/(t Steel)]')
                                     ],
                                     index=self.company_ids)
        expected_data.columns = list(range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                           TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1))
        benchmarks = self.base_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year)
        assert_pint_frame_equal(self, benchmarks.loc[:, EScope.S1S2, :], expected_data)

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series([Q_(88056.5336432, ureg('GWh')), Q_(241.762219, ureg('TWh')), Q_(38.710168585224, ureg('Mt Steel'))],
                                       index=self.company_ids,
                                       name=2025)
        production = self.base_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025]
        # FIXME: this test is broken until we fix data for POSCO
        # Tricky beacuse productions are all sorts of types, not a PintArray, and when filled with uncertainties...this is not technically a pint series array!
        assert_pint_series_equal(self, production[:, EScope.S1S2].map(lambda x: Q_(ITR.nominal_values(x.m), x.u)), expected_data_2025, places=4)

    def test_get_cumulative_value(self):
        projected_emission = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], dtype='pint[t CO2/GJ]')
        projected_production = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]], dtype='pint[GJ]')
        expected_data = pd.Series([10.0, 50.0], dtype='pint[t CO2]')
        emissions = self.data_warehouse._get_cumulative_emissions(projected_ei=projected_emission,
                                                                  projected_production=projected_production)
        assert_pint_series_equal(self, emissions.iloc[:, -1], expected_data)

    def test_get_company_data(self):
        # "US0079031078" and "US00724F1012" are both Electricity Utilities
        companies = self.data_warehouse.get_preprocessed_company_data(self.company_ids)
        company_1 = companies[2]
        company_2 = companies[6]
        self.assertEqual(company_1.company_name, "AES Corp.")
        self.assertEqual(company_2.company_name, "POSCO")
        self.assertEqual(company_1.company_id, "US00130H1059")
        self.assertEqual(company_2.company_id, "KR7005490008")
        self.assertAlmostEqual(ITR.nominal_values(company_1.ghg_s1s2.m_as('Mt CO2')), 57.666199999999996, places=7)
        self.assertAlmostEqual(ITR.nominal_values(company_2.ghg_s1s2.m_as('Mt CO2')), 93.40289000000001, places=7)
        self.assertAlmostEqual(ITR.nominal_values(company_1.cumulative_budget.m_as('Mt CO2')), 247.35955561917217, places=7)
        self.assertAlmostEqual(ITR.nominal_values(company_2.cumulative_budget.m_as('Mt CO2')), 671.322874458741, places=7)
        self.assertAlmostEqual(ITR.nominal_values(company_1.cumulative_target.m_as('Mt CO2')), 628.4206195499177, places=7)
        self.assertAlmostEqual(ITR.nominal_values(company_2.cumulative_target.m_as('Mt CO2')), 970.361209683652496, places=7)
        self.assertAlmostEqual(ITR.nominal_values(company_1.cumulative_trajectory.m_as('Mt CO2')), 2980.565984113794, places=7)
        self.assertAlmostEqual(ITR.nominal_values(company_2.cumulative_trajectory.m_as('Mt CO2')), 4182.135839585814, places=7)

    def test_get_value(self):
        expected_data = pd.Series([10189000000.0,
                                   25079000000.0,
                                   55955872344.1009],
                                  index=pd.Index(self.company_ids, name='company_id'),
                                  name='company_revenue').astype('pint[USD]')
        pd.testing.assert_series_equal(
            asPintSeries(self.template_company_data.get_value(company_ids=self.company_ids,
                                                              variable_name=ColumnsConfig.COMPANY_REVENUE)),
            expected_data)


if __name__ == "__main__":
    test = TestTemplateProviderV2()
    test.setUp()
    test.test_temp_score()
    test.test_target_projections()
    test.test_get_company_data()
