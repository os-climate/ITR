import json
import os
import unittest

import pandas as pd
from pint_pandas import PintArray as PA_
from utils import assert_pint_frame_equal, assert_pint_series_equal

import ITR
from ITR import data_dir
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.base_providers import (
    BaseProviderIntensityBenchmark,
    BaseProviderProductionBenchmark,
)
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.osc_units import Q_, asPintSeries, requantify_df_from_columns, ureg
from ITR.data.template import TemplateProviderCompany
from ITR.interfaces import (
    EScope,
    ETimeFrames,
    IEIBenchmarkScopes,
    IProductionBenchmarkScopes,
)
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore

pd.options.display.width = 999
pd.options.display.max_columns = 99
pd.options.display.min_rows = 30


class TemplateV2:
    def __init__(self) -> None:
        root = os.path.dirname(os.path.abspath(__file__))
        self.company_data_path = os.path.join(root, "inputs", "20231031 ITR V2 SBTi Data.xlsx")
        self.template_company_data = TemplateProviderCompany(excel_path=self.company_data_path)
        # load production benchmarks
        benchmark_prod_json = os.path.join(data_dir, "benchmark_production_OECM.json")
        with open(benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.model_validate(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        # load intensity benchmarks
        benchmark_EI_json = os.path.join(data_dir, "benchmark_EI_OECM_S3.json")
        with open(benchmark_EI_json) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.model_validate(parsed_json)
        self.base_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        self.data_warehouse = DataWarehouse(self.template_company_data, self.base_production_bm, self.base_EI_bm)
        self.company_ids = ["JP3672400003", "FR0000131906"]
        self.company_info_at_base_year = self.template_company_data.get_company_intensity_and_production_at_base_year(
            self.company_ids
        )


template_V2 = TemplateV2()


class TestTemplateProviderV2(unittest.TestCase):
    """
    Test the excel template provider
    """

    def setUp(self) -> None:
        self.company_data_path = template_V2.company_data_path
        self.template_company_data = template_V2.template_company_data
        self.base_production_bm = template_V2.base_production_bm
        self.base_EI_bm = template_V2.base_EI_bm
        self.data_warehouse = template_V2.data_warehouse
        self.company_ids = template_V2.company_ids
        self.company_info_at_base_year = template_V2.company_info_at_base_year

    def test_target_projections(self):
        comids = [
            "JP3672400003",
            "FR0000131906",
        ]

        company_data = self.template_company_data.get_company_data(comids)
        company_dict = {
            field: [getattr(c, field) for c in company_data]
            for field in [
                ColumnsConfig.BASE_YEAR_PRODUCTION,
                ColumnsConfig.GHG_SCOPE3,
                ColumnsConfig.SECTOR,
                ColumnsConfig.REGION,
            ]
        }
        company_dict[ColumnsConfig.SCOPE] = [EScope.AnyScope] * len(company_data)
        company_index = [c.company_id for c in company_data]
        company_sector_region_info = pd.DataFrame(company_dict, pd.Index(company_index, name="company_id"))
        bm_production_data = self.base_production_bm.get_company_projected_production(company_sector_region_info)
        # FIXME: We should pre-compute some of these target projections and make them reference data

        selected_company_ids = ["JP3672400003", "FR0000131906"]

        expected_data = pd.DataFrame(
            {
                (selected_company_ids[0], "S1S2S3"): PA_(
                    [
                        118.7703348114596,
                        107.0907156008301,
                        96.92835293332904,
                        87.72969430352153,
                        79.40329721755833,
                        71.86639712986789,
                        65.0440837940901,
                        58.86855579769937,
                        53.27844585811965,
                        47.01073049681457,
                        41.42489556733254,
                        36.44373666667907,
                        31.998794990233467,
                        28.029366288530163,
                        24.48162215744484,
                        21.30783092652994,
                        18.465666854241693,
                        15.917597619052346,
                        13.63034123051773,
                        11.574384490734468,
                        9.723556028869803,
                        8.054647722521366,
                        6.547079021037208,
                        5.182599307770745,
                        3.945023989579242,
                        2.8200004906939684,
                        1.7948007614843993,
                        0.8581372968926091,
                        -3.4493161022313725e-15,
                    ],
                    dtype="g CO2e/pkm",
                ),
                (selected_company_ids[1], "S1S2S3"): PA_(
                    [
                        184.99999999999994,
                        169.1611665059957,
                        154.5784602283449,
                        141.1483753938147,
                        128.77593517827322,
                        117.37398891523645,
                        106.86256721492924,
                        97.16829022199296,
                        88.22382463317741,
                        79.96738545716018,
                        72.34227882970991,
                        65.29648250120431,
                        58.78226089227427,
                        52.7558118691356,
                        47.176942624884035,
                        42.00877226840127,
                        37.217458920146036,
                        32.77194929544543,
                        28.643748922300688,
                        24.80671129340855,
                        21.236844392205644,
                        17.91213316130398,
                        14.812376599653215,
                        11.919038283012224,
                        9.215109201639855,
                        6.684981900257156,
                        4.314334988965992,
                        2.090027170549812,
                        1.6940658945086007e-14,
                    ],
                    dtype="g CO2e/pkm",
                )
            }
        )

        for c in company_data:
            if c.company_id not in selected_company_ids:
                continue
            expected_column = expected_data[c.company_id]
            scope_name = expected_column.columns[0]
            expected_projection = expected_column[scope_name]
            c_proj_targets = c.projected_targets[scope_name].projections
            if isinstance(c_proj_targets, pd.Series):
                c_proj_targets = c_proj_targets[c_proj_targets.index >= 2022]
                assert_pint_series_equal(self, c_proj_targets, expected_projection, places=2)
            else:
                while c_proj_targets[0].year < 2022:
                    # c_proj_targets is a list of nasty BaseModel types so we cannot use Pandas asserters
                    c_proj_targets = c_proj_targets[1:]
                    assert [
                        ITR.nominal_values(round(x.value.m_as(expected_projection.dtype.units), 2))
                        for x in c_proj_targets
                    ] == expected_projection.pint.m.tolist()

    def test_temp_score(self):
        df_portfolio = pd.read_excel(self.company_data_path, sheet_name="Portfolio")
        requantify_df_from_columns(df_portfolio, inplace=True)
        # df_portfolio = df_portfolio[df_portfolio.company_id=='US00130H1059']
        portfolio = ITR.utils.dataframe_to_portfolio(df_portfolio)

        temperature_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2, EScope.S3],
            aggregation_method=PortfolioAggregationMethod.WATS,  # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS.
        )

        try:
            portfolio_data = ITR.utils.get_data(self.data_warehouse, portfolio)
        except RuntimeWarning:
            assert False

        amended_portfolio = temperature_score.calculate(
            data_warehouse=self.data_warehouse, data=portfolio_data, portfolio=portfolio
        )
        print(amended_portfolio[["company_name", "time_frame", "scope", "temperature_score"]])

    def test_get_projected_value(self):
        company_ids = ["JP3672400003", "FR0000131906"]
        expected_data = pd.DataFrame(
            [
                pd.Series(
                    [
                        115.74084950588227,
                        105.99813573894818,
                        120.95141200733731,
                        118.7703348114596,
                        117.55790976771785,
                        116.35786133711798,
                        115.17006317737297,
                        113.99439023591643,
                        112.83071873673698,
                        111.678926167347,
                        110.53889126588427,
                        109.41049400834557,
                        108.29361559595033,
                        107.18813844263339,
                        106.09394616266545,
                        105.01092355839985,
                        103.93895660814442,
                        102.87793245415726,
                        101.82773939076483,
                        100.78826685260155,
                        99.75940540296934,
                        98.74104672231607,
                        97.73308359683148,
                        96.73540990715966,
                        95.74792061722674,
                        94.77051176318248,
                        93.80308044245491,
                        92.8455248029167,
                        91.89774403216197,
                        90.95963834689273,
                        90.03110898241354,
                        89.11205818223351,
                    ],
                    name="JP3672400003",
                    dtype="pint[g CO2/pkm]",
                ),
                pd.Series(
                    [
                        208.66666666666666,
                        203.33333333333334,
                        201.27646149224782,
                        200.2558413373124,
                        199.24039647954257,
                        198.23010067636352,
                        197.2249278182697,
                        196.22485192814995,
                        195.2298471606163,
                        194.23988780133593,
                        193.25494826636663,
                        192.2750031014957,
                        191.30002698158208,
                        190.32999470990177,
                        189.3648812174969,
                        188.4046615625276,
                        187.44931092962753,
                        186.49880462926265,
                        185.55311809709292,
                        184.61222689333783,
                        183.67610670214444,
                        182.7447333309592,
                        181.81808270990254,
                        180.89613089114707,
                        179.9788540482985,
                        179.06622847577995,
                        178.15823058821925,
                        177.25483691983953,
                        176.35602412385273,
                        175.46176897185615,
                        174.5720483532323,
                        173.68683927455157,
                    ],
                    name="FR0000131906",
                    dtype="pint[g CO2/pkm]",
                ),
            ],
            index=company_ids,
        )
        expected_data.columns = range(
            TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
            TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1,
        )
        trajectories = self.template_company_data.get_company_projected_trajectories(company_ids)
        assert_pint_frame_equal(self, trajectories.loc[:, EScope.S1S2S3, :], expected_data, places=2)

    def test_get_benchmark(self):
        # This test is a hot mess: the data are series of corp EI trajectories, which are company-specific
        # benchmarks are sector/region specific, and guide temperature scores, but we wouldn't expect
        # an exact match between the two except when the company's data was generated from the benchmark
        # (as test.utils.gen_company_data does).
        expected_data = pd.DataFrame(
            [
                pd.Series(
                    [
                        643.76,
                        657.02,
                        670.55,
                        684.36,
                        698.45,
                        712.8400000000001,
                        727.52,
                        695.68,
                        665.2399999999999,
                        636.13,
                        608.29,
                        581.67,
                        493.3,
                        405.16,
                        317.25,
                        229.56,
                        142.11,
                        126.9,
                        111.77,
                        96.716,
                        81.744,
                        66.85,
                        56.08,
                        45.36,
                        34.688,
                        24.065,
                        13.49,
                        10.792000000000002,
                        8.094,
                        5.396000000000001,
                        2.6980000000000004,
                        0.0,
                    ],
                    name="JP3672400003",
                    dtype="pint[g CO2e/pkm]",
                ),
                pd.Series(
                    [
                        96.054,
                        99.0,
                        98.818,
                        98.63599999999998,
                        98.455,
                        98.274,
                        98.093,
                        82.997,
                        67.424,
                        51.363,
                        34.804,
                        17.734,
                        14.493,
                        11.261,
                        8.037,
                        4.821,
                        1.6139999999999999,
                        1.437,
                        1.261,
                        1.087,
                        0.913,
                        0.739,
                        0.592,
                        0.444,
                        0.296,
                        0.148,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    name="FR0000131906",
                    dtype="pint[g CO2e/pkm]",
                ),
            ],
            index=self.company_ids,
        )
        expected_data.columns = list(
            range(
                TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1,
            )
        )
        benchmarks = self.base_EI_bm.get_SDA_intensity_benchmarks(self.company_info_at_base_year)
        assert_pint_frame_equal(self, benchmarks.loc[:, EScope.S1S2S3, :], expected_data)

    def test_get_projected_production(self):
        expected_data_2025 = pd.Series(
            [
                Q_(1225696.3450837212, "Mpkm"),
                Q_(454546350120.7889, "pkm"),
            ],
            index=self.company_ids,
            name=2025,
        )
        production = self.base_production_bm.get_company_projected_production(self.company_info_at_base_year)[2025]
        # FIXME: this test is broken until we fix data for POSCO
        # Tricky beacuse productions are all sorts of types, not a PintArray, and when filled with uncertainties...this is not technically a pint series array!
        assert_pint_series_equal(
            self,
            production[:, EScope.S1S2S3].map(lambda x: Q_(ITR.nominal_values(x.m), x.u)),
            expected_data_2025,
            places=2,
        )

    def test_get_company_data(self):
        # "JP3672400003" and "FR0000131906" are Automobile manufacturers
        companies = self.data_warehouse.get_preprocessed_company_data(self.company_ids)
        # The above returns ICompanyAggregate data for Scopes S3, S1S2, and S1S2S3 so we select only the S1S2 data...
        company1_s1s2 = companies[1]
        company2_s1s2 = companies[4]
        self.assertEqual(company1_s1s2.company_name, "Nissan Motor Co Ltd")
        self.assertEqual(company2_s1s2.company_name, "Renault SAS")
        self.assertEqual(company1_s1s2.company_id, "JP3672400003")
        self.assertEqual(company2_s1s2.company_id, "FR0000131906")
        self.assertAlmostEqual(
            ITR.nominal_values(company1_s1s2.ghg_s1s2.m_as("Mt CO2")),
            2.643932,
            places=6,
        )
        self.assertAlmostEqual(
            ITR.nominal_values(company1_s1s2.cumulative_budget.m_as("Mt CO2")),
            812.3091689647264,
            places=6,
        )
        self.assertAlmostEqual(
            ITR.nominal_values(company1_s1s2.cumulative_target.m_as("Mt CO2")),
            22.713585884633098,
            places=7,
        )
        self.assertAlmostEqual(
            ITR.nominal_values(company1_s1s2.cumulative_trajectory.m_as("Mt CO2")),
            54.15540844783113,
            places=6,
        )
        self.assertAlmostEqual(
            ITR.nominal_values(company2_s1s2.ghg_s1s2.m_as("Mt CO2")),
            1.4639519700000003,
            places=6,
        )
        self.assertAlmostEqual(
            ITR.nominal_values(company2_s1s2.cumulative_budget.m_as("Mt CO2")),
            34.42256021524011,
            places=6,
        )
        self.assertAlmostEqual(
            ITR.nominal_values(company2_s1s2.cumulative_target.m_as("Mt CO2")),
            14.489943706718206,
            places=7,
        )
        self.assertAlmostEqual(
            ITR.nominal_values(company2_s1s2.cumulative_trajectory.m_as("Mt CO2")),
            43.58371267950996,
            places=6,
        )

    def test_get_value(self):
        expected_data = pd.Series(
            [70480260000.0, 46699200000.0],
            index=pd.Index(self.company_ids, name="company_id"),
            name="company_revenue",
        ).astype("pint[EUR]")
        pd.testing.assert_series_equal(
            asPintSeries(
                self.template_company_data.get_value(
                    company_ids=self.company_ids,
                    variable_name=ColumnsConfig.COMPANY_REVENUE,
                )
            ),
            expected_data,
        )


if __name__ == "__main__":
    test = TestTemplateProviderV2()
    test.setUp()
    test.test_temp_score()
    test.test_target_projections()
    test.test_get_company_data()
