import unittest
import json
import os
import re
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

import ITR
from ITR.interfaces import EScope, ETimeFrames
from ITR.interfaces import (
    ICompanyData,
    ICompanyEIProjectionsScopes,
    ICompanyEIProjections,
    ICompanyEIProjection,
)
from ITR.interfaces import (
    IProductionBenchmarkScopes,
    IEIBenchmarkScopes,
    PortfolioCompany,
    ITargetData,
)

from ITR.data.base_providers import (
    BaseCompanyDataProvider,
    BaseProviderProductionBenchmark,
    BaseProviderIntensityBenchmark,
    EITargetProjector,
    EITrajectoryProjector,
)

from ITR.data.data_warehouse import DataWarehouse
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod

from pint import Quantity
from ITR.data.osc_units import ureg, Q_, PA_, asPintSeries, asPintDataFrame
from ITR.configs import ColumnsConfig

from utils import gen_company_data, DequantifyQuantity, assert_pint_series_equal

import plotly.express as px
import plotly.graph_objects as go


def print_expected(target_df, company_data):
    target_indexes = target_df.index.to_list()
    for c in company_data:
        key = f"{c.company_id} - {c.company_name}"
        suffix = c.company_name.split(" ")[-1].lower()
        print(
            f"""
        expected_{suffix} = pd.Series(PA_({round(target_df[key].pint.m, 3).to_list()}, dtype='{target_df[key].dtype}'),
                                index=range(2019,2051))
"""
        )


# https://stackoverflow.com/a/62853540/1291237
from plotly.subplots import make_subplots

subfig = make_subplots(specs=[[{"secondary_y": True}]])

# # create two independent figures with px.line each containing data from multiple columns
# fig = px.line(df, y=df.filter(regex="Linear").columns, render_mode="webgl",)
# fig2 = px.line(df, y=df.filter(regex="Log").columns, render_mode="webgl",)
#
# fig2.update_traces(yaxis="y2")
#
# subfig.add_traces(fig.data + fig2.data)
# subfig.layout.xaxis.title="Time"
# subfig.layout.yaxis.title="Linear Y"
# subfig.layout.yaxis2.type="log"
# subfig.layout.yaxis2.title="Log Y"
# # recoloring is necessary otherwise lines from fig und fig2 would share each color
# # e.g. Linear-, Log- = blue; Linear+, Log+ = red... we don't want this
# subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
# subfig.show()


# For this test case, we prime the pump with known-aligned emissions intensities.
# We can then construct companies that have some passing resemplemnce to these, and then verify alignment/non-alignment
# as expected according to how we tweak them company by company.


class TestTargets(unittest.TestCase):
    """
    Testdifferent flavours of emission intensity benchmarks
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.projector = EITargetProjector()
        self.base_company_data = BaseCompanyDataProvider([])
        # All benchmarks use OECM Production for Production
        self.benchmark_prod_json = os.path.join(
            self.root, "inputs", "json", "benchmark_production_OECM.json"
        )
        # Each EI benchmark is particular to its own construction
        # self.benchmark_EI_OECM_PC = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM_PC.json")
        self.benchmark_EI_OECM_S3 = os.path.join(
            self.root, "inputs", "json", "benchmark_EI_OECM_S3.json"
        )
        # self.benchmark_EI_TPI = os.path.join(self.root, "inputs", "json", "benchmark_EI_TPI_2_degrees.json")
        # self.benchmark_EI_TPI_below_2 = os.path.join(self.root, "inputs", "json",
        #                                              "benchmark_EI_TPI_below_2_degrees.json")
        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(
            production_benchmarks=prod_bms
        )

        # OECM (S3)
        with open(self.benchmark_EI_OECM_S3) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
        self.OECM_EI_S3_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

    def gen_company_variation(
        self,
        company_name,
        company_id,
        region,
        sector,
        base_production,
        ei_df_t,
        ei_multiplier,
        ei_offset,
        ei_nz_year,
        ei_max_negative=None,
    ) -> ICompanyData:
        year_list = [2019, 2025, 2030, 2035, 2040, 2045, 2050]
        # the last slice(None) gives us all scopes to index against
        bm_ei_t = ei_df_t.loc[year_list, (sector, region, slice(None))]

        # We set intensities to be the wonky things
        company_data = gen_company_data(
            company_name,
            company_id,
            region,
            sector,
            base_production,
            bm_ei_t * ei_multiplier + ei_offset,
            ei_nz_year,
            ei_max_negative,
        )
        # By default, gen_company_data sets targets, but we want to set intensities...
        company_data.projected_intensities = company_data.projected_targets
        # And we will test how targets expressed in target_data get interpreted/extrapolated
        company_data.projected_targets = None
        return company_data

    def test_target_netzero(self):
        # Company AG is over-budget with its intensity projections, targeting netzero intensity by 2040 via targets
        company_ag = self.gen_company_variation(
            "Company AG",
            "US0079031078",
            "North America",
            "Electricity Utilities",
            Q_(10, "TWh"),
            self.OECM_EI_S3_bm._EI_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )
        # Company AH is same as Company AG, but targeting netzero absolute by 2040 via targets
        company_ah = self.gen_company_variation(
            "Company AH",
            "US00724F1012",
            "North America",
            "Electricity Utilities",
            Q_(10, "TWh"),
            self.OECM_EI_S3_bm._EI_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        # Company AI is same as Company AG, but targeting 50% intensity reduction by 2030 and netzero by targets in 2040
        company_ai = self.gen_company_variation(
            "Company AI",
            "US00130H1059",
            "North America",
            "Electricity Utilities",
            Q_(10.0, "TWh"),
            self.OECM_EI_S3_bm._EI_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        # Company AJ is same as Company AG, but targeting 50% absolute reduction by 2030 and netzero by targets in 2040
        company_aj = self.gen_company_variation(
            "Company AJ",
            "FR0000125338",
            "North America",
            "Electricity Utilities",
            Q_(10.0, "TWh"),
            self.OECM_EI_S3_bm._EI_df_t,
            1.0,
            ei_offset=Q_(100, "kg CO2/MWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        company_data = [company_ag, company_ah, company_ai, company_aj]
        company_dict = {
            field: [getattr(c, field) for c in company_data]
            for field in [
                ColumnsConfig.BASE_YEAR_PRODUCTION,
                ColumnsConfig.GHG_SCOPE12,
                ColumnsConfig.SECTOR,
                ColumnsConfig.REGION,
            ]
        }
        company_index = [c.company_id for c in company_data]
        company_sector_region_info = pd.DataFrame(
            company_dict, pd.Index(company_index, name="company_id")
        )
        company_sector_region_info[ColumnsConfig.SCOPE] = [EScope.S1S2] * len(
            company_sector_region_info
        )
        bm_production_data = self.base_production_bm.get_company_projected_production(
            company_sector_region_info
        )

        intensity = (
            company_ag.ghg_s1s2 + company_ag.ghg_s3
        ) / company_ag.base_year_production
        absolute = company_ag.ghg_s1s2 + company_ag.ghg_s3

        ei_df_t = self.OECM_EI_S3_bm._EI_df_t

        target_ag_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2040,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.95,
            }
        )
        company_ag.target_data = [target_ag_0]
        company_ag.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ag,
            bm_production_data.loc[(company_ag.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_ah_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2040,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.95,
            }
        )
        company_ah.target_data = [target_ah_0]
        company_ah.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ah,
            bm_production_data.loc[(company_ah.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_ai_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ai_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2030,
                "target_end_year": 2040,
                "target_base_year_qty": 4.89032773,
                "target_base_year_unit": "Mt CO2",
                "target_reduction_pct": 0.95,
            }
        )
        company_ai.target_data = [target_ai_0, target_ai_1]
        company_ai.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ai,
            bm_production_data.loc[(company_ai.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_aj_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_aj_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2030,
                "target_end_year": 2040,
                "target_base_year_qty": 0.181292635,
                "target_base_year_unit": "t CO2/MWh",
                "target_reduction_pct": 0.95,
            }
        )
        company_aj.target_data = [target_aj_0, target_aj_1]
        company_aj.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_aj,
            bm_production_data.loc[(company_aj.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        plot_dict = {
            # "Trajectory": (co_productions * co_ei_trajectory).cumsum(),
            f"{c.company_id} - {c.company_name}": self.base_company_data._convert_projections_to_series(
                c, "projected_targets"
            )
            for c in company_data
        }
        target_df = pd.DataFrame(plot_dict)

        fig = px.line(
            target_df.pint.dequantify().droplevel(1, axis=1),
            y=[k for k in plot_dict.keys()],
            labels={
                "index": "Year",
                "value": f"{intensity.u:~P}",
                "variable": "test_target_netzero",
            },
        )
        # fig.show()

        # print_expected(target_df, [company_ag, company_ah, company_ai, company_aj])
        expected_ag = pd.Series(
            PA_(
                [
                    0.592,
                    0.527,
                    0.468,
                    0.416,
                    0.37,
                    0.328,
                    0.291,
                    0.257,
                    0.228,
                    0.201,
                    0.177,
                    0.155,
                    0.136,
                    0.119,
                    0.103,
                    0.089,
                    0.076,
                    0.065,
                    0.055,
                    0.046,
                    0.037,
                    0.03,
                    0.023,
                    0.018,
                    0.014,
                    0.01,
                    0.008,
                    0.005,
                    0.004,
                    0.002,
                    0.001,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_ah = pd.Series(
            PA_(
                [
                    0.592,
                    0.513,
                    0.444,
                    0.384,
                    0.332,
                    0.287,
                    0.248,
                    0.205,
                    0.17,
                    0.14,
                    0.115,
                    0.095,
                    0.08,
                    0.067,
                    0.056,
                    0.047,
                    0.039,
                    0.033,
                    0.027,
                    0.022,
                    0.018,
                    0.014,
                    0.011,
                    0.009,
                    0.007,
                    0.005,
                    0.004,
                    0.003,
                    0.002,
                    0.001,
                    0.0,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_ai = pd.Series(
            PA_(
                [
                    0.592,
                    0.556,
                    0.522,
                    0.49,
                    0.46,
                    0.432,
                    0.406,
                    0.381,
                    0.358,
                    0.336,
                    0.315,
                    0.296,
                    0.223,
                    0.168,
                    0.126,
                    0.093,
                    0.069,
                    0.052,
                    0.038,
                    0.027,
                    0.019,
                    0.012,
                    0.009,
                    0.007,
                    0.005,
                    0.004,
                    0.003,
                    0.002,
                    0.001,
                    0.001,
                    0.0,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_aj = pd.Series(
            PA_(
                [
                    0.592,
                    0.541,
                    0.495,
                    0.453,
                    0.414,
                    0.378,
                    0.346,
                    0.304,
                    0.267,
                    0.234,
                    0.205,
                    0.18,
                    0.141,
                    0.11,
                    0.085,
                    0.066,
                    0.05,
                    0.038,
                    0.028,
                    0.021,
                    0.014,
                    0.009,
                    0.007,
                    0.005,
                    0.004,
                    0.003,
                    0.002,
                    0.002,
                    0.001,
                    0.001,
                    0.0,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ag, "projected_targets"
            ),
            expected_ag,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ah, "projected_targets"
            ),
            expected_ah,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ai, "projected_targets"
            ),
            expected_ai,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_aj, "projected_targets"
            ),
            expected_aj,
            places=3,
        )

    def test_target_2030(self):
        ei_df_t = self.OECM_EI_S3_bm._EI_df_t

        # Company AG is over-budget with its intensity projections, targeting 50% reduction by 2030, another 50% by 2040, NZ by 2050
        company_ag = self.gen_company_variation(
            "Company AG",
            "US0079031078",
            "North America",
            "Electricity Utilities",
            Q_(10, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )
        # Company AH is same as Company AG, but targeting 50% absolute reduction by 2030, , another 50% by 2040, NZ by 2050
        company_ah = self.gen_company_variation(
            "Company AH",
            "US00724F1012",
            "North America",
            "Electricity Utilities",
            Q_(10, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        # Company AI is same as Company AG, but targeting 30% absolute reduction by 2030, 50% intensity reduction by 2040, NZ by 2050
        company_ai = self.gen_company_variation(
            "Company AI",
            "US00130H1059",
            "North America",
            "Electricity Utilities",
            Q_(10.0, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        # Company AJ is same as Company AG, but targeting 30% intensity reduction by 2030, 50% absolute reduction by 2040, NZ by 2050
        company_aj = self.gen_company_variation(
            "Company AJ",
            "FR0000125338",
            "North America",
            "Electricity Utilities",
            Q_(10.0, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "kg CO2/MWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        company_data = [company_ag, company_ah, company_ai, company_aj]
        company_dict = {
            field: [getattr(c, field) for c in company_data]
            for field in [
                ColumnsConfig.BASE_YEAR_PRODUCTION,
                ColumnsConfig.GHG_SCOPE12,
                ColumnsConfig.SECTOR,
                ColumnsConfig.REGION,
            ]
        }
        company_index = [c.company_id for c in company_data]
        company_sector_region_info = pd.DataFrame(
            company_dict, pd.Index(company_index, name="company_id")
        )
        company_sector_region_info[ColumnsConfig.SCOPE] = [EScope.S1S2] * len(
            company_sector_region_info
        )
        bm_production_data = self.base_production_bm.get_company_projected_production(
            company_sector_region_info
        )

        intensity = (
            company_ag.ghg_s1s2 + company_ag.ghg_s3
        ) / company_ag.base_year_production
        absolute = company_ag.ghg_s1s2 + company_ag.ghg_s3

        target_ag_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ag_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2030,
                "target_end_year": 2040,
                "target_base_year_qty": intensity.m / 2.0,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        company_ag.target_data = [target_ag_0, target_ag_1]
        company_ag.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ag,
            bm_production_data.loc[(company_ag.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_ah_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ah_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2030,
                "target_end_year": 2040,
                "target_base_year_qty": absolute.m / 2.0,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        company_ah.target_data = [target_ah_0, target_ah_1]
        company_ah.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ah,
            bm_production_data.loc[(company_ah.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_ai_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.3,
            }
        )
        target_ai_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2030,
                "target_end_year": 2040,
                "target_base_year_qty": 4.89032773,
                "target_base_year_unit": "Mt CO2",
                "target_reduction_pct": 0.5,
            }
        )
        company_ai.target_data = [target_ai_0, target_ai_1]
        company_ai.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ai,
            bm_production_data.loc[(company_ai.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_aj_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_aj_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2030,
                "target_end_year": 2040,
                "target_base_year_qty": 0.181292635,
                "target_base_year_unit": "t CO2/MWh",
                "target_reduction_pct": 0.5,
            }
        )
        company_aj.target_data = [target_aj_0, target_aj_1]
        company_aj.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_aj,
            bm_production_data.loc[(company_aj.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        plot_dict = {
            # "Trajectory": (co_productions * co_ei_trajectory).cumsum(),
            f"{c.company_id} - {c.company_name}": self.base_company_data._convert_projections_to_series(
                c, "projected_targets"
            )
            for c in company_data
        }
        target_df = pd.DataFrame(plot_dict)

        fig = px.line(
            target_df.pint.dequantify().droplevel(1, axis=1),
            y=[k for k in plot_dict.keys()],
            labels={
                "index": "Year",
                "value": f"{intensity.u:~P}",
                "variable": "test_target_2030",
            },
        )
        # fig.show()

        # print_expected(target_df, [company_ag, company_ah, company_ai, company_aj])
        expected_ag = pd.Series(
            PA_(
                [
                    0.592,
                    0.556,
                    0.522,
                    0.49,
                    0.46,
                    0.432,
                    0.406,
                    0.381,
                    0.358,
                    0.336,
                    0.315,
                    0.296,
                    0.276,
                    0.258,
                    0.24,
                    0.224,
                    0.209,
                    0.195,
                    0.182,
                    0.17,
                    0.159,
                    0.148,
                    0.115,
                    0.089,
                    0.068,
                    0.051,
                    0.038,
                    0.027,
                    0.018,
                    0.011,
                    0.005,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_ah = pd.Series(
            PA_(
                [
                    0.592,
                    0.541,
                    0.495,
                    0.453,
                    0.414,
                    0.378,
                    0.346,
                    0.304,
                    0.267,
                    0.234,
                    0.205,
                    0.18,
                    0.162,
                    0.146,
                    0.131,
                    0.118,
                    0.106,
                    0.098,
                    0.09,
                    0.083,
                    0.077,
                    0.071,
                    0.055,
                    0.043,
                    0.033,
                    0.025,
                    0.018,
                    0.013,
                    0.009,
                    0.005,
                    0.002,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_ai = pd.Series(
            PA_(
                [
                    0.592,
                    0.573,
                    0.555,
                    0.537,
                    0.52,
                    0.503,
                    0.487,
                    0.472,
                    0.457,
                    0.442,
                    0.428,
                    0.414,
                    0.361,
                    0.314,
                    0.273,
                    0.238,
                    0.207,
                    0.185,
                    0.165,
                    0.147,
                    0.132,
                    0.118,
                    0.091,
                    0.071,
                    0.054,
                    0.041,
                    0.03,
                    0.021,
                    0.014,
                    0.009,
                    0.004,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_aj = pd.Series(
            PA_(
                [
                    0.592,
                    0.541,
                    0.495,
                    0.453,
                    0.414,
                    0.378,
                    0.346,
                    0.304,
                    0.267,
                    0.234,
                    0.205,
                    0.18,
                    0.168,
                    0.157,
                    0.147,
                    0.137,
                    0.128,
                    0.119,
                    0.111,
                    0.104,
                    0.097,
                    0.091,
                    0.07,
                    0.054,
                    0.042,
                    0.031,
                    0.023,
                    0.016,
                    0.011,
                    0.007,
                    0.003,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )

        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ag, "projected_targets"
            ),
            expected_ag,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ah, "projected_targets"
            ),
            expected_ah,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ai, "projected_targets"
            ),
            expected_ai,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_aj, "projected_targets"
            ),
            expected_aj,
            places=3,
        )

    def test_target_overlaps(self):
        ei_df_t = self.OECM_EI_S3_bm._EI_df_t

        # Company AG is over-budget with its intensity projections, targeting 50% reduction by 2030, another 50% by 2040, NZ by 2050
        company_ag = self.gen_company_variation(
            "Company AG",
            "US0079031078",
            "North America",
            "Electricity Utilities",
            Q_(10, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )
        # Company AH is same as Company AG, but targeting 50% absolute reduction by 2030, , another 50% by 2040, NZ by 2050
        company_ah = self.gen_company_variation(
            "Company AH",
            "US00724F1012",
            "North America",
            "Electricity Utilities",
            Q_(10, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        # Company AI is same as Company AG, but targeting 30% absolute reduction by 2030, 50% intensity reduction by 2040, NZ by 2050
        company_ai = self.gen_company_variation(
            "Company AI",
            "US00130H1059",
            "North America",
            "Electricity Utilities",
            Q_(10.0, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        # Company AJ is same as Company AG, but targeting 30% intensity reduction by 2030, 50% absolute reduction by 2040, NZ by 2050
        company_aj = self.gen_company_variation(
            "Company AJ",
            "FR0000125338",
            "North America",
            "Electricity Utilities",
            Q_(10.0, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(100, "kg CO2/MWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )

        company_data = [company_ag, company_ah, company_ai, company_aj]
        company_dict = {
            field: [getattr(c, field) for c in company_data]
            for field in [
                ColumnsConfig.BASE_YEAR_PRODUCTION,
                ColumnsConfig.GHG_SCOPE12,
                ColumnsConfig.SECTOR,
                ColumnsConfig.REGION,
            ]
        }
        company_index = [c.company_id for c in company_data]
        company_sector_region_info = pd.DataFrame(
            company_dict, pd.Index(company_index, name="company_id")
        )
        company_sector_region_info[ColumnsConfig.SCOPE] = [EScope.S1S2] * len(
            company_sector_region_info
        )
        bm_production_data = self.base_production_bm.get_company_projected_production(
            company_sector_region_info
        )

        intensity = (
            company_ag.ghg_s1s2 + company_ag.ghg_s3
        ) / company_ag.base_year_production
        absolute = company_ag.ghg_s1s2 + company_ag.ghg_s3

        target_ag_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ag_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2019,
                "target_end_year": 2040,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.75,
            }
        )
        company_ag.target_data = [target_ag_0, target_ag_1]
        company_ag.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ag,
            bm_production_data.loc[(company_ag.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_ah_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ah_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2019,
                "target_end_year": 2040,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.75,
            }
        )
        company_ah.target_data = [target_ah_0, target_ah_1]
        company_ah.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ah,
            bm_production_data.loc[(company_ah.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_ai_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": f"{intensity.u:~P}",
                "target_reduction_pct": 0.3,
            }
        )
        target_ai_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2019,
                "target_end_year": 2040,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": "Mt CO2",
                "target_reduction_pct": 0.75,
            }
        )
        company_ai.target_data = [target_ai_0, target_ai_1]
        company_ai.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ai,
            bm_production_data.loc[(company_ai.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        target_aj_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "absolute",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": absolute.m,
                "target_base_year_unit": f"{absolute.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_aj_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2030,
                "target_base_year": 2019,
                "target_end_year": 2040,
                "target_base_year_qty": intensity.m,
                "target_base_year_unit": "t CO2/MWh",
                "target_reduction_pct": 0.75,
            }
        )
        company_aj.target_data = [target_aj_0, target_aj_1]
        company_aj.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_aj,
            bm_production_data.loc[(company_aj.company_id, EScope.S1S2)],
            ei_df_t=ei_df_t,
        )

        plot_dict = {
            # "Trajectory": (co_productions * co_ei_trajectory).cumsum(),
            f"{c.company_id} - {c.company_name}": self.base_company_data._convert_projections_to_series(
                c, "projected_targets"
            )
            for c in company_data
        }
        target_df = pd.DataFrame(plot_dict)

        fig = px.line(
            target_df.pint.dequantify().droplevel(1, axis=1),
            y=[k for k in plot_dict.keys()],
            labels={
                "index": "Year",
                "value": f"{intensity.u:~P}",
                "variable": "test_target_overlaps",
            },
        )
        # fig.show()

        # print_expected(target_df, company_data)
        expected_ag = pd.Series(
            PA_(
                [
                    0.592,
                    0.556,
                    0.522,
                    0.49,
                    0.46,
                    0.432,
                    0.406,
                    0.381,
                    0.358,
                    0.336,
                    0.315,
                    0.296,
                    0.276,
                    0.258,
                    0.24,
                    0.224,
                    0.209,
                    0.195,
                    0.182,
                    0.17,
                    0.159,
                    0.148,
                    0.115,
                    0.089,
                    0.068,
                    0.051,
                    0.038,
                    0.027,
                    0.018,
                    0.011,
                    0.005,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_ah = pd.Series(
            PA_(
                [
                    0.592,
                    0.541,
                    0.495,
                    0.453,
                    0.414,
                    0.378,
                    0.346,
                    0.304,
                    0.267,
                    0.234,
                    0.205,
                    0.18,
                    0.162,
                    0.146,
                    0.131,
                    0.118,
                    0.106,
                    0.098,
                    0.09,
                    0.083,
                    0.077,
                    0.071,
                    0.055,
                    0.043,
                    0.033,
                    0.025,
                    0.018,
                    0.013,
                    0.009,
                    0.005,
                    0.002,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_ai = pd.Series(
            PA_(
                [
                    0.592,
                    0.573,
                    0.555,
                    0.537,
                    0.52,
                    0.503,
                    0.487,
                    0.472,
                    0.457,
                    0.442,
                    0.428,
                    0.414,
                    0.343,
                    0.284,
                    0.235,
                    0.194,
                    0.161,
                    0.137,
                    0.116,
                    0.099,
                    0.084,
                    0.071,
                    0.055,
                    0.043,
                    0.033,
                    0.025,
                    0.018,
                    0.013,
                    0.009,
                    0.005,
                    0.002,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )
        expected_aj = pd.Series(
            PA_(
                [
                    0.592,
                    0.541,
                    0.495,
                    0.453,
                    0.414,
                    0.378,
                    0.346,
                    0.304,
                    0.267,
                    0.234,
                    0.205,
                    0.18,
                    0.177,
                    0.173,
                    0.17,
                    0.167,
                    0.163,
                    0.16,
                    0.157,
                    0.154,
                    0.151,
                    0.148,
                    0.115,
                    0.089,
                    0.068,
                    0.051,
                    0.038,
                    0.027,
                    0.018,
                    0.011,
                    0.005,
                    0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )

        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ag, "projected_targets"
            ),
            expected_ag,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ah, "projected_targets"
            ),
            expected_ah,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_ai, "projected_targets"
            ),
            expected_ai,
            places=3,
        )
        assert_pint_series_equal(
            self,
            self.base_company_data._convert_projections_to_series(
                company_aj, "projected_targets"
            ),
            expected_aj,
            places=3,
        )

    def test_different_starting_dates_intensity(self):
        ei_df_t = self.OECM_EI_S3_bm._EI_df_t

        # For a company with specific targets, test the effects of under- or over-acheiving those targets
        # durig the first few years of the start-date of the target

        def compute_scope_targets(c, projection_type, projection_years):
            result = (
                self.base_company_data._convert_projections_to_series(
                    c, projection_type, EScope.S1S2
                )
                .loc[projection_years]
                .add(
                    self.base_company_data._convert_projections_to_series(
                        c, projection_type, EScope.S3
                    ).loc[projection_years]
                )
            )
            return result

        company_oecm = self.gen_company_variation(
            "OECM Aligned",
            "NA-EU-OECM-ALIGNED",
            "North America",
            "Electricity Utilities",
            Q_(10, "TWh"),
            ei_df_t,
            1.0,
            ei_offset=Q_(0, "g CO2/kWh"),
            ei_nz_year=2050,
            ei_max_negative=Q_(-1, "g CO2/kWh"),
        )
        company_oecm.projected_targets = company_oecm.projected_intensities

        base_production = Q_("10 TWh")
        base_emissions_s1s2 = Q_("1 Mt CO2e")
        base_emissions_s3 = Q_("4 Mt CO2e")
        company_dict_ag = {
            "company_name": "Company AG",
            "company_id": "US0079031078",
            "region": "North America",
            "sector": "Electricity Utilities",
            "base_year_production": base_production,
            "ghg_s1s2": base_emissions_s1s2,
            "ghg_s3": base_emissions_s3,
        }

        historic_productions_ag = [
            {"year": year, "value": base_production + (year - 2015) * Q_("0 GWh")}
            for year in range(2015, 2021)
        ]
        historic_emissions_s1s2_ag = [
            {
                "year": year,
                "value": base_emissions_s1s2 - (year - 2015) * Q_("50 kt CO2e"),
            }
            for year in range(2015, 2021)
        ]
        historic_emissions_s3_ag = [
            {
                "year": year,
                "value": base_emissions_s3 - (year - 2015) * Q_("100 kt CO2e"),
            }
            for year in range(2015, 2021)
        ]
        company_dict_ag["historic_data"] = {
            "productions": historic_productions_ag,
            "emissions": {
                "S1": [],
                "S2": [],
                "S1S2": historic_emissions_s1s2_ag,
                "S3": historic_emissions_s3_ag,
                "S1S2S3": [],
            },
            "emissions_intensities": {
                "S1": [],
                "S2": [],
                "S1S2": [
                    {
                        "year": em_dict["year"],
                        "value": em_dict["value"] / prod_dict["value"],
                    }
                    for em_dict, prod_dict in zip(
                        historic_emissions_s1s2_ag, historic_productions_ag
                    )
                ],
                "S3": [
                    {
                        "year": em_dict["year"],
                        "value": em_dict["value"] / prod_dict["value"],
                    }
                    for em_dict, prod_dict in zip(
                        historic_emissions_s3_ag, historic_productions_ag
                    )
                ],
                "S1S2S3": [],
            },
        }
        company_ag = ICompanyData.parse_obj(company_dict_ag)

        company_dict_ah = company_dict_ag.copy()
        company_dict_ah["company_name"] = "Company AH"
        company_dict_ah["company_id"] = "US00724F1012"
        historic_productions_ah = [
            {"year": year, "value": base_production + (year - 2015) * Q_("0 GWh")}
            for year in range(2015, 2025)
        ]
        historic_emissions_s1s2_ah = [
            {
                "year": year,
                "value": base_emissions_s1s2 - (year - 2015) * Q_("50 kt CO2e"),
            }
            for year in range(2015, 2025)
        ]
        historic_emissions_s3_ah = [
            {
                "year": year,
                "value": base_emissions_s3 - (year - 2015) * Q_("100 kt CO2e"),
            }
            for year in range(2015, 2025)
        ]
        company_dict_ah["historic_data"] = {
            "productions": historic_productions_ah,
            "emissions": {
                "S1": [],
                "S2": [],
                "S1S2": historic_emissions_s1s2_ah,
                "S3": historic_emissions_s3_ah,
                "S1S2S3": [],
            },
            "emissions_intensities": {
                "S1": [],
                "S2": [],
                "S1S2": [
                    {
                        "year": em_dict["year"],
                        "value": em_dict["value"] / prod_dict["value"],
                    }
                    for em_dict, prod_dict in zip(
                        historic_emissions_s1s2_ah, historic_productions_ah
                    )
                ],
                "S3": [
                    {
                        "year": em_dict["year"],
                        "value": em_dict["value"] / prod_dict["value"],
                    }
                    for em_dict, prod_dict in zip(
                        historic_emissions_s3_ah, historic_productions_ah
                    )
                ],
                "S1S2S3": [],
            },
        }
        company_ah = ICompanyData.parse_obj(company_dict_ah)

        company_dict_ai = company_dict_ag.copy()
        company_dict_ai["company_name"] = "Company AI"
        company_dict_ai["company_id"] = "US00130H1059"
        historic_productions_ai = [
            {"year": year, "value": base_production + (year - 2015) * Q_("0 GWh")}
            for year in range(2015, 2025)
        ]
        historic_emissions_s1s2_ai = [
            {
                "year": year,
                "value": base_emissions_s1s2 - (year - 2015) * Q_("100 kt CO2e"),
            }
            for year in range(2015, 2025)
        ]
        historic_emissions_s3_ai = [
            {
                "year": year,
                "value": base_emissions_s3 - (year - 2015) * Q_("200 kt CO2e"),
            }
            for year in range(2015, 2025)
        ]
        company_dict_ai["historic_data"] = {
            "productions": historic_productions_ai,
            "emissions": {
                "S1": [],
                "S2": [],
                "S1S2": historic_emissions_s1s2_ai,
                "S3": historic_emissions_s3_ai,
                "S1S2S3": [],
            },
            "emissions_intensities": {
                "S1": [],
                "S2": [],
                "S1S2": [
                    {
                        "year": em_dict["year"],
                        "value": em_dict["value"] / prod_dict["value"],
                    }
                    for em_dict, prod_dict in zip(
                        historic_emissions_s1s2_ai, historic_productions_ai
                    )
                ],
                "S3": [
                    {
                        "year": em_dict["year"],
                        "value": em_dict["value"] / prod_dict["value"],
                    }
                    for em_dict, prod_dict in zip(
                        historic_emissions_s3_ai, historic_productions_ai
                    )
                ],
                "S1S2S3": [],
            },
        }
        company_ai = ICompanyData.parse_obj(company_dict_ai)

        company_data = [company_ag, company_ah, company_ai, company_oecm]
        company_dict = {
            field: [getattr(c, field) for c in company_data]
            for field in [
                ColumnsConfig.BASE_YEAR_PRODUCTION,
                ColumnsConfig.GHG_SCOPE12,
                ColumnsConfig.GHG_SCOPE3,
                ColumnsConfig.SECTOR,
                ColumnsConfig.REGION,
            ]
        }
        company_index = [c.company_id for c in company_data]
        company_sector_region = pd.DataFrame(
            company_dict, pd.Index(company_index, name="company_id")
        )
        company_sector_region[ColumnsConfig.SCOPE] = [EScope.AnyScope] * len(
            company_sector_region
        )
        bm_production_data = self.base_production_bm.get_company_projected_production(
            company_sector_region
        )

        intensity_s1s2 = company_ag.ghg_s1s2 / company_ag.base_year_production
        intensity_s3 = company_ag.ghg_s3 / company_ag.base_year_production

        target_ag_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2015,
                "target_end_year": 2030,
                "target_base_year_qty": intensity_s1s2.m,
                "target_base_year_unit": f"{intensity_s1s2.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ag_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S3,
                "target_start_year": 2020,
                "target_base_year": 2015,
                "target_end_year": 2030,
                "target_base_year_qty": intensity_s3.m,
                "target_base_year_unit": f"{intensity_s3.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        company_ag.target_data = [target_ag_0, target_ag_1]
        company_ag.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ag,
            bm_production_data.loc[(company_ag.company_id, EScope.AnyScope)],
            ei_df_t=ei_df_t,
        )

        # Same, since copied from company_ag...
        # intensity_s1s2 = company_ah.ghg_s1s2 / company_ah.base_year_production
        # intensity_s3 = company_ah.ghg_s3 / company_ah.base_year_production

        target_ah_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2015,
                "target_end_year": 2030,
                "target_base_year_qty": intensity_s1s2.m,
                "target_base_year_unit": f"{intensity_s1s2.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ah_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S3,
                "target_start_year": 2020,
                "target_base_year": 2015,
                "target_end_year": 2030,
                "target_base_year_qty": intensity_s3.m,
                "target_base_year_unit": f"{intensity_s3.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        company_ah.target_data = [target_ah_0, target_ah_1]
        company_ah.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ah,
            bm_production_data.loc[(company_ah.company_id, EScope.AnyScope)],
            ei_df_t=ei_df_t,
        )

        target_ai_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2015,
                "target_end_year": 2030,
                "target_base_year_qty": intensity_s1s2.m,
                "target_base_year_unit": f"{intensity_s1s2.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        target_ai_1 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S3,
                "target_start_year": 2020,
                "target_base_year": 2015,
                "target_end_year": 2030,
                "target_base_year_qty": intensity_s3.m,
                "target_base_year_unit": f"{intensity_s3.u:~P}",
                "target_reduction_pct": 0.5,
            }
        )
        company_ai.target_data = [target_ai_0, target_ai_1]
        company_ai.projected_targets = EITargetProjector(
            self.projector.projection_controls
        ).project_ei_targets(
            company_ai,
            bm_production_data.loc[(company_ai.company_id, EScope.AnyScope)],
            ei_df_t=ei_df_t,
        )

        self.base_company_data = BaseCompanyDataProvider(company_data)
        # Since we are not using a Data Warehouse to compute our graphics, we have to do this projection manually, with the benchmark's internal dataframe.
        self.base_company_data._validate_projected_trajectories(
            self.base_company_data._companies, ei_df_t
        )

        co_pp = bm_production_data.droplevel("scope")

        def _pint_cumsum(ser: pd.Series) -> pd.Series:
            return ser.pint.m.cumsum().astype(ser.dtype)

        # fig = px.line(df, x='id', y='value', color='variable')
        cumsum_units = "Mt CO2e"
        target_dict = {
            f"Target: {c.company_id} - {c.company_name}": compute_scope_targets(
                c, "projected_targets", co_pp.columns
            )
            for c in company_data
        }
        target_cumulative = {
            f"TargetCumulative: {c.company_id} - {c.company_name}": co_pp.loc[
                c.company_id
            ]
            .mul(
                target_dict[f"Target: {c.company_id} - {c.company_name}"].loc[
                    co_pp.columns
                ]
            )
            .pint.m_as(cumsum_units)
            .cumsum()
            for c in company_data
        }
        trajectory_dict = {
            f"Trajectory: {c.company_id} - {c.company_name}": compute_scope_targets(
                c, "projected_intensities", co_pp.columns
            )
            for c in company_data
        }
        trajectory_cumulative = {
            f"TrajectoryCumulative: {c.company_id} - {c.company_name}": co_pp.loc[
                c.company_id
            ]
            .mul(
                trajectory_dict[f"Trajectory: {c.company_id} - {c.company_name}"].loc[
                    co_pp.columns
                ]
            )
            .pint.m_as(cumsum_units)
            .cumsum()
            for c in company_data
        }
        target_df = asPintDataFrame(
            pd.concat(
                [
                    pd.DataFrame(target_dict),
                    pd.DataFrame(target_cumulative).astype(f"pint[{cumsum_units}]"),
                    pd.DataFrame(trajectory_dict),
                    pd.DataFrame(trajectory_cumulative).astype(f"pint[{cumsum_units}]"),
                ],
                axis=1,
            )
        )
        dequantified_df = target_df.pint.dequantify().droplevel(1, axis=1)
        # May have uncertainties...or some columns may be Float64
        dequantified_df = dequantified_df.apply(
            lambda col: col
            if col.dtype == "float64"
            else ITR.nominal_values(col).astype(np.float64)
        )
        fig_target = px.line(
            dequantified_df, y=dequantified_df.filter(regex="Target:").columns
        )
        fig_trajectory = px.line(
            dequantified_df, y=dequantified_df.filter(regex="Trajectory:").columns
        )
        fig_trajectory.update_traces(line={"dash": "dash"})
        fig_target_cumulative = px.line(
            dequantified_df, y=dequantified_df.filter(regex="TargetCumulative:").columns
        )
        fig_target_cumulative.update_traces(yaxis="y2")
        fig_trajectory_cumulative = px.line(
            dequantified_df,
            y=dequantified_df.filter(regex="TrajectoryCumulative:").columns,
        )
        fig_trajectory_cumulative.update_traces(yaxis="y2", line={"dash": "dash"})
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        subfig.add_traces(
            fig_target.data
            + fig_trajectory.data
            + fig_target_cumulative.data
            + fig_trajectory_cumulative.data
        )
        # subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
        # subfig.show()

        # fig = px.line(target_df.pint.dequantify().droplevel(1, axis=1), y=[k for k in plot_dict.keys()],
        # labels={'index':'Year', 'value':f"{intensity_s1s2.u:~P}", 'variable':'test_different_starting_dates_intensity'})
        # fig.show()

        self.data_warehouse = DataWarehouse(
            self.base_company_data, self.base_production_bm, self.OECM_EI_S3_bm
        )
        companies = self.data_warehouse.get_preprocessed_company_data(company_index)

        print_expected(
            target_df.filter(regex="Target:").rename(
                columns=lambda x: re.sub("Target: ", "", x)
            ),
            company_data,
        )
        expected_ag = pd.Series(
            PA_(
                [
                    0.44,
                    0.425,
                    0.403,
                    0.382,
                    0.362,
                    0.344,
                    0.326,
                    0.309,
                    0.293,
                    0.278,
                    0.264,
                    0.25,
                    0.221,
                    0.194,
                    0.171,
                    0.15,
                    0.131,
                    0.115,
                    0.1,
                    0.086,
                    0.074,
                    0.064,
                    0.054,
                    0.045,
                    0.038,
                    0.031,
                    0.024,
                    0.018,
                    0.013,
                    0.008,
                    0.004,
                    -0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )

        expected_ah = pd.Series(
            PA_(
                [
                    0.44,
                    0.425,
                    0.41,
                    0.395,
                    0.38,
                    0.365,
                    0.342,
                    0.321,
                    0.301,
                    0.283,
                    0.266,
                    0.25,
                    0.221,
                    0.194,
                    0.171,
                    0.15,
                    0.131,
                    0.115,
                    0.1,
                    0.086,
                    0.074,
                    0.064,
                    0.054,
                    0.045,
                    0.038,
                    0.031,
                    0.024,
                    0.018,
                    0.013,
                    0.008,
                    0.004,
                    -0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )

        expected_ai = pd.Series(
            PA_(
                [
                    0.38,
                    0.35,
                    0.32,
                    0.29,
                    0.26,
                    0.23,
                    0.226,
                    0.221,
                    0.217,
                    0.213,
                    0.209,
                    0.206,
                    0.181,
                    0.16,
                    0.141,
                    0.124,
                    0.108,
                    0.095,
                    0.082,
                    0.071,
                    0.062,
                    0.053,
                    0.045,
                    0.038,
                    0.031,
                    0.025,
                    0.02,
                    0.015,
                    0.011,
                    0.007,
                    0.003,
                    -0.0,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )

        expected_aligned = pd.Series(
            PA_(
                [
                    0.392,
                    0.358,
                    0.324,
                    0.290,
                    0.256,
                    0.222,
                    0.188,
                    0.157,
                    0.125,
                    0.094,
                    0.062,
                    0.031,
                    0.030,
                    0.030,
                    0.030,
                    0.029,
                    0.029,
                    0.025,
                    0.021,
                    0.018,
                    0.014,
                    0.010,
                    0.009,
                    0.009,
                    0.008,
                    0.008,
                    0.007,
                    0.007,
                    0.007,
                    0.006,
                    0.006,
                    0.000,
                ],
                dtype="pint[CO2 * metric_ton / megawatt_hour]",
            ),
            index=range(2019, 2051),
        )

        assert_pint_series_equal(
            self,
            compute_scope_targets(company_ag, "projected_targets", range(2019, 2051)),
            expected_ag,
            places=3,
        )
        assert_pint_series_equal(
            self,
            compute_scope_targets(company_ah, "projected_targets", range(2019, 2051)),
            expected_ah,
            places=3,
        )
        assert_pint_series_equal(
            self,
            compute_scope_targets(company_ai, "projected_targets", range(2019, 2051)),
            expected_ai,
            places=3,
        )
        assert_pint_series_equal(
            self,
            compute_scope_targets(company_oecm, "projected_targets", range(2019, 2051)),
            expected_aligned,
            places=3,
        )


if __name__ == "__main__":
    test = TestTargets()
    test.setUp()
    test.test_target_netzero()
    test.test_target_2030()
    test.test_target_overlaps()
