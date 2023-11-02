import unittest
import json
import os
import pandas as pd
from numpy.testing import assert_array_equal

import ITR
from ITR import data_dir
from ITR.interfaces import EScope, ETimeFrames, IntensityMetric
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
)

from ITR.data.base_providers import (
    BaseCompanyDataProvider,
    BaseProviderProductionBenchmark,
    BaseProviderIntensityBenchmark,
)

from ITR.data.data_warehouse import DataWarehouse
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod

from pint import Quantity
from ITR.data.osc_units import ureg, Q_, PA_

from utils import gen_company_data, DequantifyQuantity

# from utils import ITR_Encoder

# For this test case, we prime the pump with known-aligned emissions intensities.
# We can then construct companies that have some passing resemplemnce to these, and then verify alignment/non-alignment
# as expected according to how we tweak them company by company.

oecm_global_power_utilities_ei = pd.Series(
    [502.0, 291.0, 135.0, 52.0, 24.0, 7.0, 0.0],
    index=[2019, 2025, 2030, 2035, 2040, 2045, 2050],
    dtype="pint[g CO2/kWh]",
)

oecm_na_power_utilities_ei = pd.Series(
    [376.0, 195.0, 40.0, 35.0, 12.0, 7.0, 6.0],
    index=[2019, 2025, 2030, 2035, 2040, 2045, 2050],
    dtype="pint[g CO2/kWh]",
)

oecm_eu_power_utilities_ei = pd.Series(
    [272.0, 182.0, 92.0, 61.0, 24.0, 7.0, 0.0],
    index=[2019, 2025, 2030, 2035, 2040, 2045, 2050],
    dtype="pint[g CO2/kWh]",
)

# All TPI benchmarks are global

tpi_2C_power_utilities_ei = pd.Series(
    [0.608, 0.36, 0.245, 0.151, 0.097, 0.056, 0.04],
    index=[2019, 2025, 2030, 2035, 2040, 2045, 2050],
    dtype="pint[t CO2/MWh]",
)

tpi_below_2C_power_utilities_ei = pd.Series(
    [0.608, 0.33, 0.229, 0.141, 0.072, 0.002, -0.008],
    index=[2019, 2025, 2030, 2035, 2040, 2045, 2050],
    dtype="pint[t CO2/MWh]",
)


def gen_company_variation(
    company_name,
    company_id,
    region,
    sector,
    base_production,
    bm_ei,
    ei_multiplier,
    ei_offset,
    ei_nz_year,
    ei_max_negative=None,
) -> ICompanyData:
    # We set intensities to be the wonky things
    company_data = gen_company_data(
        company_name,
        company_id,
        region,
        sector,
        base_production,
        oecm_na_power_utilities_ei * ei_multiplier + ei_offset,
        ei_nz_year,
        ei_max_negative,
    )
    projected_intensities = company_data.projected_targets
    # We set targets to be the nicely aligned things
    # (which vary due to different sectors/regions/benchmarks)
    company_data = gen_company_data(
        company_name,
        company_id,
        region,
        sector,
        base_production,
        oecm_na_power_utilities_ei,
        2051,
        ei_max_negative,
    )
    company_data.projected_intensities = projected_intensities
    return company_data


# Company AG is over-budget with its intensity projections, but OECM-aligned with their target projections
company_ag = gen_company_variation(
    "Company AG",
    "US0079031078",
    "North America",
    "Electricity Utilities",
    Q_(9.9, "TWh"),
    oecm_na_power_utilities_ei,
    1.0,
    ei_offset=Q_(100, "g CO2/kWh"),
    ei_nz_year=2051,
    ei_max_negative=Q_(-1, "g CO2/kWh"),
)

company_ah = gen_company_variation(
    "Company AH",
    "US00724F1012",
    "North America",
    "Electricity Utilities",
    Q_(1.9, "TWh"),
    oecm_na_power_utilities_ei,
    1.5,
    ei_offset=Q_(0, "g CO2/kWh"),
    ei_nz_year=2031,
)

company_ai = gen_company_variation(
    "Company AI",
    "FR0000125338",
    "Europe",
    "Electricity Utilities",
    Q_(4.9, "PJ"),
    oecm_eu_power_utilities_ei * 0.8,
    1.0,
    ei_offset=Q_(0, "t CO2/MWh"),
    ei_nz_year=2051,
)

# print(json.dumps(company_ag.dict(), cls=DequantifyQuantity, indent=2))


class TestEIBenchmarks(unittest.TestCase):
    """
    Testdifferent flavours of emission intensity benchmarks
    """

    def setUp(self) -> None:
        self.benchmark_prod_json = os.path.join(data_dir, "benchmark_production_OECM.json")
        self.benchmark_EI_OECM_PC = os.path.join(data_dir, "benchmark_EI_OECM_PC.json")
        self.benchmark_EI_TPI = os.path.join(data_dir, "benchmark_EI_TPI_2_degrees.json")
        self.benchmark_EI_TPI_below_2 = os.path.join(data_dir, "benchmark_EI_TPI_below_2_degrees.json")

        # load company data
        self.companies = [company_ag, company_ah, company_ai]
        self.base_company_data = BaseCompanyDataProvider(self.companies)

        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.model_validate(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        # load intensity benchmarks

        # OECM
        with open(self.benchmark_EI_OECM_PC) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.model_validate(parsed_json)
        self.OECM_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        # TPI
        with open(self.benchmark_EI_TPI) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.model_validate(parsed_json)
        self.TPI_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        # TPI below 2
        with open(self.benchmark_EI_TPI_below_2) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.model_validate(parsed_json)
        self.TPI_below_2_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        self.OECM_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.OECM_EI_bm)
        self.TPI_warehouse = DataWarehouse(self.base_company_data, self.base_production_bm, self.TPI_EI_bm)
        self.TPI_below_2_warehouse = DataWarehouse(
            self.base_company_data, self.base_production_bm, self.TPI_below_2_EI_bm
        )

        self.company_ids = ["US0079031078", "US00724F1012", "FR0000125338"]

    def test_all_benchmarks(self):
        # Calculate Temp Scores
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio = []
        for company in self.company_ids:
            portfolio.append(
                PortfolioCompany(
                    company_name=company,
                    company_id=company,
                    investment_value=Q_(100, "USD"),
                    company_isin=company,
                )
            )
        # OECM
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.OECM_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        print(
            scores[
                [
                    "company_name",
                    "company_id",
                    "temperature_score",
                    "trajectory_score",
                    "trajectory_overshoot_ratio",
                    "target_score",
                    "target_overshoot_ratio",
                ]
            ]
        )

        # verify company scores:
        expected = pd.Series([1.76, 1.55, 1.52], dtype="pint[delta_degC]")
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(1.61, ureg.delta_degC), places=2)

        # TPI
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.TPI_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        print(
            scores[
                [
                    "company_name",
                    "company_id",
                    "temperature_score",
                    "trajectory_score",
                    "trajectory_overshoot_ratio",
                    "target_score",
                    "target_overshoot_ratio",
                ]
            ]
        )

        # verify company scores:
        expected = pd.Series([1.85, 1.76, 1.76], dtype="pint[delta_degC]")
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(1.79, ureg.delta_degC), places=2)

        # TPI below 2
        # portfolio data
        portfolio_data = ITR.utils.get_data(self.TPI_below_2_warehouse, portfolio)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        print(
            scores[
                [
                    "company_name",
                    "company_id",
                    "temperature_score",
                    "trajectory_score",
                    "trajectory_overshoot_ratio",
                    "target_score",
                    "target_overshoot_ratio",
                ]
            ]
        )

        # verify company scores:
        expected = pd.Series([1.65, 1.54, 1.53], dtype="pint[delta_degC]")
        assert_array_equal(scores.temperature_score.values, expected)
        # verify that results exist
        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, Q_(1.57, ureg.delta_degC), places=2)


if __name__ == "__main__":
    test = TestEIBenchmarks()
    test.setUp()
    test.test_all_benchmarks()
