import json
import os
from typing import List
from unittest import TestCase

import pandas as pd
import pytest

import ITR
from ITR import data_dir
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data import PA_
from ITR.data.base_providers import (
    BaseProviderIntensityBenchmark,
    BaseProviderProductionBenchmark,
)
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.osc_units import Q_, ureg
from ITR.data.nzdpu_providers import NZDPU_CompanyDataProvider
from ITR.interfaces import (
    EScope,
    ETimeFrames,
    IEIBenchmarkScopes,
    IProductionBenchmarkScopes,
)
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.utils import asPintSeries, requantify_df_from_columns

from utils import assert_pint_frame_equal, assert_pint_series_equal

pd.options.display.width = 999
pd.options.display.max_columns = 99
pd.options.display.min_rows = 30


class NZDPU:
    def __init__(self, ei_filename: str, company_ids: List[str]) -> None:
        root = os.path.dirname(os.path.abspath(__file__))
        self.company_data_path = "~/Downloads/nzdpu_sample_data/nzdpu_data_sample.xlsx"
        self.nzdpu_company_data = NZDPU_CompanyDataProvider(excel_path=self.company_data_path)
        # load production benchmarks
        benchmark_prod_json = os.path.join(data_dir, "benchmark_production_OECM.json")
        with open(benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.model_validate(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        # load intensity benchmarks
        benchmark_EI_json = os.path.join(data_dir, ei_filename)
        with open(benchmark_EI_json) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.model_validate(parsed_json)
        self.base_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

        self.data_warehouse = DataWarehouse(self.nzdpu_company_data, self.base_production_bm, self.base_EI_bm)
        self.company_ids = company_ids
        self.company_info_at_base_year = self.nzdpu_company_data.get_company_intensity_and_production_at_base_year(
            self.company_ids
        )


@pytest.fixture(scope="session")
def nzdpu_PC() -> NZDPU:
    return NZDPU("benchmark_EI_OECM_PC.json", ["US00130H1059", "US26441C2044", "KR7005490008"])


@pytest.fixture(scope="session")
def nzdpu_S3() -> NZDPU:
    return NZDPU("benchmark_EI_OECM_S3.json", ["US0921131092+Electricity Utilities", "US0921131092+Gas Utilities"])


tc = TestCase()


def test_target_projections(nzdpu_PC: NZDPU):
    breakpoint()

if __name__=="__main__":
   main()
