from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from ITR.interfaces import ICompanyData, ICompanyAggregates
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.configs import ColumnsConfig


class DataWarehouse(ABC):
    """
    General data provider super class.
    """

    def __init__(self, company_data: CompanyDataProvider,
                 benchmark_projected_production: ProductionBenchmarkDataProvider,
                 benchmarks_projected_emission_intensity: IntensityBenchmarkDataProvider):
        """
        Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        self.company_data = company_data
        self.benchmark_projected_production = benchmark_projected_production
        self.benchmarks_projected_emission_intensity = benchmarks_projected_emission_intensity

    def get_company_aggregates(self, company_ids: List[str]) -> List[ICompanyAggregates]:
        """
        Get all relevant data for a list of company ids. This method should return a list of ICompanyAggregates
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data and additional precalculated fields
        """
        df_company_data = pd.DataFrame.from_records([c.dict() for c in self.company_data.get_company_data(company_ids)])

        assert pd.Series(company_ids).isin(df_company_data.loc[:, ColumnsConfig.COMPANY_ID]).all(), \
            "some of the company ids are not included in the fundamental data"

        df_ghg_scope12 = df_company_data[[
            ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.GHG_SCOPE12]].set_index(
            ColumnsConfig.COMPANY_ID)
        projected_production = self.benchmark_projected_production.get_projected_production_per_company(df_ghg_scope12)

        df_company_data.loc[:, ColumnsConfig.CUMULATIVE_TRAJECTORY] = self._get_cumulative_emission(
            projected_emission_intensity=self.company_data.get_projected_intensities(company_ids),
            projected_production=projected_production).to_numpy()

        df_company_data.loc[:, ColumnsConfig.CUMULATIVE_TARGET] = self._get_cumulative_emission(
            projected_emission_intensity=self.company_data.get_projected_targets(company_ids),
            projected_production=projected_production).to_numpy()

        df_company_data.loc[:, ColumnsConfig.CUMULATIVE_BUDGET] = self._get_cumulative_emission(
            projected_emission_intensity=self.benchmarks_projected_emission_intensity.get_intensity_benchmarks(
                df_ghg_scope12),
            projected_production=projected_production).to_numpy()

        companies = df_company_data.to_dict(orient="records")

        aggregate_company_data: List[ICompanyAggregates] = [ICompanyAggregates.parse_obj(company) for company in
                                                            companies]

        return aggregate_company_data

    def _get_cumulative_emission(self, projected_emission_intensity: pd.DataFrame, projected_production: pd.DataFrame
                                 ) -> pd.Series:
        """
        get the weighted sum of the projected emission times the projected production
        :param projected_emission_intensity: series of projected emissions
        :param projected_production: series of projected production series
        :return: weighted sum of production and emission
        """
        return projected_emission_intensity.reset_index(drop=True).multiply(projected_production.reset_index(
            drop=True)).sum(axis=1)
