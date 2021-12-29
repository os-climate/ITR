from abc import ABC
from typing import List
import pandas as pd
from pydantic import ValidationError
import numpy as np

import pint
import pint_pandas

ureg = pint.get_application_registry()
Q_ = ureg.Quantity

from ITR.interfaces import ICompanyAggregates
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from typing import Type
import logging


class DataWarehouse(ABC):
    """
    General data provider super class.
    """

    def __init__(self, company_data: CompanyDataProvider,
                 benchmark_projected_production: ProductionBenchmarkDataProvider,
                 benchmarks_projected_emission_intensity: IntensityBenchmarkDataProvider,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        """
        Create a new data warehouse instance.

        :param company_data: CompanyDataProvider
        :param benchmark_projected_production: ProductionBenchmarkDataProvider
        :param benchmarks_projected_emission_intensity: IntensityBenchmarkDataProvider
        """
        self.company_data = company_data
        self.benchmark_projected_production = benchmark_projected_production
        self.benchmarks_projected_emission_intensity = benchmarks_projected_emission_intensity
        self.temp_config = tempscore_config
        self.column_config = column_config

    def get_preprocessed_company_data(self, company_ids: List[str]) -> List[ICompanyAggregates]:
        """
        Get all relevant data for a list of company ids. This method should return a list of ICompanyAggregates
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data and additional precalculated fields
        """
        company_data = self.company_data.get_company_data(company_ids)
        df_company_data = pd.DataFrame.from_records([c.dict() for c in company_data])\
            .set_index(self.column_config.COMPANY_ID)

        missing_ids = [c_id for c_id in company_ids if c_id not in df_company_data.index]
        assert not missing_ids, f"Company IDs are not included in the fundamental data: {missing_ids}"

        company_info_at_base_year = self.company_data.get_company_intensity_and_production_at_base_year(company_ids)
        projected_production = self.benchmark_projected_production.get_company_projected_production(
            company_info_at_base_year).sort_index()

        df_company_data.loc[:, self.column_config.CUMULATIVE_TRAJECTORY] = self._get_cumulative_emission(
            projected_emission_intensity=self.company_data.get_company_projected_intensities(company_ids),
            projected_production=projected_production)

        df_new = self._get_cumulative_emission(
            projected_emission_intensity=self.company_data.get_company_projected_targets(company_ids),
            projected_production=projected_production)

        df_new = self._get_cumulative_emission(
            projected_emission_intensity=self.benchmarks_projected_emission_intensity.get_SDA_intensity_benchmarks(
                company_info_at_base_year),
            projected_production=projected_production)

        df_company_data[self.column_config.BENCHMARK_GLOBAL_BUDGET] = pint_pandas.PintArray([self.benchmarks_projected_emission_intensity.benchmark_global_budget.m]*
                                                                                            len(df_company_data), dtype='pint[t CO2]')
        df_company_data[self.column_config.BENCHMARK_TEMP] = pint_pandas.PintArray([self.benchmarks_projected_emission_intensity.benchmark_temperature.m]*
                                                                                   len(df_company_data), dtype='pint[delta_degC]')

        companies = df_company_data.reset_index().to_dict(orient="records")

        aggregate_company_data: List[ICompanyAggregates] = [ICompanyAggregates.parse_obj(company) for company in
                                                            companies]

        return aggregate_company_data

    def _convert_df_to_model(self, df_company_data: pd.DataFrame) -> List[ICompanyAggregates]:

        """
        transforms Dataframe Company data and preprocessed values into list of IDataProviderTarget instances

        :param df_company_data: pandas Dataframe with targets
        :return: A list containing the targets
        """
        logger = logging.getLogger(__name__)
        df_company_data = df_company_data.where(pd.notnull(df_company_data), None).replace(
            {np.nan: None})  # set NaN to None since NaN is float instance
        companies_data_dict = df_company_data.to_dict(orient="records")
        model_companies: List[ICompanyAggregates] = []
        for company_data in companies_data_dict:
            try:
                model_companies.append(ICompanyAggregates.parse_obj(company_data))
            except ValidationError as e:
                logger.warning(
                    "(one of) the input(s) of company %s is invalid and will be skipped" % company_data[
                        self.column_config.COMPANY_NAME])
                pass
        return model_companies

    def _weighted_mean(df, values, weights, groupby):
        df = df.copy()
        grouped = df.groupby(groupby)
        df['weighted_average'] = df[values] / grouped[weights].transform('sum') * df[weights]
        return grouped['weighted_average'].sum(min_count=1) #min_count is required for Grouper objects

    def _get_cumulative_emission(self, projected_emission_intensity: pd.DataFrame, production_weights: pd.DataFrame
                                 ) -> pd.DataFrame:
        """
        get the weighted sum of the projected emission times the projected production
        :param projected_emission_intensity: series of projected emissions
        :param production_weights: series of weights to use for emission_intensity sums
        :return: weighted sum of production and emission
        """
        return projected_emission_intensity.multiply(projected_production).sum(axis=1)
