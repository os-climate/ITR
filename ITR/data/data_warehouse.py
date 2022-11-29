import warnings  # needed until apply behaves better with Pint quantities in arrays
import logging
import pandas as pd
import numpy as np
from abc import ABC
from typing import List, Type
from pydantic import ValidationError

from ITR.interfaces import IEmissionRealization, IEIRealization, ICompanyAggregates, ICompanyEIProjection, EScope
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, LoggingConfig

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class DataWarehouse(ABC):
    """
    General data provider super class.
    """

    def __init__(self, company_data: CompanyDataProvider,
                 benchmark_projected_production: ProductionBenchmarkDataProvider,
                 benchmarks_projected_ei: IntensityBenchmarkDataProvider,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        """
        Create a new data warehouse instance.

        :param company_data: CompanyDataProvider
        :param benchmark_projected_production: ProductionBenchmarkDataProvider
        :param benchmarks_projected_ei: IntensityBenchmarkDataProvider
        """
        self.benchmark_projected_production = benchmark_projected_production
        self.benchmarks_projected_ei = benchmarks_projected_ei
        self.temp_config = tempscore_config
        self.column_config = column_config
        self.company_data = company_data
        self.company_data._calculate_target_projections(benchmark_projected_production,
                                                        benchmarks_projected_ei.scope_to_calc)

        # If benchmark's scope to calculate is S3 - skip shifting data into S1S2
        if benchmarks_projected_ei.scope_to_calc == EScope.S3:
            return

        # After projections have been made, shift S3 data into S1S2.  If we shift before we project,
        # then S3 targets will not be projected correctly.
        for c in self.company_data._companies:
            if c.ghg_s3:
                # For Production-centric and energy-only data (except for Cement), convert all S3 numbers to S1 numbers
                c.ghg_s1s2 = c.ghg_s1s2 + c.ghg_s3
                c.ghg_s3 = 0
            if c.historic_data:
                if c.historic_data.emissions and c.historic_data.emissions.S3:
                    c.historic_data.emissions.S1S2 = list( map(IEmissionRealization.add, c.historic_data.emissions.S1S2, c.historic_data.emissions.S3) )
                    c.historic_data.emissions.S3 = []
                if c.historic_data.emissions_intensities and c.historic_data.emissions_intensities.S3:
                    c.historic_data.emissions_intensities.S1S2 = \
                        list( map(IEIRealization.add, c.historic_data.emissions_intensities.S1S2, c.historic_data.emissions_intensities.S3) )
                    c.historic_data.emissions_intensities.S3 = []
            if c.projected_intensities.S3:
                c.projected_intensities.S1S2.projections = list( map(ICompanyEIProjection.add, c.projected_intensities.S1S2.projections, c.projected_intensities.S3.projections) )
                c.projected_intensities.S3 = None
            if c.projected_targets.S3:
                c.projected_targets.S1S2.projections = list( map(ICompanyEIProjection.add, c.projected_targets.S1S2.projections, c.projected_targets.S3.projections) )
                c.projected_targets.S3 = None

    def get_preprocessed_company_data(self, company_ids: List[str]) -> List[ICompanyAggregates]:
        """
        Get all relevant data for a list of company ids. This method should return a list of ICompanyAggregates
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data and additional precalculated fields
        """
        company_data = self.company_data.get_company_data(company_ids)
        df_company_data = pd.DataFrame.from_records([c.dict() for c in company_data]).set_index(self.column_config.COMPANY_ID, drop=False)

        company_info_at_base_year = self.company_data.get_company_intensity_and_production_at_base_year(company_ids)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/128
            projected_production = self.benchmark_projected_production.get_company_projected_production(
                company_info_at_base_year, self.benchmarks_projected_ei.scope_to_calc).sort_index()

        # trajectories are projected from historic data and we are careful to fill all gaps between historic and projections
        projected_trajectories = self.company_data.get_company_projected_trajectories(company_ids)
        df_trajectory = self._get_cumulative_emissions(
            projected_ei=projected_trajectories,
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_TRAJECTORY)

        projected_targets = self.company_data.get_company_projected_targets(company_ids)
        df_target = self._get_cumulative_emissions(
            projected_ei=projected_targets,
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_TARGET)
        df_budget = self._get_cumulative_emissions(
            projected_ei=self.benchmarks_projected_ei.get_SDA_intensity_benchmarks(company_info_at_base_year),
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_BUDGET)
        df_company_data = pd.concat([df_company_data, df_trajectory, df_target, df_budget], axis=1)
        df_company_data[self.column_config.BENCHMARK_GLOBAL_BUDGET] = \
            pd.Series([self.benchmarks_projected_ei.benchmark_global_budget] * len(df_company_data),
                      dtype='pint[Gt CO2]',
                      index=df_company_data.index)
        df_company_data[self.column_config.BENCHMARK_TEMP] = \
            pd.Series([self.benchmarks_projected_ei.benchmark_temperature] * len(df_company_data),
                      dtype='pint[delta_degC]',
                      index=df_company_data.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            for col in [self.column_config.CUMULATIVE_TRAJECTORY, self.column_config.CUMULATIVE_TARGET, self.column_config.CUMULATIVE_BUDGET]:
                df_company_data[col] = df_company_data[col].apply(lambda x: str(x))
        companies = df_company_data.to_dict(orient="records")
        aggregate_company_data = [ICompanyAggregates.parse_obj(company) for company in companies]
        return aggregate_company_data

    def _convert_df_to_model(self, df_company_data: pd.DataFrame) -> List[ICompanyAggregates]:
        """
        transforms Dataframe Company data and preprocessed values into list of ICompanyAggregates instances

        :param df_company_data: pandas Dataframe with targets
        :return: A list containing the targets
        """
        df_company_data = df_company_data.where(pd.notnull(df_company_data), None).replace(
            {np.nan: None})  # set NaN to None since NaN is float instance
        companies_data_dict = df_company_data.to_dict(orient="records")
        model_companies: List[ICompanyAggregates] = []
        for company_data in companies_data_dict:
            try:
                model_companies.append(ICompanyAggregates.parse_obj(company_data))
            except ValidationError:
                logger.warning(
                    "(one of) the input(s) of company %s is invalid and will be skipped" % company_data[
                        self.column_config.COMPANY_NAME])
                pass
        return model_companies

    def _get_cumulative_emissions(self, projected_ei: pd.DataFrame, projected_production: pd.DataFrame) -> pd.Series:
        """
        get the weighted sum of the projected emission
        :param projected_ei: series of projected emissions intensities
        :param projected_production: PintArray of projected production amounts
        :return: cumulative emissions based on weighted sum of emissions intensity * production
        """
        projected_emissions = projected_ei.multiply(projected_production)
        projected_emissions = projected_emissions.applymap(lambda x: x if isinstance(x,float) else x if np.isfinite(x.m) else np.nan)
        null_idx = projected_emissions.index[projected_emissions.isnull().all(axis=1)]
        return pd.concat([projected_emissions.loc[null_idx, projected_emissions.columns[0]],
                          projected_emissions.loc[projected_emissions.index.difference(null_idx)].sum(axis=1)]).astype('pint[Mt CO2]')
