import warnings  # needed until apply behaves better with Pint quantities in arrays
import logging
import pandas as pd
import numpy as np

from abc import ABC
from typing import List, Type
from pydantic import ValidationError

import ITR
from ITR.data.osc_units import ureg, Q_, asPintSeries
from ITR.interfaces import EScope, IEmissionRealization, IEIRealization, ICompanyAggregates, ICompanyEIProjection
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, LoggingConfig

import pint

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
        self.company_data._calculate_target_projections(benchmark_projected_production)
        self.company_scope = {}
        
        if (getattr(benchmarks_projected_ei._EI_benchmarks, 'S1S2')
            and benchmarks_projected_ei._EI_benchmarks['S1S2'].production_centric):
            # Production-Centric benchmark: After projections have been made, shift S3 data into S1S2.
            # If we shift before we project, then S3 targets will not be projected correctly.
            logger.info(
                f"Shifting S3 emissions data into S1 according to Production-Centric benchmark rules"
            )
            for c in self.company_data._companies:
                if c.ghg_s3 and not ITR.isnan(c.ghg_s3.m):
                    # For Production-centric and energy-only data (except for Cement), convert all S3 numbers to S1 numbers
                    c.ghg_s1s2 = c.ghg_s1s2 + c.ghg_s3
                    c.ghg_s3 = None # Q_(0.0, c.ghg_s3.u)
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
                    c.projected_intensities.S1S2S3 = None
                if c.projected_targets.S3:
                    c.projected_targets.S1S2.projections = list( map(ICompanyEIProjection.add, c.projected_targets.S1S2.projections, c.projected_targets.S3.projections) )
                    c.projected_targets.S3 = None
                    c.projected_targets.S1S2S3 = None

        # Set scope information based on what company reports and what benchmark requres
        # benchmarks_projected_ei._EI_df makes life a bit easier...
        missing_company_scopes = []
        for c in self.company_data._companies:
            region = c.region
            try:
                bm_company_sector_region = benchmarks_projected_ei._EI_df.loc[c.sector, region]
            except KeyError:
                try:
                    region = 'Global'
                    bm_company_sector_region = benchmarks_projected_ei._EI_df.loc[c.sector, region]
                except KeyError:
                    missing_company_scopes.append(c.company_id)
                    continue
            scopes = benchmarks_projected_ei._EI_df.loc[c.sector, region].index.tolist()
            if len(scopes) == 1:
                self.company_scope[c.company_id] = scopes[0]
                continue
            for scope in [EScope.S1S2S3, EScope.S1S2, EScope.S1, EScope.S3]:
                if scope in scopes:
                    self.company_scope[c.company_id] = scope
                    break
            if c.company_id not in self.company_scope:
                missing_company_scopes.append(c.company_id)

        if missing_company_scopes:
            logger.warning(
                f"The following companies do not disclose scope data required by benchmark and will be removed: {missing_company_scopes}"
            )
            breakpoint()


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
        # FIXME: we just computed ALL company data above into a dataframe.  Why not use that?
        projected_trajectories = self.company_data.get_company_projected_trajectories(company_ids)
        df_trajectory = self._get_cumulative_emissions(
            projected_ei=projected_trajectories,
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_TRAJECTORY)

        projected_targets = self.company_data.get_company_projected_targets(company_ids)
        # Fill in ragged left edge of projected_targets with historic data, interpolating where we need to
        for col, year_data in projected_targets.items():
            mask = year_data.isna()
            if mask.any():
                projected_targets.loc[mask, col] = projected_trajectories.loc[mask, col]
            else:
                break

        # Fill in target projections so that we compute cumulative emissions consistently for targets and trajectories
        projected_targets = pd.concat([projected_trajectories.loc[projected_targets.index,
                                                                  projected_trajectories.columns.difference(projected_targets.columns)],
                                       projected_targets], axis=1)

        df_target = self._get_cumulative_emissions(
            projected_ei=projected_targets,
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_TARGET)
        df_budget = self._get_cumulative_emissions(
            projected_ei=self.benchmarks_projected_ei.get_SDA_intensity_benchmarks(company_info_at_base_year),
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_BUDGET)
        df_scope_data = pd.concat([df_trajectory, df_target, df_budget], axis=1)
        df = df_company_data.join(df_scope_data.reset_index('scope'))
        scope = df.scope
        invalid_scope_mask = scope.isna()
        if invalid_scope_mask.any():
            logger.error(
                f"dropping companies with invalid scope data: {scope[invalid_scope_mask].index.to_list()}"
            )
            df = df[~invalid_scope_mask]
        scope_index = df.columns.get_loc('region')+1
        df = df.drop('scope', axis=1)
        df.insert(scope_index, 'scope', scope)
        df[self.column_config.BENCHMARK_GLOBAL_BUDGET] = \
            pd.Series([self.benchmarks_projected_ei.benchmark_global_budget] * len(df),
                      dtype='pint[Gt CO2]',
                      index=df.index)
        # ICompanyAggregates wants this Quantity as a `str`
        df[self.column_config.BENCHMARK_TEMP] = [str(self.benchmarks_projected_ei.benchmark_temperature)] * len(df)
        companies = df.to_dict(orient="records")
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
        try:
            na_values = projected_emissions.iloc[:, 0].map(lambda x: ITR.isnan(x.m)).values
        except AttributeError:
            breakpoint()
        cumulative_emissions = projected_emissions[~na_values].sum(axis=1).astype('pint[Mt CO2]')
        return cumulative_emissions
