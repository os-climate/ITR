import warnings  # needed until apply behaves better with Pint quantities in arrays
import logging
import pandas as pd
import numpy as np

from abc import ABC
from typing import List, Type
from pydantic import ValidationError

import ITR
from ITR.data.osc_units import ureg, Q_, asPintSeries
from ITR.interfaces import EScope, IEmissionRealization, IEIRealization, ICompanyAggregates, ICompanyEIProjection, ICompanyEIProjections
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.logger import logger

import pint

logger = logging.getLogger(__name__)


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
        # benchmarks_projected_ei._EI_df is the EI dataframe for the benchmark
        # benchmark_projected_production.get_company_projected_production(company_sector_region_scope) gives production data per company (per year)
        # multiplying these two gives aligned emissions data for the company, in case we want to add missing data based on sector averages
        self.temp_config = tempscore_config
        self.column_config = column_config
        self.company_data = company_data
        self.company_data._calculate_target_projections(benchmark_projected_production)
        self.company_scope = {}

        # For production-centric benchmarks, S3 emissions are counted against S1 (and/or the S1 in S1+S2)
        def _align_and_sum_projected_targets(s1_projections, s3_projections):
            if s1_projections[0].year < s3_projections[0].year:
                while s1_projections[0].year < s3_projections[0].year:
                    s1_projections = s1_projections[1:]
            elif s1_projections[0].year > s3_projections[0].year:
                while s1_projections[0].year > s3_projections[0].year:
                    s3_projections = s3_projections[1:]
            return list( map(ICompanyEIProjection.add, s1_projections, s3_projections) )

        assert getattr(benchmarks_projected_ei._EI_benchmarks, 'S1S2') or (getattr(benchmarks_projected_ei._EI_benchmarks, 'S1') == None)
        if (getattr(benchmarks_projected_ei._EI_benchmarks, 'S1S2')
            and benchmarks_projected_ei._EI_benchmarks['S1S2'].production_centric):
            # Production-Centric benchmark: After projections have been made, shift S3 data into S1S2.
            # If we shift before we project, then S3 targets will not be projected correctly.
            logger.info(
                f"Shifting S3 emissions data into S1 according to Production-Centric benchmark rules"
            )
            for c in self.company_data._companies:
                if c.ghg_s3:
                    # For Production-centric and energy-only data (except for Cement), convert all S3 numbers to S1 numbers
                    if not ITR.isnan(c.ghg_s3.m):
                        c.ghg_s1s2 = c.ghg_s1s2 + c.ghg_s3
                    c.ghg_s3 = None # Q_(0.0, c.ghg_s3.u)
                if c.historic_data:
                    if c.historic_data.emissions and c.historic_data.emissions.S3:
                        if c.historic_data.emissions.S1:
                            c.historic_data.emissions.S1 = list( map(IEmissionRealization.add, c.historic_data.emissions.S1, c.historic_data.emissions.S3) )
                        c.historic_data.emissions.S1S2 = list( map(IEmissionRealization.add, c.historic_data.emissions.S1S2, c.historic_data.emissions.S3) )
                        c.historic_data.emissions.S3 = []
                        c.historic_data.emissions.S1S2S3 = []
                    if c.historic_data.emissions_intensities and c.historic_data.emissions_intensities.S3:
                        if c.historic_data.emissions_intensities.S1:
                            c.historic_data.emissions_intensities.S1 = \
                                list( map(IEIRealization.add, c.historic_data.emissions_intensities.S1, c.historic_data.emissions_intensities.S3) )
                        c.historic_data.emissions_intensities.S1S2 = \
                            list( map(IEIRealization.add, c.historic_data.emissions_intensities.S1S2, c.historic_data.emissions_intensities.S3) )
                        c.historic_data.emissions_intensities.S3 = []
                        c.historic_data.emissions_intensities.S1S2S3 = []
                if c.projected_intensities and c.projected_intensities.S3:
                    if c.projected_intensities.S1:
                        c.projected_intensities.S1.projections = list( map(ICompanyEIProjection.add, c.projected_intensities.S1.projections, c.projected_intensities.S3.projections) )
                    c.projected_intensities.S1S2.projections = list( map(ICompanyEIProjection.add, c.projected_intensities.S1S2.projections, c.projected_intensities.S3.projections) )
                    c.projected_intensities.S3 = None
                    c.projected_intensities.S1S2S3 = None
                if c.projected_targets and c.projected_targets.S3:
                    if c.projected_targets.S1:
                        c.projected_targets.S1.projections = _align_and_sum_projected_targets(c.projected_targets.S1.projections, c.projected_targets.S3.projections)
                    try:
                        # S3 projected targets may have been synthesized from a netzero S1S2S3 target and might need to be date-aligned with S1S2
                        c.projected_targets.S1S2.projections = _align_and_sum_projected_targets(c.projected_targets.S1S2.projections, c.projected_targets.S3.projections)
                    except AttributeError:
                        if c.projected_targets.S2:
                            logger.warning(f"Scope 1+2 target projections should have been created for {c.company_id}; repairing")
                            c.projected_targets.S1S2 = ICompanyEIProjections(ei_metric = c.projected_targets.S1.ei_metric,
                                                                             projections = list( map(ICompanyEIProjection.add, c.projected_targets.S1.projections, c.projected_targets.S2.projections) ))
                        else:
                            logger.warning(f"Scope 2 target projections missing from company with ID {c.company_id}; treating as zero")
                            c.projected_targets.S1S2 = ICompanyEIProjections(ei_metric = c.projected_targets.S1.ei_metric,
                                                                             projections = c.projected_targets.S1.projections)
                        if c.projected_targets.S3:
                            c.projected_targets.S1S2.projections = _align_and_sum_projected_targets(c.projected_targets.S1S2.projections, c.projected_targets.S3.projections)
                        else:
                            logger.warning(f"Scope 3 target projections missing from company with ID {c.company_id}; treating as zero")
                    except ValueError:
                        logger.error(f"S1+S2 targets not aligned with S3 targets for company with ID {c.company_id}; ignoring S3 data")
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


    def get_preprocessed_company_data(self, company_ids: List[str]) -> List[ICompanyAggregates]:
        """
        Get all relevant data for a list of company ids. This method should return a list of ICompanyAggregates
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data and additional precalculated fields
        """

        company_data = self.company_data.get_company_data(company_ids)
        df_company_data = pd.DataFrame.from_records([c.dict() for c in company_data]).set_index(self.column_config.COMPANY_ID, drop=False)
        valid_company_ids = df_company_data.index.to_list()

        company_info_at_base_year = self.company_data.get_company_intensity_and_production_at_base_year(valid_company_ids)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/128
            projected_production = self.benchmark_projected_production.get_company_projected_production(
                company_info_at_base_year) # .sort_index()

        # trajectories are projected from historic data and we are careful to fill all gaps between historic and projections
        # FIXME: we just computed ALL company data above into a dataframe.  Why not use that?
        projected_trajectories = self.company_data.get_company_projected_trajectories(valid_company_ids)
        df_trajectory = self._get_cumulative_emissions(
            projected_ei=projected_trajectories,
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_TRAJECTORY)

        projected_targets = self.company_data.get_company_projected_targets(valid_company_ids)
        # Fill in ragged left edge of projected_targets with historic data, interpolating where we need to
        for col, year_data in projected_targets.items():
            mask = year_data.apply(lambda x: ITR.isnan(x.m))
            if mask.all():
                # No sense trying to do anything with left-side all-NaN columns
                projected_targets = projected_targets.drop(columns=col)
                continue
            if mask.any():
                projected_targets.loc[mask[mask].index, col] = projected_trajectories.loc[mask[mask].index, col]
            else:
                break

        df_target = self._get_cumulative_emissions(
            projected_ei=projected_targets,
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_TARGET)
        df_budget = self._get_cumulative_emissions(
            projected_ei=self.benchmarks_projected_ei.get_SDA_intensity_benchmarks(company_info_at_base_year),
            projected_production=projected_production).rename(self.column_config.CUMULATIVE_BUDGET)
        df_scope_data = pd.concat([df_trajectory, df_target, df_budget], axis=1)
        na_scope_mask = df_scope_data.isna().apply(lambda x: x.any(), axis=1)
        if na_scope_mask.any():
            logger.info(
                f"Dropping invalid scope data: {na_scope_mask[na_scope_mask].index}"
            )
            df_scope_data = df_scope_data[~na_scope_mask]
        df_company_scope = df_company_data.join(df_scope_data).reset_index('scope')
        na_company_mask = df_company_scope.scope.isna()
        if na_company_mask.any():
            logger.warning(
                f"Dropping companies with no scope data: {df_company_scope[na_company_mask].index.get_level_values(level='company_id').to_list()}"
            )
        df_company_data = df_company_scope[~na_company_mask].copy()
        df_company_data[self.column_config.BENCHMARK_GLOBAL_BUDGET] = \
            pd.Series([self.benchmarks_projected_ei.benchmark_global_budget] * len(df_company_data),
                      dtype='pint[Gt CO2]',
                      index=df_company_data.index)
        # ICompanyAggregates wants this Quantity as a `str`
        df_company_data[self.column_config.BENCHMARK_TEMP] = [str(self.benchmarks_projected_ei.benchmark_temperature)] * len(df_company_data)
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
        # By picking only the rows of projected_production (columns of projected_production.T)
        # that match projected_ei (columns of projected_ei.T), the rows of the DataFrame are not re-sorted
        projected_emissions_t = projected_ei.T.mul(projected_production.T[projected_ei.T.columns])
        cumulative_emissions = projected_emissions_t.T.sum(axis=1).astype('pint[Mt CO2]')
        return cumulative_emissions
