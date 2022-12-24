import warnings  # needed until quantile behaves better with Pint quantities in arrays
import numpy as np
import pandas as pd
import pint
from pint import DimensionalityError
import pydantic
from pydantic import ValidationError

from functools import reduce, partial
from operator import add
from typing import List, Type, Dict

import ITR
from ITR.data.osc_units import Q_, PA_, asPintSeries, PintType

from ITR.configs import ColumnsConfig, TemperatureScoreConfig, VariablesConfig, ProjectionControls
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, \
    IntensityBenchmarkDataProvider
from ITR.interfaces import ICompanyData, EScope, IProductionBenchmarkScopes, IEIBenchmarkScopes, \
    IBenchmark, IProjection, ICompanyEIProjections, ICompanyEIProjectionsScopes, IHistoricEIScopes, \
    IHistoricEmissionsScopes, IProductionRealization, ITargetData, IHistoricData, ICompanyEIProjection, \
    IEmissionRealization
from ITR.interfaces import EI_Quantity
from ITR.logger import logger

# TODO handling of scopes in benchmarks


# The benchmark projected production format is based on year-over-year growth and starts out like this:

#                                                2019     2020            2049       2050
# region                 sector        scope                    ...                                                  
# Steel                  Global        AnyScope   0.0   0.00306  ...     0.0155     0.0155
#                        Europe        AnyScope   0.0   0.00841  ...     0.0155     0.0155
#                        North America AnyScope   0.0   0.00748  ...     0.0155     0.0155
# Electricity Utilities  Global        AnyScope   0.0    0.0203  ...     0.0139     0.0139
#                        Europe        AnyScope   0.0    0.0306  ...   -0.00113   -0.00113
#                        North America AnyScope   0.0    0.0269  ...   0.000426   0.000426
# etc.

# To compute the projected production for a company in given sector/region, we need to start with the
# base_year_production for that company and apply the year-over-year changes projected by the benchmark
# until all years are computed.  We need to know production of each year, not only the final year
# because the cumumulative emissions of the company will be the sum of the emissions of each year,
# which depends on both the production projection (computed here) and the emissions intensity projections
# (computed elsewhere).

# Let Y2019 be the production of a company in 2019.
# Y2020 = Y2019 + (Y2019 * df_pp[2020]) = Y2019 + Y2019 * (1.0 + df_pp[2020])
# Y2021 = Y2020 + (Y2020 * df_pp[2020]) = Y2020 + Y2020 * (1.0 + df_pp[2021])
# etc.

# The Pandas `cumprod` function calculates precisely the cumulative product we need
# As the math shows above, the terms we need to accumulate are 1.0 + growth.

# df.add(1).cumprod(axis=1).astype('pint[]') results in a project that looks like this:
# 
#                                                2019     2020  ...      2049      2050
# region                 sector        scope                    ...                    
# Steel                  Global        AnyScope   1.0  1.00306  ...  1.419076  1.441071
#                        Europe        AnyScope   1.0  1.00841  ...  1.465099  1.487808
#                        North America AnyScope   1.0  1.00748  ...  1.457011  1.479594
# Electricity Utilities  Global        AnyScope   1.0  1.02030  ...  2.907425  2.947838
#                        Europe        AnyScope   1.0  1.03060  ...  1.751802  1.749822
#                        North America AnyScope   1.0  1.02690  ...  2.155041  2.155959
# etc.

class BaseProviderProductionBenchmark(ProductionBenchmarkDataProvider):

    def __init__(self, production_benchmarks: IProductionBenchmarkScopes,
                 column_config: Type[ColumnsConfig] = ColumnsConfig):
        """
        Base provider that relies on pydantic interfaces. Default for FastAPI usage
        :param production_benchmarks: List of IProductionBenchmarkScopes
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        super().__init__()
        self.column_config = column_config
        self._productions_benchmarks = production_benchmarks
        try:
            self._prod_delta_df = pd.DataFrame([self._convert_benchmark_to_series(bm, EScope.AnyScope) for bm in getattr(self._productions_benchmarks, EScope.AnyScope.name).benchmarks])
        except AttributeError:
            breakpoint()
        self._prod_delta_df.index.names = [self.column_config.SECTOR, self.column_config.REGION, self.column_config.SCOPE]
        # See comment above to understand use of `cumprod` function
        self._prod_df = self._prod_delta_df.add(1.0).cumprod(axis=1).astype('pint[]')

    # Note that benchmark production series are dimensionless.
    # FIXME: They also don't need a scope.  Remove scope when we change IBenchmark format...
    def _convert_benchmark_to_series(self, benchmark: IBenchmark, scope: EScope) -> pd.Series:
        """
        extracts the company projected intensity or production targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        units = str(benchmark.benchmark_metric)
        years, values = list(map(list, zip(*{r.year: r.value.to(units).m for r in benchmark.projections}.items())))
        return pd.Series(PA_(values, dtype=units),
                         index = years, name=(benchmark.sector, benchmark.region, scope))

    # Production benchmarks are dimensionless, relevant for AnyScope
    def _get_projected_production(self, scope: EScope = EScope.AnyScope) -> pd.DataFrame:
        """
        Converts IProductionBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: pd.DataFrame
        """
        return self._prod_df
    
        # The call to this function generates a 42-row (and counting...) DataFrame for the one row we're going to end up needing...
        df_bm = pd.DataFrame([self._convert_benchmark_to_series(bm, scope) for bm in getattr(self._productions_benchmarks, scope.name).benchmarks])
        df_bm.index.names = [self.column_config.SECTOR, self.column_config.REGION, self.column_config.SCOPE]
        
        df_partial_pp = df_bm.add(1).cumprod(axis=1).astype('pint[]')

        return df_partial_pp

    def get_company_projected_production(self, company_sector_region_scope: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for list of companies
        :param company_sector_region_scope: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE
        :return: DataFrame of projected productions for [base_year through 2050]
        """
        # get_benchmark_projections is an expensive call.  It's designed to return ALL benchmark info for ANY sector/region combo passed
        # and it does all that work whether we need all the data or just one row.  Best to lift this out of any inner loop
        # and use the valuable DataFrame it creates.
        company_benchmark_projections = self.get_benchmark_projections(company_sector_region_scope)
        company_production = company_sector_region_scope.set_index(self.column_config.SCOPE, append=True)[self.column_config.BASE_YEAR_PRODUCTION]
        # If we don't have valid production data for base year, we get back a nan result that's a pain to debug
        assert not ITR.isnan(company_production.values[0].m)
        # We transpose the operation so that Pandas is happy to preserve the dtype integrity of the column
        company_projected_productions_t = company_benchmark_projections.T.mul(company_production, axis=1)
        return company_projected_productions_t.T

    def get_benchmark_projections(self, company_sector_region_scope: pd.DataFrame, scope: EScope = EScope.AnyScope) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with production benchmarks per company_id given a region and sector.
        :param company_sector_region_scope: DataFrame indexed by ColumnsConfig.COMPANY_ID
        with at least the following columns: ColumnsConfig.SECTOR, ColumnsConfig.REGION, and ColumnsConfig.SCOPE
        :param scope: a scope
        :return: An all-quantified DataFrame with intensity benchmark data per calendar year per row, indexed by company.
        """

        benchmark_projection = self._get_projected_production(scope)  # TODO optimize performance
        df = (company_sector_region_scope[['sector', 'region', 'scope']]
              .reset_index()
              .drop_duplicates()
              .set_index(['company_id', 'scope']))
        # We drop the meaningless S1S2 from the production benchmark and replace it with the company's scope.
        # This is needed to make indexes align when we go to multiply production times intensity for a scope.
        company_benchmark_projections = df.merge(benchmark_projection.droplevel('scope'),
                                                 left_on=['sector', 'region'], right_index=True, how='left')
        mask = company_benchmark_projections.iloc[:, -1].isna()
        if mask.any():
            # Patch up unknown regions as "Global"
            global_benchmark_projections = df[mask].merge(benchmark_projection.loc[(slice(None), 'Global'), :].droplevel(['region','scope']),
                                                          left_on=['sector'], right_index=True, how='left').drop(columns='region')
            combined_benchmark_projections = pd.concat([company_benchmark_projections[~mask].drop(columns='region'),
                                                        global_benchmark_projections])
            return combined_benchmark_projections.drop(columns='sector')
        return company_benchmark_projections.drop(columns=['sector', 'region'])


class BaseProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(self, EI_benchmarks: IEIBenchmarkScopes,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 projection_controls: ProjectionControls = ProjectionControls()):
        super().__init__(EI_benchmarks.benchmark_temperature, EI_benchmarks.benchmark_global_budget,
                         EI_benchmarks.is_AFOLU_included)
        self._EI_benchmarks = EI_benchmarks
        self.column_config = column_config
        self.projection_controls = projection_controls
        benchmarks_as_series = []
        for scope in EScope.get_result_scopes():
            try:
                for bm in getattr(EI_benchmarks, scope.name).benchmarks:
                    benchmarks_as_series.append(self._convert_benchmark_to_series(bm, scope))
            except AttributeError:
                pass
        with warnings.catch_warnings():
            # pd.DataFrame.__init__ (in pandas/core/frame.py) ignores the beautiful dtype information adorning the pd.Series list elements we are providing.  Sad!
            warnings.simplefilter("ignore")
            self._EI_df = pd.DataFrame(benchmarks_as_series).sort_index()
        self._EI_df.index.names = [self.column_config.SECTOR, self.column_config.REGION, self.column_config.SCOPE]
        

    # SDA stands for Sectoral Decarbonization Approach; see https://sciencebasedtargets.org/resources/files/SBTi-Power-Sector-15C-guide-FINAL.pdf
    def get_SDA_intensity_benchmarks(self, company_info_at_base_year: pd.DataFrame, scope_to_calc: EScope = None) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_info_at_base_year: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        intensity_benchmarks = self._get_intensity_benchmarks(company_info_at_base_year,
                                                              scope_to_calc)
        decarbonization_paths = self._get_decarbonizations_paths(intensity_benchmarks)
        last_ei = intensity_benchmarks[self.projection_controls.TARGET_YEAR]
        ei_base = intensity_benchmarks[self.projection_controls.BASE_YEAR]
        df = decarbonization_paths.mul((ei_base - last_ei), axis=0)
        df = df.add(last_ei, axis=0)
        idx = pd.Index.intersection(df.index,
                                    pd.MultiIndex.from_arrays([company_info_at_base_year.index,
                                                               company_info_at_base_year.scope]))
        df = df.loc[idx]
        return df

    def _get_decarbonizations_paths(self, intensity_benchmarks: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        Returns a DataFrame with the projected decarbonization paths for the supplied companies in intensity_benchmarks.
        :param: A DataFrame with company and intensity benchmarks per calendar year per row
        :return: A pd.DataFrame with company and decarbonisation path s per calendar year per row
        """
        return intensity_benchmarks.apply(lambda row: self._get_decarbonization(row), axis=1)

    def _get_decarbonization(self, intensity_benchmark_row: pd.Series) -> pd.Series:
        """
        Overrides subclass method
        returns a Series with the decarbonization path for a benchmark.
        :param: A Series with a company's intensity benchmarks per calendar year per row
        :return: A pd.Series with a company's decarbonisation paths per calendar year per row
        """
        first_ei = intensity_benchmark_row[self.projection_controls.BASE_YEAR]
        last_ei = intensity_benchmark_row[self.projection_controls.TARGET_YEAR]
        # TODO: does this still throw a warning when processing a NaN?  convert to base units before accessing .magnitude
        return intensity_benchmark_row.apply(lambda x: (x - last_ei) / (first_ei - last_ei))

    def _convert_benchmark_to_series(self, benchmark: IBenchmark, scope: EScope) -> pd.Series:
        """
        extracts the company projected intensities or targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        s = pd.Series({p.year: p.value for p in benchmark.projections}, name=(benchmark.sector, benchmark.region, scope),
                      dtype=f'pint[{str(benchmark.benchmark_metric)}]')
        return s

    def _get_intensity_benchmarks(self, company_sector_region_scope: pd.DataFrame, scope_to_calc: EScope = None) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_scope: DataFrame indexed by ColumnsConfig.COMPANY_ID
        with at least the following columns: ColumnsConfig.SECTOR, ColumnsConfig.REGION, and ColumnsConfig.SCOPE
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        benchmark_projections = self._EI_df
        df = company_sector_region_scope[['sector', 'region', 'scope']]
        if scope_to_calc is not None:
            df = df[df.scope.eq(scope_to_calc)]

        df = df.join(benchmark_projections, on=['sector','region','scope'], how='left')
        mask = df.iloc[:, -1].isna()
        if mask.any():
            # We have request for benchmark data for either regions or scopes we don't have...
            # Resetting the index gives us row numbers useful for editing DataFrame with fallback data
            df = df.reset_index()
            mask = df.iloc[:, -1].isna()
            benchmark_global = benchmark_projections.loc[:, 'Global', :]
            # DF1 selects all global data matching sector and scope...
            df1 = df[mask].iloc[:, 0:4].join(benchmark_global, on=['sector','scope'], how='inner')
            # ...which we can then mark as 'Global'
            df1.region = 'Global'
            df.loc[df1.index, :] = df1
            # Remove any NaN rows from DF we could not update
            mask1 = df.iloc[:, -1].isna()
            df2 = df[~mask1]
            # Restore the COMPANY_ID index; we no longer need row numbers to keep edits straight
            company_benchmark_projections = df2.set_index('company_id')
        else:
            company_benchmark_projections = df
        company_benchmark_projections.set_index('scope', append=True, inplace=True)
        # Drop SECTOR and REGION as the result will be used by math functions operating across the whole DataFrame
        return company_benchmark_projections.drop(['sector', 'region'], axis=1)


class BaseCompanyDataProvider(CompanyDataProvider):
    """
    Data provider skeleton for JSON files parsed by the fastAPI json encoder. This class serves primarily for connecting
    to the ITR tool via API.

    :param companies: A list of ICompanyData objects that each contain fundamental company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param projection_controls: An optional ProjectionControls object containing projection settings
    """

    def __init__(self,
                 companies: List[ICompanyData],
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 projection_controls: ProjectionControls = ProjectionControls()):
        super().__init__()
        self.column_config = column_config
        self.projection_controls = projection_controls
        self.missing_ids = set([])
        self._companies = self._validate_projected_trajectories(companies)

    def _validate_projected_trajectories(self, companies: List[ICompanyData]) -> List[ICompanyData]:
        companies_without_data = [c.company_id for c in companies if
                                  not c.historic_data and not c.projected_intensities]
        if companies_without_data:
            error_message = f"Provide either historic emission data or projections for companies with " \
                            f"IDs {companies_without_data}"
            logger.error(error_message)
            raise ValueError(error_message)
        companies_without_projections = [c for c in companies if not c.projected_intensities]
        if companies_without_projections:
            companies_with_projections = [c for c in companies if c.projected_intensities]
            return companies_with_projections + EITrajectoryProjector(self.projection_controls).project_ei_trajectories(
                companies_without_projections)
        else:
            return companies

    # Because this presently defaults to S1S2 always, targets spec'd for S1 only, S2 only, or S1+S2+S3 are not well-handled.
    def _convert_projections_to_series(self, company: ICompanyData, feature: str,
                                       scope: EScope = EScope.S1S2) -> pd.Series:
        """
        extracts the company projected intensities or targets for a given scope
        :param feature: PROJECTED_TRAJECTORIES or PROJECTED_TARGETS (both are intensities)
        :param scope: a scope
        :return: pd.Series
        """
        company_dict = company.dict()
        production_units = str(company_dict[self.column_config.PRODUCTION_METRIC])
        emissions_units = str(company_dict[self.column_config.EMISSIONS_METRIC])

        if company_dict[feature][scope.name]:
            # Simple case: just one scope
            projections = company_dict[feature][scope.name]['projections']
            return pd.Series(
                {p['year']: p['value'] for p in projections},
                name=(company.company_id, scope), dtype=f'pint[{emissions_units}/({production_units})]')
        else:
            assert False
            # Complex case: S1+S2 or S1+S2+S3...we really don't handle yet
            scopes = [EScope[s] for s in scope.value.split('+')]
            projection_scopes = {s: company_dict[feature][s]['projections'] for s in scopes if company_dict[feature][s.name]}
            if len(projection_scope_names) > 1:
                projection_series = {}
                for s in scopes:
                    projection_series[s] = pd.Series(
                        {p['year']: p['value'] for p in company_dict[feature][s.name]['projections']},
                        name=(company.company_id, s), dtype=f'pint[{emissions_units}/({production_units})]')
                series_adder = partial(pd.Series.add, fill_value=0)
                res = reduce(series_adder, projection_series.values())
                return res
            elif len(projection_scopes) == 0:
                return pd.Series(
                    {year: np.nan for year in range(self.historic_years[-1] + 1, self.projection_controls.TARGET_YEAR + 1)},
                    name=company.company_id, dtype=f'pint[{emissions_units}/({production_units})]'
                )
            else:
                projections = company_dict[feature][list(projection_scopes.keys())[0]]['projections']

    def _calculate_target_projections(self, production_bm: BaseProviderProductionBenchmark):
        """
        We cannot calculate target projections until after we have loaded benchmark data.
        We do so when companies are associated with benchmarks, in the DataWarehouse construction
        
        :param production_bm: A Production Benchmark (multi-sector, single-scope, 2020-2050)
        """        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # FIXME: Note that we don't need to call with a scope, because production is independent of scope.
            # We use the arbitrary EScope.AnyScope just to be explicit about that.
            df_partial_pp = production_bm._get_projected_production(EScope.AnyScope)

        for c in self._companies:
            if c.projected_targets is not None:
                continue
            if c.target_data is None:
                logger.warning(f"No target data for {c.company_name}")
                c.projected_targets = ICompanyEIProjectionsScopes()
            else:
                base_year_production = next((p.value for p in c.historic_data.productions if
                                             p.year == self.projection_controls.BASE_YEAR), None)
                try:
                    co_cumprod = df_partial_pp.loc[c.sector, c.region, EScope.AnyScope] * base_year_production
                except KeyError:
                    # FIXME: Should we fix region info upstream when setting up comopany data?
                    co_cumprod = df_partial_pp.loc[c.sector, "Global", EScope.AnyScope] * base_year_production
                try:
                    c.projected_targets = EITargetProjector(self.projection_controls).project_ei_targets(c, co_cumprod)
                except Exception as err:
                    logger.error(f"Exception {err} raised while calculating target projections for {c.company_id}")

    
    # ??? Why prefer TRAJECTORY over TARGET?
    def _get_company_intensity_at_year(self, year: int, company_ids: List[str]) -> pd.Series:
        """
        Returns projected intensities for a given set of companies and year
        :param year: calendar year
        :param company_ids: List of company ids
        :return: pd.Series with intensities for given company ids
        """
        return self.get_company_projected_trajectories(company_ids, year=year)

    def get_company_data(self, company_ids: List[str]) -> List[ICompanyData]:
        """
        Get all relevant data for a list of company ids. This method should return a list of ICompanyData
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        company_data = [company for company in self._companies if company.company_id in company_ids]

        if len(company_data) is not len(company_ids):
            self.missing_ids.update(set([c_id for c_id in company_ids if c_id not in [c.company_id for c in company_data]]))
            logger.warning(f"Companies not found in fundamental data and excluded from further computations: "
                           f"{self.missing_ids}")

        return company_data

    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        Gets the value of a variable for a list of companies ids
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        # FIXME: this is an expensive operation as it converts all fields in the model just to get a single VARIABLE_NAME
        return self.get_company_fundamentals(company_ids)[variable_name]

    def get_company_intensity_and_production_at_base_year(self, company_ids: List[str]) -> pd.DataFrame:
        """
        overrides subclass method
        :param: company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.PRODUCTION_METRIC, ColumnsConfig.BASE_EI,
        ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE,
        ColumnsConfig.GHG_SCOPE12, ColumnsConfig.GHG_SCOPE3
        
        The BASE_EI column is for the scope in the SCOPE column.
        """
        # FIXME: this is an expensive operation as it converts many fields in the model just to get a small subset of the DataFrame
        # FIXME: this creates an untidy data mess.  GHG_SCOPE12 and GHG_SCOPE3 are anachronisms.
        df_fundamentals = self.get_company_fundamentals(company_ids)
        base_year = self.projection_controls.BASE_YEAR
        company_info = df_fundamentals.loc[
            company_ids, [self.column_config.SECTOR, self.column_config.REGION,
                          self.column_config.BASE_YEAR_PRODUCTION,
                          self.column_config.GHG_SCOPE12,
                          self.column_config.GHG_SCOPE3]]
        ei_at_base = self._get_company_intensity_at_year(base_year, company_ids).rename(self.column_config.BASE_EI)
        df = company_info.merge(ei_at_base, left_index=True, right_index=True)
        df.reset_index('scope', inplace=True)
        cols = df.columns.tolist()        
        df = df[cols[1:3] + [cols[0]] + cols[3:]]
        return df

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company (company_id is a column)
        """
        excluded_cols = ['projected_targets', 'projected_intensities', 'historic_data']
        df = pd.DataFrame.from_records(
            [ICompanyData.parse_obj({k:v for k, v in c.dict().items() if k not in excluded_cols}).dict()
             for c in self.get_company_data(company_ids)]).set_index(self.column_config.COMPANY_ID)
        return df

    def get_company_projected_trajectories(self, company_ids: List[str], year=None) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :param year: values for a specific year, or all years if None
        :return: A pandas DataFrame with projected intensity trajectories per company, indexed by company_id and scope
        """
        projected_trajectories = [(c.company_id, scope, getattr(c.projected_intensities, scope.name).projections)
                                  # FIXME: we should make _companies a dict so we can look things up rather than searching every time!
                                  for c in self._companies for scope in EScope.get_result_scopes()
                                  if c.company_id in company_ids
                                  if getattr(c.projected_intensities, scope.name)]
        if projected_trajectories:
            if year is not None:
                company_ids, scopes, values = list(map(list, zip(*[(pt[0], pt[1], [yvp.value for yvp in pt[2] if yvp.year==year][0]) for pt in projected_trajectories])))
                index=pd.MultiIndex.from_tuples(zip(company_ids, scopes), names=["company_id", "scope"])
                return pd.Series(data=values, index=index)
            else:
                company_ids, scopes, yvp_dicts = list(map(list, zip(*[(pt[0], pt[1], {yvp.year:yvp.value for yvp in pt[2]}) for pt in projected_trajectories])))
                index=pd.MultiIndex.from_tuples(zip(company_ids, scopes), names=["company_id", "scope"])
                return pd.DataFrame(data=yvp_dicts, index=index)
        return pd.DataFrame()

    def get_company_projected_targets(self, company_ids: List[str], year=None) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :param year: values for a specific year, or all years if None
        :return: A pandas DataFrame with projected intensity targets per company, indexed by company_id
        """
        # Tempting as it is to follow the pattern of constructing the same way we create `projected_trajectories`
        # targets are trickier because they have ragged left edges that want to fill with NaNs when put into DataFrames.
        # _convert_projections_to_series has the nice side effect that PintArrays produce NaNs with units.
        # So if we just need a year from this dataframe, we compute the whole dataframe and return one column.
        # Feel free to write a better implementation if you have time!
        target_list = [self._convert_projections_to_series(c, self.column_config.PROJECTED_TARGETS, scope)
                       for c in self.get_company_data(company_ids)
                       for scope in EScope.get_result_scopes()
                       if c.projected_targets and getattr(c.projected_targets, scope.name)]
        if target_list:
            with warnings.catch_warnings():
                # pd.DataFrame.__init__ (in pandas/core/frame.py) ignores the beautiful dtype information adorning the pd.Series list elements we are providing.  Sad!
                warnings.simplefilter("ignore")
                # If target_list produces a ragged left edge, resort columns so that earliest year is leftmost
                df = pd.DataFrame(target_list).sort_index(axis=1)
                df.index.set_names(['company_id', 'scope'], inplace=True)
                if year is not None:
                    return df[year]
                return df
        return pd.DataFrame()


class EIProjector(object):
    """
    This class implements generic projection functions used for both trajectory and target projection.
    """

    def __init__(self, projection_controls: ProjectionControls = ProjectionControls()):
        self.projection_controls = projection_controls

    def _get_bounded_projections(self, results):
        if isinstance(results, list):
            projections = [projection for projection in results
                           if projection.year in range(self.projection_controls.BASE_YEAR, self.projection_controls.TARGET_YEAR+1)]
        else:
            projections = [ICompanyEIProjection(year=year, value=value) for year, value in results.items()
                           if year in range(self.projection_controls.BASE_YEAR, self.projection_controls.TARGET_YEAR+1)]
        return projections


class EITrajectoryProjector(EIProjector):
    """
    This class projects emissions intensities on company level based on historic data on:
    - A company's emission history (in t CO2)
    - A company's production history (units depend on industry, e.g. TWh for electricity)

    It returns the full set of both historic emissions intensities and projected emissions intensities.
    """

    def __init__(self, projection_controls: ProjectionControls = ProjectionControls()):
        super().__init__(projection_controls=projection_controls)

    def project_ei_trajectories(self, companies: List[ICompanyData]) -> List[ICompanyData]:
        historic_data = self._extract_historic_data(companies)
        self._compute_missing_historic_ei(companies, historic_data)

        historic_years = [column for column in historic_data.columns if type(column) == int]
        projection_years = range(max(historic_years), self.projection_controls.TARGET_YEAR)
        historic_intensities = historic_data[historic_years].query(
            f"variable=='{VariablesConfig.EMISSIONS_INTENSITIES}'")
        standardized_intensities = self._standardize(historic_intensities)
        intensity_trends = self._get_trends(standardized_intensities)
        extrapolated = self._extrapolate(intensity_trends, projection_years, historic_data)
        self._add_projections_to_companies(companies, extrapolated)
        return companies

    def _extract_historic_data(self, companies: List[ICompanyData]) -> pd.DataFrame:
        data = []
        for company in companies:
            if not company.historic_data:
                continue
            if company.historic_data.productions:
                data.append(self._historic_productions_to_dict(company.company_id, company.historic_data.productions))
            if company.historic_data.emissions:
                data.extend(self._historic_emissions_to_dicts(company.company_id, company.historic_data.emissions))
            if company.historic_data.emissions_intensities:
                data.extend(self._historic_ei_to_dicts(company.company_id,
                                                       company.historic_data.emissions_intensities))
        if not data:
            logger.error(f"No historic data for companies: {[c.company_id for c in companies]}")
            raise ValueError("No historic data anywhere")
        return pd.DataFrame.from_records(data).set_index(
            [ColumnsConfig.COMPANY_ID, ColumnsConfig.VARIABLE, ColumnsConfig.SCOPE])

    def _historic_productions_to_dict(self, id: str, productions: List[IProductionRealization]) -> Dict[str, str]:
        prods = {prod.year: prod.value for prod in productions}
        return {ColumnsConfig.COMPANY_ID: id, ColumnsConfig.VARIABLE: VariablesConfig.PRODUCTIONS,
                ColumnsConfig.SCOPE: 'Production', **prods}

    def _historic_emissions_to_dicts(self, id: str, emissions_scopes: IHistoricEmissionsScopes) -> List[Dict[str, str]]:
        data = []
        for scope, emissions in emissions_scopes.dict().items():
            if emissions:
                ems = {em['year']: em['value'] for em in emissions}
                data.append({ColumnsConfig.COMPANY_ID: id, ColumnsConfig.VARIABLE: VariablesConfig.EMISSIONS,
                             ColumnsConfig.SCOPE: EScope[scope], **ems})
        return data

    def _historic_ei_to_dicts(self, id: str, intensities_scopes: IHistoricEIScopes) \
            -> List[Dict[str, str]]:
        data = []
        for scope, intensities in intensities_scopes.dict().items():
            if intensities:
                intsties = {intsty['year']: intsty['value'] for intsty in intensities}
                data.append(
                    {ColumnsConfig.COMPANY_ID: id, ColumnsConfig.VARIABLE: VariablesConfig.EMISSIONS_INTENSITIES,
                     ColumnsConfig.SCOPE: EScope[scope], **intsties})
        return data

    # Each benchmark defines its own scope requirements on a per-sector/per-region basis.
    # The purpose of this function is not to infer scope data that might be interesting,
    # but rather to impute the scope data that is actually required, no more, no less.
    def _compute_missing_historic_ei(self, companies: List[ICompanyData], historic_data: pd.DataFrame):
        scopes = EScope.get_result_scopes()
        missing_data = []
        for company in companies:
            # Create keys to index historic_data DataFrame for readability
            production_key = (company.company_id, VariablesConfig.PRODUCTIONS, 'Production')
            emissions_keys = {scope: (company.company_id, VariablesConfig.EMISSIONS, scope) for scope in scopes}
            ei_keys = {scope: (company.company_id, VariablesConfig.EMISSIONS_INTENSITIES, scope) for scope in scopes}
            this_missing_data = []
            append_this_missing_data = True
            for scope in scopes:
                if ei_keys[scope] in historic_data.index:
                    append_this_missing_data = False
                    continue
                # Emissions intensities not yet computed for this scope
                if scope == EScope.S1S2:
                    try:  # Try to add S1 and S2 emissions intensities
                        historic_data.loc[ei_keys[scope]] = (
                            historic_data.loc[(company.company_id, VariablesConfig.EMISSIONS_INTENSITIES, EScope.S1)]
                            + historic_data.loc[(company.company_id, VariablesConfig.EMISSIONS_INTENSITIES, EScope.S2)])
                        append_this_missing_data = False
                    except KeyError:  # Either S1 or S2 emissions intensities not readily available
                        try:  # Try to compute S1+S2 EIs from S1+S2 emissions and productions
                            historic_data.loc[ei_keys[scope]] = (
                                historic_data.loc[emissions_keys[scope]] / historic_data.loc[production_key])
                            append_this_missing_data = False
                        except KeyError:
                            this_missing_data.append(f"{company.company_id} - {scope.name}")
                elif scope == EScope.S1S2S3:  # Implement when S3 data is available
                    try:  # Try to add S1S2 and S3 emissions intensities
                        historic_data.loc[ei_keys[scope]] = (
                            historic_data.loc[(company.company_id, VariablesConfig.EMISSIONS_INTENSITIES, EScope.S1S2)]
                            + historic_data.loc[(company.company_id, VariablesConfig.EMISSIONS_INTENSITIES, EScope.S3)])
                        append_this_missing_data = False
                    except KeyError:  # Either S1 or S2 emissions intensities not readily available
                        try:  # Try to compute S1+S2 EIs from S1+S2 emissions and productions
                            historic_data.loc[ei_keys[scope]] = (
                                historic_data.loc[emissions_keys[scope]] / historic_data.loc[production_key])
                            append_this_missing_data = False
                        except KeyError:
                            this_missing_data.append(f"{company.company_id} - {scope.name}")
                else:  # S1, S2, and S3 cannot be computed from other EIs, so use emissions and productions
                    try:
                        historic_data.loc[ei_keys[scope]] = (
                            historic_data.loc[emissions_keys[scope]] / historic_data.loc[production_key])
                        append_this_missing_data = False
                    except KeyError:
                        this_missing_data.append(f"{company.company_id} - {scope.name}")
            # This only happens if ALL scope data is missing.  If ANY scope data is present, we'll work with what we get.
            if this_missing_data and append_this_missing_data:
                missing_data.extend(this_missing_data)
        if missing_data:
            error_message = f"Provide either historic emissions intensity data, or historic emission and " \
                            f"production data for these company - scope combinations: {missing_data}"
            logger.error(error_message)
            raise ValueError(error_message)

    def _add_projections_to_companies(self, companies: List[ICompanyData], extrapolations: pd.DataFrame):
        for company in companies:
            scope_projections = {}
            scope_dfs = {}
            scope_names = EScope.get_scopes()
            for scope_name in scope_names:
                if not company.historic_data.emissions_intensities or not getattr(company.historic_data.emissions_intensities, scope_name):
                    scope_projections[scope_name] = None
                    continue
                results = extrapolations.loc[(company.company_id, VariablesConfig.EMISSIONS_INTENSITIES, EScope[scope_name])]
                results = asPintSeries(results)
                if not isinstance(results.dtype, PintType):
                    if results.isna().all():
                        scope_projections[scope_name] = None
                        continue
                    assert False
                units = f"{results.dtype.units:~P}"
                scope_dfs[scope_name] = results
                scope_projections[scope_name] = ICompanyEIProjections(ei_metric=units,
                                                                      projections=self._get_bounded_projections(results))
            if scope_projections['S1'] and scope_projections['S2'] and not scope_projections['S1S2']:
                results = scope_dfs['S1'] + scope_dfs['S2']
                units = f"{results.values[0].u:~P}"
                scope_dfs['S1S2'] = results
                scope_projections['S1S2'] = ICompanyEIProjections(ei_metric=units,
                                                                  projections=self._get_bounded_projections(results))
            if scope_projections['S1S2'] and scope_projections['S3'] and not scope_projections['S1S2S3']:
                results = scope_dfs['S1S2'] + scope_dfs['S3']
                units = f"{results.values[0].u:~P}"
                # We don't need to compute scope_dfs['S1S2S3'] because nothing further depends on accessing it here
                scope_projections['S1S2S3'] = ICompanyEIProjections(ei_metric=units,
                                                                    projections=self._get_bounded_projections(results))
            company.projected_intensities = ICompanyEIProjectionsScopes(**scope_projections)

    def _standardize(self, intensities: pd.DataFrame) -> pd.DataFrame:
        # When columns are years and rows are all different intensity types, we cannot winsorize
        # Transpose the dataframe, winsorize the columns (which are all coherent because they belong to a single variable/company), then transpose again
        intensities = intensities.T
        if ITR.HAS_UNCERTAINTIES:
            for col in intensities.columns:
                pa = PA_._from_sequence(intensities[col])
                pa_units = pa[0].u
                pa_nans = ITR.isnan(pa.data)
                if pa_nans.any():
                    if [isinstance(pt.m, ITR.UFloat) for pt in pa[np.where(pa_nans)]]:
                        u_pa_data = [ITR._ufloat_nan if pn else pt if isinstance(pt, ITR.UFloat) else ITR.ufloat(pt, 0)
                                     for pt, pn in zip(pa.data, pa_nans)]
                        intensities[col] = pd.Series(PA_(u_pa_data, pa_units), dtype=pa.dtype,
                                                     index=intensities[col].index, name=intensities[col].name)

        # At the starting point, we expect that if we have S1, S2, and S1S2 intensities, that S1+S2 = S1S2
        # After winsorization, this is no longer true, because S1 and S2 will be clipped differently than S1S2.
        winsorized_intensities: pd.DataFrame = self._winsorize(intensities)
        for col in winsorized_intensities.columns:
            winsorized_intensities[col] = winsorized_intensities[col].astype(intensities[col].dtype)
        standardized_intensities: pd.DataFrame = self._interpolate(winsorized_intensities)
        with warnings.catch_warnings():
            # Don't worry about warning that we are intentionally dropping units as we transpose
            warnings.simplefilter("ignore")
            return standardized_intensities.T

    def _winsorize(self, historic_intensities: pd.DataFrame) -> pd.DataFrame:
        # quantile doesn't handle pd.NA inside Quantity; FIXME: we can use np.nan because not expecting UFloat in input data

        # Turns out we have to dequantify here: https://github.com/pandas-dev/pandas/issues/45968
        # Can try again when ExtensionArrays are supported by `quantile`, `clip`, and friends
        units = historic_intensities.apply(lambda x: x[x.first_valid_index()].u
                                           if x.first_valid_index() and isinstance(x[x.first_valid_index()], pint.Quantity) else None)
        if ITR.HAS_UNCERTAINTIES:
            nominal_intensities = historic_intensities.apply(lambda x: ITR.nominal_values(x.map(lambda y: y.m if isinstance(y, pint.Quantity) else np.nan)))
            uncertain_intensities = historic_intensities.apply(lambda x: ITR.std_devs(x.map(lambda y: y.m if isinstance(y, pint.Quantity) else 0)))
        else:
            nominal_intensities = historic_intensities.apply(lambda x: x.map(lambda y: y.m if isinstance(y, pint.Quantity) else np.nan))
        # See https://github.com/hgrecco/pint-pandas/issues/114
        lower=nominal_intensities.quantile(q=self.projection_controls.LOWER_PERCENTILE, axis='index', numeric_only=False)
        upper=nominal_intensities.quantile(q=self.projection_controls.UPPER_PERCENTILE, axis='index', numeric_only=False)
        winsorized: pd.DataFrame = nominal_intensities.clip(
            lower=lower,
            upper=upper,
            axis='columns'
        )
        if ITR.HAS_UNCERTAINTIES:
            # FIXME: the clipping process can properly introduce uncertainties.  The low and high values that are clipped could be
            # replaced by the clipped values +/- the lower and upper percentile values respectively.
            if uncertain_intensities.values.sum() != 0:
                uwinsorized = winsorized.apply(lambda x: PA_(ITR.uarray(x.values.data, uncertain_intensities[x.name].values), dtype=units[x.name]))
                return uwinsorized
        # FIXME: If we have S1, S2, and S1S2 intensities, should we treat winsorized(S1)+winsorized(S2) as winsorized(S1S2)?
        # FIXME: If we have S1S2 (or S1 and S2) and S3 and S1S23 intensities, should we treat winsorized(S1S2)+winsorized(S3) as winsorized(S1S2S3)?
        winsorized_and_unitized = winsorized.apply(lambda x: PA_(x.values.data, dtype=units[x.name]))
        return winsorized_and_unitized

    def _interpolate(self, historic_intensities: pd.DataFrame) -> pd.DataFrame:
        # Interpolate NaNs surrounded by values, but don't extrapolate NaNs with last known value
        df = historic_intensities.copy()
        for col in df.columns:
            pa = PA_._from_sequence(df[col])
            if ITR.isnan(pa.data).all():
                continue
            # pd.Series.interpolate only works on numeric data, so push down into PintArray
            # FIXME: throw some uncertainty into the mix.  If we see ... X NA Y ... and interpolat the NA as Z as function of X and Y
            ser = pd.Series(data=pa.data, index=df.index)
            df[col] = pd.Series(PA_(ser.interpolate(method='linear', inplace=False, limit_direction='forward', limit_area='inside').values,
                                    dtype=pa.units),
                                dtype=pa.dtype, index=df.index)
        return df

    def _get_trends(self, intensities: pd.DataFrame):
        # Compute year-on-year growth ratios of emissions intensities
        # Transpose so we can work with homogeneous units in columns.  This means rows are years.
        intensities = intensities.T
        for col in intensities.columns:
            # ratios are dimensionless, so get rid of units, which confuse rolling/apply.  Some columns are NaN-only
            intensities[col] = intensities[col].map(lambda x: x.m if isinstance(x, pint.Quantity) else x)
            # FIXME: rolling windows require conversion to float64.  Don't want to be a nuisance...
            if ITR.HAS_UNCERTAINTIES:
                intensities[col] = ITR.nominal_values(intensities[col])
        ratios: pd.DataFrame = intensities.rolling(window=2, axis='index', closed='right') \
            .apply(func=self._year_on_year_ratio, raw=True)
        # # Add weight to trend movements across multiple years (normalized to year-over-year, not over two years...)
        # # FIXME: we only want to do this for median, not mean.
        # if self.projection_controls.TREND_CALC_METHOD==pd.DataFrame.median:
        #     ratios_2 = ratios
        #     ratios_3: pd.DataFrame = intensities.rolling(window=3, axis='index', closed='right') \
        #         .apply(func=self._year_on_year_ratio, raw=True).div(2.0)
        #     ratios = pd.concat([ratios_2, ratios_3])
        # elif self.projection_controls.TREND_CALC_METHOD==pd.DataFrame.mean:
        #     pass
        # else:
        #     raise ValueError("Unhanlded TREND_CALC_METHOD")

        trends: pd.DataFrame = self.projection_controls.TREND_CALC_METHOD(ratios, axis='index', skipna=True).clip(
            lower=self.projection_controls.LOWER_DELTA,
            upper=self.projection_controls.UPPER_DELTA,
        )
        return trends.T

    def _extrapolate(self, trends: pd.DataFrame, projection_years: range, historic_data: pd.DataFrame) -> pd.DataFrame:
        projected_intensities = historic_data.loc[historic_data.index.intersection(trends.index)].copy()
        # We need to do a mini-extrapolation if we don't have complete historic data
        # These columns are heterogeneous as to units, so don't try to use PintArrays
        for year in historic_data.columns.tolist()[:-1]:
            # FIXME: need a version of ITR.isnan that can see into Quantities
            mask = ITR.isnan(projected_intensities[year + 1].map(lambda x: x.m if isinstance(x, pint.Quantity) else np.nan))
            projected_intensities.loc[mask, year + 1] = projected_intensities.loc[mask, year] * (1 + trends.loc[mask])

        # Now the big extrapolation
        for year in projection_years:
            projected_intensities[year + 1] = projected_intensities[year] * (1 + trends)
        # Clean up rows by converting NaN/None into Quantity(np.nan, unit_type)
        columnwise_intensities = ITR.data.osc_units.asPintDataFrame(projected_intensities.T)
        # Restore row-wise shape of DataFrame
        projected_intensities =  columnwise_intensities.T
        return projected_intensities

    # Might return a float, might return a ufloat
    def _year_on_year_ratio(self, arr: np.ndarray):
        return (arr[-1] / arr[0]) - 1.0


class EITargetProjector(EIProjector):
    """
    This class projects emissions intensities from a company's targets and historic data. Targets are specified per
    scope in terms of either emissions or emission intensity reduction. Interpolation between last known historic data
    and (a) target(s) is CAGR-based, but not entirely CAGR (beacuse zero can only be approached asymptotically
    and any CAGR that approaches zero in finite time must have extraordinarily steep initial drop, which is unrealistic).

    Remember that pd.Series are always well-behaved with pint[] quantities.  pd.DataFrame columns are well-behaved,
    but data across columns is not always well-behaved.  We therefore make this function assume we are projecting targets
    for a specific company, in a specific sector.  If we want to project targets for multiple sectors, we have to call it multiple times.
    This function doesn't need to know what sector it's computing for...only tha there is only one such, for however many scopes.
    """

    def __init__(self, projection_controls: ProjectionControls = ProjectionControls()):
        self.projection_controls = projection_controls

    def _order_scope_targets(self, scope_targets):
        if not scope_targets:
            # Nothing to do
            return scope_targets
        # If there are multiple targets that land on the same year for the same scope, choose the most recently set target
        unique_target_years = [(target.target_end_year, target.target_start_year) for target in scope_targets]
        # This sorts targets into ascending target years and descending start years
        unique_target_years.sort(key=lambda t: (t[0], -t[1]))
        # Pick the first target year most recently articulated, preserving ascending order of target yeares
        unique_target_years = [(uk, next(v for k, v in unique_target_years if k == uk)) for uk in
                               dict(unique_target_years).keys()]
        # Now use those pairs to select just the targets we want
        unique_scope_targets = [unique_targets[0] for unique_targets in \
                                [[target for target in scope_targets if
                                  (target.target_end_year, target.target_start_year) == u] \
                                 for u in unique_target_years]]
        unique_scope_targets.sort(key=lambda target: (target.target_end_year))

        # We only trust the most recently communicated netzero target, but prioritize the most recently communicated, most aggressive target
        netzero_scope_targets = [target for target in unique_scope_targets if target.netzero_year]
        netzero_scope_targets.sort(key=lambda t: (-t.target_start_year, t.netzero_year))
        if netzero_scope_targets:
            netzero_year = netzero_scope_targets[0].netzero_year
            for target in unique_scope_targets:
                target.netzero_year = netzero_year
        return unique_scope_targets

    def calculate_nz_target_years(self, targets: List[ITargetData]) -> dict:
        """Input:
        @target: A list of stated carbon reduction targets
        @returns: A dict of SCOPE_NAME: NETZERO_YEAR pairs
        """
        # We first try to find the earliest netzero year target for each scope
        nz_target_years = {'S1': 9999, 'S2': 9999, 'S1S2': 9999, 'S3': 9999, 'S1S2S3': 9999}
        for target in targets:
            scope_name = target.target_scope.name
            if target.netzero_year < nz_target_years[scope_name]:
                nz_target_years[scope_name] = target.netzero_year
            if target.target_reduction_pct == 1.0 and target.target_end_year < nz_target_years[scope_name]:
                nz_target_years[scope_name] = target.target_end_year

        # We then infer netzero year targets for constituents of compound scopes from compound scopes
        # and infer netzero year taregts for compound scopes as the last of all constituents
        if nz_target_years['S1S2S3'] < nz_target_years['S1S2']:
            logger.warn(f"target S1S2S3 date <= S1S2 date")
            nz_target_years['S1S2'] = nz_target_years['S1S2S3']
        nz_target_years['S1'] = min(nz_target_years['S1S2'], nz_target_years['S1'])
        nz_target_years['S2'] = min(nz_target_years['S1S2'], nz_target_years['S2'])
        nz_target_years['S1S2'] = min(nz_target_years['S1S2'], max(nz_target_years['S1'], nz_target_years['S2']))
        nz_target_years['S3'] = min(nz_target_years['S1S2S3'], nz_target_years['S3'])
        # nz_target_years['S1S2'] and nz_target_years['S3'] must both be <= nz_target_years['S1S2S3'] at this point
        nz_target_years['S1S2S3'] = max(nz_target_years['S1S2'], nz_target_years['S3'])
        return {scope_name: nz_year if nz_year<9999 else None for scope_name, nz_year in nz_target_years.items()}

    def project_ei_targets(self, company: ICompanyData, production_proj: pd.Series) -> ICompanyEIProjectionsScopes:
        """Input:
        @company: Company-specific data: target_data and base_year_production
        @production_proj: company's production projection computed from region-sector benchmark growth rates

        If the company has no target or the target can't be processed, then the output the emission database, unprocessed
        If successful, it returns the full set of historic emissions intensities and projections based on targets
        """
        targets = company.target_data
        ei_projection_scopes = {'S1': None, 'S2': None, 'S1S2': None, 'S3': None, 'S1S2S3': None}
        nz_target_years = self.calculate_nz_target_years(targets)

        for scope_name in ei_projection_scopes:
            netzero_year = nz_target_years[scope_name]
            # If there are no other targets specified (which can happen when we are dealing with inferred netzero targets)
            # target_year and target_ei_value pick up the year and value of the last EI realized
            # Otherwise, they are specified by the targets (intensity or absolute)
            target_year = None
            target_ei_value = None

            scope_targets = [target for target in targets if target.target_scope.name == scope_name]
            # If we don't have an explicit scope target but we do have an implicit netzero target that applies to this scope,
            # prime the pump for projecting that netzero target, in case we ever need such a projection.  For example,
            # a netzero target for S1+S2 implies netzero targets for both S1 and S2.  The TPI benchmark needs an S1 target
            # for some sectors, and projecting a netzero target for S1 from S1+S2 makes that benchmark useable.
            # Note that we can only infer separate S1 and S2 targets from S1+S2 targets when S1+S2 = 0, because S1=0 + S2=0 is S1+S2=0
            if not scope_targets:
                if company.historic_data is None:
                    # This just defends against poorly constructed test cases
                    nz_target_years[scope_name] = None
                    continue
                if nz_target_years[scope_name]:
                    ei_realizations = getattr(company.historic_data.emissions_intensities, scope_name)
                    # We can infer a netzero target.  Use our last year historic year of data as the target_year (i.e., target_base_year) value
                    # Due to ragged right edge, we have to hunt.  But we know there's at least one such value.
                    # If there's a proper target for this scope, historic values will be replaced by target values
                    for i in range(len(ei_realizations)-1, -1, -1):
                        target_ei_value = ei_realizations[i].value
                        if not ITR.isnan(target_ei_value.m):
                            target_year = ei_realizations[i].year
                            break
                    if target_year is None:
                        # Either no realizations or they are all NaN
                        continue
                    # FIXME: if we have aggressive targets for source of this inference, the inferred
                    # netzero targets may be very slack (because non-netzero targets are not part of the inference)
            scope_targets_intensity = self._order_scope_targets(
                [target for target in scope_targets if target.target_type == "intensity"])
            scope_targets_absolute = self._order_scope_targets(
                [target for target in scope_targets if target.target_type == "absolute"])
            while scope_targets_intensity or scope_targets_absolute:
                if scope_targets_intensity and scope_targets_absolute:
                    target_i = scope_targets_intensity[0]
                    target_a = scope_targets_absolute[0]
                    if target_i.target_end_year == target_a.target_end_year:
                        if target_i.target_start_year >= target_a.target_start_year:
                            if target_i.target_start_year == target_a.target_start_year:
                                warnings.warn(
                                    f"intensity target overrides absolute target for target_start_year={target_i.target_start_year} and target_end_year={target_i.target_end_year}")
                            scope_targets_absolute.pop(0)
                            scope_targets = scope_targets_intensity
                        else:
                            scope_targets_intensity.pop(0)
                            scope_targets = scope_targets_absolute
                    elif target_i.target_end_year < target_a.target_end_year:
                        scope_targets = scope_targets_intensity
                    else:
                        scope_targets = scope_targets_absolute
                elif not scope_targets_intensity:
                    scope_targets = scope_targets_absolute
                else:  # not scope_targets_absolute
                    scope_targets = scope_targets_intensity

                target = scope_targets.pop(0)
                base_year = target.target_base_year

                # Put these variables into scope
                last_ei_year = None
                last_ei_value = None

                # Solve for intensity and absolute
                model_ei_projections = None
                if target.target_type == "intensity":
                    # Simple case: the target is in intensity
                    # If target is not the first one for this scope, we continue from last year of the previous target
                    if ei_projection_scopes[scope_name]:
                        (_, last_ei_year), (_, last_ei_value) = ei_projection_scopes[scope_name].projections[-1]
                        last_ei_value = last_ei_value.to(target.target_base_year_unit)
                        skip_first_year = 1
                    else:
                        # When starting from scratch, use recent historic data if available.
                        if not company.historic_data:
                            ei_realizations = []
                        else:
                            ei_realizations = getattr(company.historic_data.emissions_intensities, scope_name)
                        skip_first_year = 0
                        if ei_realizations == []:
                            # Alas, we have no data to align with constituent or containing scope
                            last_ei_year = target.target_base_year
                            last_ei_value = Q_(target.target_base_year_qty, target.target_base_year_unit)
                        else:
                            for i in range(len(ei_realizations)-1, -1, -1):
                                last_ei_year, last_ei_value = ei_realizations[i].year, ei_realizations[i].value
                                if ITR.isnan(last_ei_value.m):
                                    continue
                                model_ei_projections = [ICompanyEIProjection(year=ei_realizations[j].year, value=ei_realizations[j].value)
                                                        for j in range(0,i+1)
                                                        if not ITR.isnan(ei_realizations[j].value.m)]
                                ei_projection_scopes[scope_name] = ICompanyEIProjections(ei_metric=EI_Quantity(f"{last_ei_value.u:~P}"),
                                                                                         projections=self._get_bounded_projections(model_ei_projections))
                                skip_first_year = 1
                                break
                            if last_ei_year < target.target_base_year:
                                logger.error(f"Target data for {company.company_id} more up-to-date than disclosed data; please fix and re-run")
                                breakpoint()
                                raise ValueError
                    target_year = target.target_end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_ei_value = Q_(target.target_base_year_qty * (1 - target.target_reduction_pct),
                                         target.target_base_year_unit)
                    if target_ei_value >= last_ei_value:
                        # We've already achieved target, so aim for the next one
                        target_year = last_ei_year
                        target_ei_value = last_ei_value
                        continue
                    CAGR = self._compute_CAGR(last_ei_year, last_ei_value, target_year, target_ei_value)
                    model_ei_projections = [ICompanyEIProjection(year=year, value=CAGR[year])
                                            for year in range(last_ei_year+skip_first_year, 1+target_year)
                                            if year >= self.projection_controls.BASE_YEAR]

                elif target.target_type == "absolute":
                    # Complicated case, the target must be switched from absolute value to intensity.
                    # We use benchmark production data

                    # If target is not the first one for this scope, we continue from last year of the previous target
                    if ei_projection_scopes[scope_name]:
                        (_, last_ei_year), (_, last_ei_value) = ei_projection_scopes[scope_name].projections[-1]
                        last_prod_value = production_proj.loc[last_ei_year]
                        last_em_value = last_ei_value * last_prod_value
                        last_em_value = last_em_value.to(target.target_base_year_unit)
                        skip_first_year = 1
                    else:
                        if not company.historic_data:
                            em_realizations = []
                        else:
                            em_realizations = getattr(company.historic_data.emissions, scope_name)
                        skip_first_year = 0
                        if em_realizations == []:
                            last_ei_year = target.target_base_year
                            last_em_value = Q_(target.target_base_year_qty, target.target_base_year_unit)
                            # FIXME: should be target.base_year_production !!
                            last_prod_value = company.base_year_production
                        else:
                            for i in range(len(em_realizations)-1, -1, -1):
                                last_ei_year, last_em_value = em_realizations[i].year, em_realizations[i].value
                                if ITR.isnan(last_em_value.m):
                                    continue
                                model_ei_projections = [ICompanyEIProjection(year=em_realizations[j].year, value=em_realizations[j].value / production_proj.loc[em_realizations[j].year])
                                                        for j in range(0,i+1)
                                                        if em_realizations[j].year in production_proj.index and not ITR.isnan(em_realizations[j].value.m)]
                                last_prod_value = production_proj.loc[last_ei_year]
                                ei_projection_scopes[scope_name] = ICompanyEIProjections(ei_metric=EI_Quantity(f"{(last_em_value/last_prod_value).u:~P}"),
                                                                                         projections=self._get_bounded_projections(model_ei_projections))
                                skip_first_year = 1
                                break
                            assert last_ei_year >= target.target_base_year
                        # FIXME: just have to trust that this particular value ties to the first target's year/value pair
                        try:
                            last_ei_value = last_em_value / last_prod_value
                        except UnboundLocalError:
                            breakpoint()

                    target_year = target.target_end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_em_value = Q_(target.target_base_year_qty * (1 - target.target_reduction_pct),
                                      target.target_base_year_unit)
                    if target_em_value >= last_em_value:
                        # We've already achieved target, so aim for the next one
                        target_year = last_ei_year
                        target_ei_value = last_ei_value
                        continue
                    CAGR = self._compute_CAGR(last_ei_year, last_em_value, target_year, target_em_value)

                    model_emissions_projections = CAGR.loc[(last_ei_year+skip_first_year):target_year]
                    emissions_projections = model_emissions_projections.astype(f'pint[{target.target_base_year_unit}]')
                    idx = production_proj.index.intersection(emissions_projections.index)
                    ei_projections = emissions_projections.loc[idx] / production_proj.loc[idx]

                    model_ei_projections = [ICompanyEIProjection(year=year, value=ei_projections[year])
                                            for year in range(last_ei_year+skip_first_year, 1+target_year)
                                            if year >= self.projection_controls.BASE_YEAR]
                else:
                    # No target (type) specified
                    ei_projection_scopes[scope_name] = None
                    continue

                target_ei_value = model_ei_projections[-1].value
                if ei_projection_scopes[scope_name] is not None:
                    ei_projection_scopes[scope_name].projections.extend(model_ei_projections)
                else:
                    ei_projection_scopes[scope_name] = ICompanyEIProjections(ei_metric=EI_Quantity (f"{target_ei_value.u:~P}"),
                                                                             projections=self._get_bounded_projections(model_ei_projections))

                if scope_targets_intensity and scope_targets_intensity[0].netzero_year:
                    # Let a later target set the netzero year
                    continue
                if scope_targets_absolute and scope_targets_absolute[0].netzero_year:
                    # Let a later target set the netzero year
                    continue

            # Handle final netzero targets.  Note that any absolute zero target is also zero intensity target (so use target_ei_value)
            # TODO What if target is a 100% reduction.  Does it work whether or not netzero_year is set?
            if netzero_year and netzero_year > target_year:  # add in netzero target at the end
                netzero_qty = Q_(0.0, target_ei_value.u)
                CAGR = self._compute_CAGR(target_year, target_ei_value, netzero_year, netzero_qty)
                ei_projections = [ICompanyEIProjection(year=year, value=CAGR[year])
                                  for year in range(1 + target_year, 1 + netzero_year)]
                if ei_projection_scopes[scope_name]:
                    ei_projection_scopes[scope_name].projections.extend(ei_projections)
                else:
                    ei_projection_scopes[scope_name] = ICompanyEIProjections(projections=self._get_bounded_projections(ei_projections),
                                                                             ei_metric=EI_Quantity (f"{target_ei_value.u:~P}"))
                target_year = netzero_year
                target_ei_value = netzero_qty
            if ei_projection_scopes[scope_name] and target_year < ProjectionControls.TARGET_YEAR:
                # Assume everything stays flat until 2050
                ei_projection_scopes[scope_name].projections.extend(
                    [ICompanyEIProjection(year=year, value=target_ei_value)
                     for y, year in enumerate(range(1 + target_year, 1 + ProjectionControls.TARGET_YEAR))]
                )

        return ICompanyEIProjectionsScopes(**ei_projection_scopes)

    def _compute_CAGR(self, first_year: int, first_value: pint.Quantity, last_year: int, last_value: pint.Quantity) -> pd.Series:
        """Compute CAGR, returning pd.Series of the growth (or reduction) applied to first to converge with last
        :param first_year: the year of the first datapoint in the Calculation (most recent actual datapoint)
        :param first_value: the value of the first datapoint in the Calculation (most recent actual datapoint)
        :param last_year: the year of the final target
        :param last_value: the value of the final target

        :return: pd.Series index by the years from first_year:last_year, with units based on last_value (the target value)
        """

        period = last_year - first_year
        if period <= 0:
            return pd.Series(PA_([], dtype=f"pint[{first_value.u:~P}]"))
        if last_value >= first_value or first_value.m == 0:
            # If we have a slack target, i.e., target goal is actually above current data, clamp so CAGR computes as zero
            return pd.Series(PA_([first_value.m] * (period+1), dtype=f"{first_value.u:~P}"),
                             index=range(first_year, last_year+1),
                             name='CAGR')

        # CAGR doesn't work well with large reductions, so solve with cases:
        CAGR_limit = 1/11.11
        # PintArrays make it easy to convert arrays of magnitudes to types, so ensure magnitude consistency
        first_value = first_value.to(last_value.u)
        if last_value < first_value * CAGR_limit:
            # - If CAGR target > 90% reduction, blend a linear reduction with CAGR to get CAGR-like shape that actually hits the target
            cagr_factor = CAGR_limit ** (1 / period)
            cagr_data = [cagr_factor ** y * first_value.m - (CAGR_limit * first_value.m - last_value.m) * (y / period)
                         for y, year in enumerate(range(first_year, last_year+1))]
        else:
            # - If CAGR target <= 90% reduction, use CAGR model directly
            cagr_factor = (last_value / first_value).m ** (1 / period)
            cagr_data = [cagr_factor ** y * first_value.m
                         for y, year in enumerate(range(first_year, last_year+1))]
        cagr_result = pd.Series(PA_(cagr_data, dtype=f"{last_value.u:~P}"),
                                index=range(first_year, last_year+1),
                                name='CAGR')
        return cagr_result
