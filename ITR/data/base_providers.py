import warnings  # needed until quantile behaves better with Pint quantities in arrays
import numpy as np
import pandas as pd
from functools import reduce, partial
from typing import List, Type, Dict
import logging

from ITR.data.osc_units import Q_, PA_
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, VariablesConfig, LoggingConfig
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, \
    IntensityBenchmarkDataProvider
from ITR.interfaces import ICompanyData, EScope, IProductionBenchmarkScopes, IEIBenchmarkScopes, \
    IBenchmark, IProjection, ICompanyEIProjections, ICompanyEIProjectionsScopes, IHistoricEIScopes, \
    IHistoricEmissionsScopes, IProductionRealization, ITargetData, IHistoricData, ICompanyEIProjection, \
    IEmissionRealization, IntensityMetric, ProjectionControls

# TODO handling of scopes in benchmarks

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BaseProviderProductionBenchmark(ProductionBenchmarkDataProvider):

    def __init__(self, production_benchmarks: IProductionBenchmarkScopes,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        """
        Base provider that relies on pydantic interfaces. Default for FastAPI usage
        :param production_benchmarks: List of IProductionBenchmarkScopes
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
        """
        super().__init__()
        self.temp_config = tempscore_config
        self.column_config = column_config
        self._productions_benchmarks = production_benchmarks

    # Note that bencharmk production series are dimensionless.
    def _convert_benchmark_to_series(self, benchmark: IBenchmark) -> pd.Series:
        """
        extracts the company projected intensity or production targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        return pd.Series({r.year: r.value for r in benchmark.projections}, name=(benchmark.region, benchmark.sector),
                         dtype=f'pint[{benchmark.benchmark_metric.units}]')

    # Production benchmarks are dimensionless.  S1S2 has nothing to do with any company data.
    # It's a label in the top-level of benchmark data.  Currently S1S2 is the only label with any data.
    def _get_projected_production(self, scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Converts IProductionBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: pd.DataFrame
        """
        result = []
        for bm in self._productions_benchmarks.dict()[str(scope)]['benchmarks']:
            result.append(self._convert_benchmark_to_series(IBenchmark.parse_obj(bm)))
        df_bm = pd.DataFrame(result)
        df_bm.index.names = [self.column_config.REGION, self.column_config.SECTOR]

        return df_bm

    def get_company_projected_production(self, company_sector_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for list of companies in ghg_scope12
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: DataFrame of projected productions for [base_year - base_year + 50]
        """
        benchmark_production_projections = self.get_benchmark_projections(company_sector_region_info)
        company_production = company_sector_region_info[self.column_config.BASE_YEAR_PRODUCTION]
        return benchmark_production_projections.add(1).cumprod(axis=1).mul(
            company_production, axis=0)

    def get_benchmark_projections(self, company_sector_region_info: pd.DataFrame,
                                  scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with production benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :param scope: a scope
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        benchmark_projection = self._get_projected_production(scope)  # TODO optimize performance
        sectors = company_sector_region_info[self.column_config.SECTOR]
        regions = company_sector_region_info[self.column_config.REGION]
        benchmark_regions = regions.copy()
        mask = benchmark_regions.isin(benchmark_projection.reset_index()[self.column_config.REGION])
        benchmark_regions.loc[~mask] = "Global"

        benchmark_projection = benchmark_projection.loc[list(zip(benchmark_regions, sectors))]
        benchmark_projection.index = sectors.index
        return benchmark_projection


class BaseProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(self, EI_benchmarks: IEIBenchmarkScopes,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__(EI_benchmarks.benchmark_temperature, EI_benchmarks.benchmark_global_budget,
                         EI_benchmarks.is_AFOLU_included)
        self._EI_benchmarks = EI_benchmarks
        self.temp_config = tempscore_config
        self.column_config = column_config

    def get_SDA_intensity_benchmarks(self, company_info_at_base_year: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_info_at_base_year: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        intensity_benchmarks = self._get_intensity_benchmarks(company_info_at_base_year)
        decarbonization_paths = self._get_decarbonizations_paths(intensity_benchmarks)
        last_ei = intensity_benchmarks[self.temp_config.CONTROLS_CONFIG.target_end_year]
        ei_base = company_info_at_base_year[self.column_config.BASE_EI]
        df = decarbonization_paths.mul((ei_base - last_ei), axis=0)
        df = df.add(last_ei, axis=0).astype(ei_base.dtype)
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
        first_ei = intensity_benchmark_row[self.temp_config.CONTROLS_CONFIG.base_year]
        last_ei = intensity_benchmark_row[self.temp_config.CONTROLS_CONFIG.target_end_year]
        # TODO: does this still throw a warning when processing a NaN?  convert to base units before accessing .magnitude
        return intensity_benchmark_row.apply(lambda x: (x - last_ei) / (first_ei - last_ei))

    def _convert_benchmark_to_series(self, benchmark: IBenchmark) -> pd.Series:
        """
        extracts the company projected intensities or targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        return pd.Series({p.year: p.value for p in benchmark.projections}, name=(benchmark.region, benchmark.sector),
                         dtype=f'pint[{benchmark.benchmark_metric.units}]')

    def _get_projected_intensities(self, scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Converts IEmissionIntensityBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: pd.DataFrame
        """
        results = []
        for bm in self._EI_benchmarks.__getattribute__(str(scope)).benchmarks:
            results.append(self._convert_benchmark_to_series(bm))
        with warnings.catch_warnings():
            # pd.DataFrame.__init__ (in pandas/core/frame.py) ignores the beautiful dtype information adorning the pd.Series list elements we are providing.  Sad!
            warnings.simplefilter("ignore")
            df_bm = pd.DataFrame(results)
        df_bm.index.names = [self.column_config.REGION, self.column_config.SECTOR]
        return df_bm

    def _get_intensity_benchmarks(self, company_sector_region_info: pd.DataFrame,
                                  scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with production benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :param scope: a scope
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        benchmark_projection = self._get_projected_intensities(scope)  # TODO optimize performance
        reg_sec = company_sector_region_info[[self.column_config.REGION,self.column_config.SECTOR]].copy()
        merged_df=reg_sec.reset_index().merge(benchmark_projection.reset_index()[[self.column_config.REGION,self.column_config.SECTOR]], how='left', indicator=True).set_index('index') # checking which combinations of reg-sec are missing in the benchmark
        reg_sec.loc[merged_df._merge == 'left_only', self.column_config.REGION] = "Global" # change region in missing combination to "Global"
        sectors = reg_sec.sector
        regions = reg_sec.region
        benchmark_projection = benchmark_projection.loc[list(zip(regions, sectors))]
        benchmark_projection.index = sectors.index
        return benchmark_projection


class BaseCompanyDataProvider(CompanyDataProvider):
    """
    Data provider skeleton for JSON files parsed by the fastAPI json encoder. This class serves primarily for connecting
    to the ITR tool via API.

    :param companies: A list of ICompanyData objects that each contain fundamental company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
    """

    def __init__(self,
                 companies: List[ICompanyData],
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig,
                 projection_controls: ProjectionControls = ProjectionControls()):
        super().__init__()
        self.column_config = column_config
        self.temp_config = tempscore_config
        self.projection_controls = projection_controls
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

    # Because this presently defaults to S1S2 always, targets spec'd for S1 only ro S1+S2+S3 are not well-handled.
    def _convert_projections_to_series(self, company: ICompanyData, feature: str,
                                       scope: EScope = EScope.S1S2) -> pd.Series:
        """
        extracts the company projected intensities or targets for a given scope
        :param feature: PROJECTED_TRAJECTORIES or PROJECTED_TARGETS (both are intensities)
        :param scope: a scope
        :return: pd.Series
        """
        company_dict = company.dict()
        production_units = company_dict[self.column_config.PRODUCTION_METRIC]['units']
        emissions_units = company_dict[self.column_config.EMISSIONS_METRIC]['units']
        if company_dict[feature][scope.name]:
            projections = company_dict[feature][scope.name]['projections']
        else:
            scopes = scope.value.split('+')
            projection_scopes = {s: company_dict[feature][s]['projections'] for s in scopes if company_dict[feature][s]}
            if len(projection_scopes) > 1:
                projection_series = {}
                for s in scopes:
                    projection_series[s] = pd.Series(
                        {p['year']: p['value'] for p in company_dict[feature][s]['projections']},
                        name=company.company_id, dtype=f'pint[{emissions_units}/{production_units}]')
                series_adder = partial(pd.Series.add, fill_value=0)
                res = reduce(series_adder, projection_series.values())
                return res
            elif len(projection_scopes) == 0:
                return pd.Series(
                    {year: np.nan for year in range(self.historic_years[-1] + 1, self.projection_controls.TARGET_YEAR + 1)},
                    name=company.company_id, dtype=f'pint[{emissions_units}/{production_units}]'
                )
            else:
                # This clause is only accessed if the scope is S1S2 or S1S2S3 of which only one scope is provided.
                projections = company_dict[feature][scopes[0]]['projections']
                # projections = []
        return pd.Series(
            {p['year']: p['value'] for p in projections},
            name=company.company_id, dtype=f'pint[{emissions_units}/{production_units}]')

    def _calculate_target_projections(self, production_bm: BaseProviderProductionBenchmark):
        """
        We cannot calculate target projections until after we have loaded benchmark data.
        We do so when companies are associated with benchmarks, in the DataWarehouse construction
        
        :param production_bm: A Production Benchmark (multi-sector, single-scope, 2020-2050)
        """
        for c in self._companies:
            if c.projected_targets is not None:
                continue
            if c.target_data is None:
                logger.warning(f"No target data for {c.company_name}")
                c.projected_targets = ICompanyEIProjectionsScopes()
            else:
                base_year_production = next((p.value for p in c.historic_data.productions if
                                             p.year == self.temp_config.CONTROLS_CONFIG.base_year), None)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    company_sector_region_info = pd.DataFrame({
                        self.column_config.COMPANY_ID: [c.company_id],
                        self.column_config.BASE_YEAR_PRODUCTION: [base_year_production.to(c.production_metric.units)],
                        self.column_config.GHG_SCOPE12: [c.ghg_s1s2],
                        self.column_config.SECTOR: [c.sector],
                        self.column_config.REGION: [c.region],
                    }, index=[0])
                bm_production_data = (production_bm.get_company_projected_production(company_sector_region_info)
                                      # We transpose the data so that we get a pd.Series that will accept the pint units as a whole (not element-by-element)
                                      .iloc[0].T
                                      .astype(f'pint[{str(base_year_production.units)}]'))
                c.projected_targets = EITargetProjector().project_ei_targets(c, bm_production_data)
    
    # ??? Why prefer TRAJECTORY over TARGET?
    def _get_company_intensity_at_year(self, year: int, company_ids: List[str]) -> pd.Series:
        """
        Returns projected intensities for a given set of companies and year
        :param year: calendar year
        :param company_ids: List of company ids
        :return: pd.Series with intensities for given company ids
        """
        return self.get_company_projected_trajectories(company_ids)[year]

    def get_company_data(self, company_ids: List[str]) -> List[ICompanyData]:
        """
        Get all relevant data for a list of company ids. This method should return a list of ICompanyData
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        company_data = [company for company in self._companies if company.company_id in company_ids]

        if len(company_data) is not len(company_ids):
            missing_ids = [c_id for c_id in company_ids if c_id not in [c.company_id for c in company_data]]
            logger.warning(f"Companies not found in fundamental data and excluded from further computations: "
                           f"{missing_ids}")

        return company_data

    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        Gets the value of a variable for a list of companies ids
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        return self.get_company_fundamentals(company_ids)[variable_name]

    def get_company_intensity_and_production_at_base_year(self, company_ids: List[str]) -> pd.DataFrame:
        """
        overrides subclass method
        :param: company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.PRODUCTION_METRIC, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.BASE_EI,
        ColumnsConfig.SECTOR and ColumnsConfig.REGION
        """
        df_fundamentals = self.get_company_fundamentals(company_ids)
        base_year = self.temp_config.CONTROLS_CONFIG.base_year
        company_info = df_fundamentals.loc[
            company_ids, [self.column_config.SECTOR, self.column_config.REGION,
                          self.column_config.BASE_YEAR_PRODUCTION,
                          self.column_config.GHG_SCOPE12]]
        ei_at_base = self._get_company_intensity_at_year(base_year, company_ids).rename(self.column_config.BASE_EI)
        return company_info.merge(ei_at_base, left_index=True, right_index=True)

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company (company_id is a column)
        """
        return pd.DataFrame.from_records(
            [ICompanyData.parse_obj(c.dict()).dict() for c in self.get_company_data(company_ids)],
            exclude=['projected_targets', 'projected_intensities', 'historic_data']).set_index(
            self.column_config.COMPANY_ID)

    def get_company_projected_trajectories(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected intensity trajectories per company, indexed by company_id
        """
        trajectory_list = [self._convert_projections_to_series(c, self.column_config.PROJECTED_EI) for c in
                           self.get_company_data(company_ids)]
        if trajectory_list:
            with warnings.catch_warnings():
                # pd.DataFrame.__init__ (in pandas/core/frame.py) ignores the beautiful dtype information adorning the pd.Series list elements we are providing.  Sad!
                warnings.simplefilter("ignore")
                return pd.DataFrame(trajectory_list)
        return pd.DataFrame()

    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected intensity targets per company, indexed by company_id
        """
        target_list = [self._convert_projections_to_series(c, self.column_config.PROJECTED_TARGETS)
                       for c in self.get_company_data(company_ids)]
        if target_list:
            with warnings.catch_warnings():
                # pd.DataFrame.__init__ (in pandas/core/frame.py) ignores the beautiful dtype information adorning the pd.Series list elements we are providing.  Sad!
                warnings.simplefilter("ignore")
                return pd.DataFrame(target_list)
        return pd.DataFrame()


class EITrajectoryProjector(object):
    """
    This class projects emissions intensities on company level based on historic data on:
    - A company's emission history (in t CO2)
    - A company's production history (units depend on industry, e.g. TWh for electricity)
    """

    def __init__(self, projection_controls: ProjectionControls = ProjectionControls()):
        self.projection_controls = projection_controls

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
                             ColumnsConfig.SCOPE: scope, **ems})
        return data

    def _historic_ei_to_dicts(self, id: str, intensities_scopes: IHistoricEIScopes) \
            -> List[Dict[str, str]]:
        data = []
        for scope, intensities in intensities_scopes.dict().items():
            if intensities:
                intsties = {intsty['year']: intsty['value'] for intsty in intensities}
                data.append(
                    {ColumnsConfig.COMPANY_ID: id, ColumnsConfig.VARIABLE: VariablesConfig.EMISSIONS_INTENSITIES,
                     ColumnsConfig.SCOPE: scope, **intsties})
        return data

    def _compute_missing_historic_ei(self, companies, historic_data):
        scopes = EScope.get_scopes()
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
                if scope == 'S1S2':
                    try:  # Try to add S1 and S2 emissions intensities
                        historic_data.loc[ei_keys[scope]] = historic_data.loc[ei_keys['S1']] + \
                                                            historic_data.loc[ei_keys['S2']]
                        append_this_missing_data = False
                    except KeyError:  # Either S1 or S2 emissions intensities not readily available
                        try:  # Try to compute S1+S2 EIs from S1+S2 emissions and productions
                            historic_data.loc[ei_keys[scope]] = historic_data.loc[emissions_keys[scope]] / \
                                                                historic_data.loc[production_key]
                            append_this_missing_data = False
                        except KeyError:
                            this_missing_data.append(f"{company.company_id} - {scope}")
                elif scope == 'S1S2S3':  # Implement when S3 data is available
                    pass
                elif scope == 'S3':  # Remove when S3 data is available - will be handled by 'else'
                    pass
                else:  # S1 and S2 cannot be computed from other EIs, so use emissions and productions
                    try:
                        historic_data.loc[ei_keys[scope]] = historic_data.loc[emissions_keys[scope]] / \
                                                            historic_data.loc[production_key]
                        append_this_missing_data = False
                    except KeyError:
                        this_missing_data.append(f"{company.company_id} - {scope}")
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
            for scope in ICompanyEIProjectionsScopes.__fields__:
                if not company.historic_data.emissions_intensities or not company.historic_data.emissions_intensities.__getattribute__(
                        scope):
                    scope_projections[scope] = None
                    continue
                results = extrapolations.loc[(company.company_id, VariablesConfig.EMISSIONS_INTENSITIES, scope)]
                units = f"{results.values[0].u:~P}"
                scope_dfs[scope] = results.astype(f"pint[{units}]")
                projections = [IProjection(year=year, value=value) for year, value in results.items()
                               if year >= TemperatureScoreConfig.CONTROLS_CONFIG.base_year]
                scope_projections[scope] = ICompanyEIProjections(ei_metric={'units': units}, projections=projections)
            if scope_projections.get('S1') and scope_projections.get('S2') and not scope_projections.get('S1S2'):
                results = scope_dfs['S1'] + scope_dfs['S2']
                units = f"{results.values[0].u:~P}"
                projections = [IProjection(year=year, value=value) for year, value in results.items()
                               if year >= TemperatureScoreConfig.CONTROLS_CONFIG.base_year]
                scope_projections['S1S2'] = ICompanyEIProjections(ei_metric={'units': units}, projections=projections)
            company.projected_intensities = ICompanyEIProjectionsScopes(**scope_projections)

    def _standardize(self, intensities: pd.DataFrame) -> pd.DataFrame:
        # When columns are years and rows are all different intensity types, we cannot winsorize
        # Transpose the dataframe, winsorize the columns (which are all coherent because they belong to a single variable/company), then transpose again
        intensities = intensities.T
        for col in intensities.columns:
            s = intensities[col]
            ei_units = f"{s.loc[s.first_valid_index()].u:~P}"
            if s.notnull().any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    intensities[col] = s.map(lambda x: Q_(np.nan, ei_units)
                    if x.m is np.nan or x.m is pd.NA else x).astype(f"pint[{ei_units}]")

        winsorized_intensities: pd.DataFrame = self._winsorize(intensities)
        for col in winsorized_intensities.columns:
            winsorized_intensities[col] = winsorized_intensities[col].astype(intensities[col].dtype)
        standardized_intensities: pd.DataFrame = self._interpolate(winsorized_intensities)
        with warnings.catch_warnings():
            # Don't worry about warning that we are intentionally dropping units as we transpose
            warnings.simplefilter("ignore")
            return standardized_intensities.T

    def _winsorize(self, historic_intensities: pd.DataFrame) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # quantile doesn't handle pd.NA inside Quantity
            historic_intensities = historic_intensities.applymap(lambda x: np.nan if pd.isnull(x.m) else x)
            # See https://github.com/hgrecco/pint-pandas/issues/114
            winsorized: pd.DataFrame = historic_intensities.clip(
                # Must set numeric_only to false to process Quantities
                lower=historic_intensities.quantile(q=self.projection_controls.LOWER_PERCENTILE, axis='index',
                                                    numeric_only=False),
                upper=historic_intensities.quantile(q=self.projection_controls.UPPER_PERCENTILE, axis='index',
                                                    numeric_only=False),
                axis='columns'
            )
        return winsorized

    def _interpolate(self, historic_intensities: pd.DataFrame) -> pd.DataFrame:
        # Interpolate NaNs surrounded by values, and extrapolate NaNs with last known value
        interpolated = historic_intensities.copy()
        for col in interpolated.columns:
            if interpolated[col].isnull().all():
                continue
            qty = interpolated[col].values.quantity
            s = pd.Series(data=qty.m, index=interpolated.index)
            interpolated[col] = pd.Series(PA_(s.interpolate(method='linear', inplace=False, limit_direction='forward'),
                                              dtype=f"{qty.u:~P}"), index=interpolated.index)
        return interpolated

    def _get_trends(self, intensities: pd.DataFrame):
        # Compute year-on-year growth ratios of emissions intensities
        # Transpose so we can work with homogeneous units in columns.  This means rows are years.
        intensities = intensities.T
        for col in intensities.columns:
            # ratios are dimensionless, so get rid of units, which confuse rolling/apply.  Some columns are NaN-only
            intensities[col] = intensities[col].map(lambda x: x if isinstance(x, float) else x.m)
        # TODO: do we want to fillna(0) or dropna()?
        ratios: pd.DataFrame = intensities.rolling(window=2, axis='index', closed='right') \
            .apply(func=self._year_on_year_ratio, raw=True)  # .dropna(how='all',axis=0) # .fillna(0)

        trends: pd.DataFrame = self.projection_controls.TREND_CALC_METHOD(ratios, axis='index', skipna=True).clip(
            lower=self.projection_controls.LOWER_DELTA,
            upper=self.projection_controls.UPPER_DELTA,
        )
        return trends.T

    def _extrapolate(self, trends: pd.DataFrame, projection_years: range, historic_data: pd.DataFrame) -> pd.DataFrame:
        projected_intensities = historic_data.loc[historic_data.index.intersection(trends.index)].copy()
        # We need to do a mini-extrapolation if we don't have complete historic data
        for year in historic_data.columns.tolist()[:-1]:
            m = projected_intensities[year + 1].apply(lambda x: np.isnan(x.m))
            projected_intensities.loc[m, year + 1] = projected_intensities.loc[m, year] * (1 + trends.loc[m])

        # Now the big extrapolation
        for year in projection_years:
            projected_intensities[year + 1] = projected_intensities[year] * (1 + trends)
        return projected_intensities

    def _year_on_year_ratio(self, arr: np.ndarray) -> float:
        return (arr[1] / arr[0]) - 1.0


class EITargetProjector(object):
    """
    This class projects emissions intensities from a company's targets and historic data. Targets are specified per
    scope in terms of either emissions or emission intensity reduction. Interpolation between last known historic data
    and (a) target(s) is CAGR-based.

    Remember that pd.Series are always well-behaved with pint[] quantities.  pd.DataFrame columns are well-behaved,
    but data across columns is not always well-behaved.  We therefore make this function assume we are projecting targets
    for a specific company, in a specific sector.  If we want to project targets for multiple sectors, we have to call it multiple times.
    This function doesn't need to know what sector it's computing for...only tha there is only one such, for however many scopes.
    """

    def __init__(self):
        pass

    def _normalize_scope_targets(self, scope_targets):
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

    def project_ei_targets(self, company: ICompanyData, production_bm: pd.Series) -> ICompanyEIProjectionsScopes:
        """Input:
        @targets: a list of a company's targets
        @historic_data: a company's historic production, emissions, and emission intensities realizations per scope
        @production_bm: company's production projection computed from region-sector benchmark growth rates

        If the company has no target or the target can't be processed, then the output the emission database, unprocessed
        """
        targets, historic_data, projected_intensities = company.target_data, company.historic_data, company.projected_intensities
        ei_projection_scopes = {"S1": None, "S2": None, "S1S2": None, "S3": None, "S1S2S3": None}
        for scope in ei_projection_scopes:
            scope_targets = [target for target in targets if target.target_scope.name == scope]
            if not scope_targets:
                continue
            netzero_year = max([t.netzero_year for t in scope_targets if t.netzero_year] + [0])
            scope_targets_intensity = self._normalize_scope_targets(
                [target for target in scope_targets if target.target_type == "intensity"])
            scope_targets_absolute = self._normalize_scope_targets(
                [target for target in scope_targets if target.target_type == "absolute"])
            while scope_targets_intensity or scope_targets_absolute:
                if scope_targets_intensity and scope_targets_absolute:
                    target_i = scope_targets_intensity[0]
                    target_a = scope_targets_absolute[0]
                    if target_i.target_end_year == target_a.target_end_year:
                        if target_i.target_start_year == target_a.target_start_year:
                            warnings.warn(
                                f"intensity target overrides absolute target for target_start_year={target_i.target_start_year} and target_end_year={target_i.target_end_year}")
                            scope_targets_absolute.pop(0)
                            scope_targets = scope_targets_intensity
                        elif target_i.target_start_year > target_a.target_start_year:
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

                # Solve for intensity and absolute
                if target.target_type == "intensity":
                    # Simple case: the target is in intensity

                    # If target is not the first one for this scope, we continue from last year of the previous target
                    if ei_projection_scopes[scope] is not None:
                        last_year_data = ei_projection_scopes[scope].projections[-1]
                    else:
                        # Get the intensity data
                        intensity_data = historic_data.emissions_intensities.__getattribute__(scope)
                        last_year_data = next((i for i in reversed(intensity_data) if np.isfinite(i.value.magnitude)),
                                              None)

                    if last_year_data is None:  # No historic data, so no trajectory projections to use either
                        ei_projection_scopes[scope] = None
                        continue

                    if base_year > last_year_data.year:
                        trajectory_ei = projected_intensities.__getattribute__(scope).projections
                        last_year_data = next((ei for ei in trajectory_ei if ei.year == base_year), None)
                        warnings.warn(f"Emission intensity at base year for scope {scope} target for company "
                                      f"{company.company_name} is estimated with trajectory projection.")

                    # Removed condition base year > first_year. Do we care as long as base_year_qty is known?
                    last_year, value_last_year = last_year_data.year, last_year_data.value
                    target_year = target.target_end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_value = Q_(target.target_base_year_qty * (1 - target.target_reduction_pct),
                                      target.target_base_year_unit)
                    CAGR = self._compute_CAGR(value_last_year, target_value, (target_year - last_year))
                    ei_projections = [ICompanyEIProjection(year=year, value=value_last_year * (1 + CAGR) ** (y + 1))
                                      for y, year in enumerate(range(1 + last_year, 1 + target_year))]
                    if ei_projection_scopes[scope] is not None:
                        ei_projection_scopes[scope].projections.extend(ei_projections)
                    else:
                        ei_projection_scopes[scope] = ICompanyEIProjections(projections=ei_projections,
                                                                            ei_metric=IntensityMetric.parse_obj({'units': target.target_base_year_unit}))
                elif target.target_type == "absolute":
                    # Complicated case, the target must be switched from absolute value to intensity.
                    # We use the benchmark production data
                    # Compute Emission CAGR
                    emissions_data = historic_data.emissions.__getattribute__(scope)

                    # Get last year data with non-null value
                    if ei_projection_scopes[scope] is not None:
                        last_year_ei = ei_projection_scopes[scope].projections[-1]
                        last_year = last_year_ei.year
                        last_year_prod = production_bm.loc[last_year]
                        last_year_data = IEmissionRealization(year=last_year, value=last_year_ei.value*last_year_prod)
                    else:
                        last_year_data = next((e for e in reversed(emissions_data) if np.isfinite(e.value.magnitude)),
                                              None)

                    if last_year_data is None:  # No trajectory available either
                        ei_projection_scopes[scope] = None
                        continue

                    # Use trajectory info for data at base_year
                    if base_year > last_year_data.year:
                        trajectory_ei = projected_intensities.__getattribute__(scope).projections
                        last_year_ei = next((ei for ei in trajectory_ei if ei.year == base_year), None)
                        last_year = last_year_ei.year
                        last_year_prod = production_bm.loc[last_year]
                        last_year_data = IEmissionRealization(year=last_year, value=last_year_ei.value*last_year_prod)
                        warnings.warn(f"Emissions at base year for scope {scope} target for company "
                                      f"{company.company_name} are estimated with trajectory projection.")

                    last_year, value_last_year = last_year_data.year, last_year_data.value
                    target_year = target.target_end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_value = Q_(target.target_base_year_qty * (1 - target.target_reduction_pct),
                                      target.target_base_year_unit)
                    CAGR = self._compute_CAGR(value_last_year, target_value, (target_year - last_year))

                    emissions_projections = [value_last_year * (1 + CAGR) ** (y + 1)
                                             for y, year in enumerate(range(last_year + 1, target_year + 1))]
                    emissions_projections = pd.Series(emissions_projections,
                                                      index=range(last_year + 1, target_year + 1),
                                                      dtype=f'pint[{target.target_base_year_unit}]')
                    production_projections = production_bm.loc[last_year + 1: target_year]
                    ei_projections = emissions_projections / production_projections

                    ei_projections = [ICompanyEIProjection(year=year, value=ei_projections[year])
                                      for year in range(last_year + 1, target_year + 1)]
                    # TODO: this condition should not arise if prioritization logic above is correct
                    # From here out most useful to have target_value as EI
                    target_value = ei_projections[-1].value
                    if ei_projection_scopes[scope] is not None:
                        ei_projection_scopes[scope].projections.extend(ei_projections)
                    else:
                        ei_projection_scopes[scope] = ICompanyEIProjections(projections=ei_projections,
                                                                            ei_metric=IntensityMetric.parse_obj(
                                                                                {'units': f"{target_value.u:~P}"}))
                else:
                    # No target (type) specified
                    ei_projection_scopes[scope] = None
                    continue

                if scope_targets_intensity and scope_targets_intensity[0].netzero_year:
                    # Let a later target set the netzero year
                    continue
                if scope_targets_absolute and scope_targets_absolute[0].netzero_year:
                    # Let a later target set the netzero year
                    continue
                # TODO What if target is a 100% reduction.  Does it work whether or not netzero_year is set?
                if netzero_year > target_year:  # add in netzero target at the end
                    netzero_qty = Q_(0, target_value.u)
                    CAGR = self._compute_CAGR(target_value, netzero_qty, (netzero_year - target_year))
                    ei_projections = [ICompanyEIProjection(year=year, value=target_value * (1 + CAGR) ** (y + 1))
                                      for y, year in enumerate(range(1 + target_year, 1 + netzero_year))]
                    ei_projection_scopes[scope].projections.extend(ei_projections)
                    target_year = netzero_year
                    target_value = netzero_qty
                if target_year < 2050:
                    # Assume everything stays flat until 2050
                    ei_projection_scopes[scope].projections.extend(
                        [ICompanyEIProjection(year=year, value=target_value)
                         for y, year in enumerate(range(1 + target_year, 1 + 2050))]
                    )

        return ICompanyEIProjectionsScopes(**ei_projection_scopes)

    def _compute_CAGR(self, first, last, period):
        """Input:
        @first: the value of the first datapoint in the Calculation (most recent actual datapoint)
        @last: last value (value at future target year)
        @period: number of periods in the CAGR"""

        if period == 0:
            res = 0
        else:
            # TODO: Replace ugly fix => pint unit error in below expression
            # CAGR doesn't work well with 100% reduction, so set it to small
            if last == 0:
                last = first / 201.0
            elif last > first:
                # If we have a slack target, i.e., target goal is actually above current data, clamp so CAGR computes as zero
                last = first
            try:
                res = (last / first).to_base_units().magnitude ** (1 / period) - 1
            except ZeroDivisionError as e:
                if last > 0:
                    logger.warning("last > 0 and first==0 in CAGR...setting CAGR to 0-.5")
                    res = -0.5
                else:
                    # It's all zero from here on out...clamp down on any emissions that poke up
                    res = -1
        return res
