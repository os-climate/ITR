import pandas as pd

import pint
import pint_pandas
from ITR.data.osc_units import ureg, PA_

from typing import List, Type
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.interfaces import ICompanyData, EScope, \
    IEmissionIntensityBenchmarkScopes, IYOYBenchmarkScopes, \
    IEIBenchmark, IYOYBenchmark


# TODO handling of scopes in benchmarks

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
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__()
        self._companies = companies
        self.column_config = column_config
        self.temp_config = tempscore_config

    def _convert_projections_to_series(self, company: ICompanyData, feature: str,
                                       scope: EScope = EScope.S1S2) -> pd.Series:
        """
        extracts the company projected intensities or targets for a given scope
        :param feature: PROJECTED_TRAJECTORIES or PROJECTED_TARGETS (both are intensities)
        :param scope: a scope
        :return: pd.Series
        """
        feature_to_units = { self.column_config.PROJECTED_TRAJECTORIES:'pint[t CO2/MWh]', self.column_config.PROJECTED_TARGETS:'pint[t CO2/MWh]' }
        return pd.Series(
            {r['year']: r['value'] for r in company.dict()[feature][str(scope)]['projections']},
            name=company.company_id, dtype=feature_to_units[feature])

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
            missing_ids = [company.company_id for company in self._companies if company.company_id not in company_ids]
            assert not missing_ids, f"Company IDs not found in fundamental data: {missing_ids}"

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
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI,
        ColumnsConfig.SECTOR and ColumnsConfig.REGION
        """
        df_fundamentals = self.get_company_fundamentals(company_ids)
        # print(f"df_fundamentals = {df_fundamentals}")
        base_year = self.temp_config.CONTROLS_CONFIG.base_year
        company_info = df_fundamentals.loc[
            company_ids, [self.column_config.SECTOR, self.column_config.REGION,
                          self.column_config.GHG_SCOPE12]]
        ei_at_base = self._get_company_intensity_at_year(base_year, company_ids).rename(self.column_config.BASE_EI)
        return company_info.merge(ei_at_base, left_index=True, right_index=True)

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company (company_id is a column)
        """
        return pd.DataFrame.from_records(
            [ICompanyData.parse_obj(c).dict() for c in self.get_company_data(company_ids)],
            exclude=['projected_ei_targets', 'projected_ei_trajectories']).set_index(self.column_config.COMPANY_ID)

    def get_company_projected_trajectories(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected intensity trajectories per company, indexed by company_id
        """
        return pd.DataFrame(
            [self._convert_projections_to_series(c, self.column_config.PROJECTED_TRAJECTORIES) for c in
             self.get_company_data(company_ids)], dtype='pint[t CO2/MWh]')

    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected intensity targets per company, indexed by company_id
        """
        return pd.DataFrame(
            [self._convert_projections_to_series(c, self.column_config.PROJECTED_TARGETS) for c in
             self.get_company_data(company_ids)], dtype='pint[t CO2/MWh]')

# This is actual output production (whatever the output production units may be).
# Not to be confused with the term "projected production" as it relates to energy intensity.

class BaseProviderProductionBenchmark(ProductionBenchmarkDataProvider):

    def __init__(self, production_benchmarks: IYOYBenchmarkScopes,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        """
        Base provider that relies on pydantic interfaces. Default for FastAPI usage
        :param production_benchmarks: List of IYOYBenchmarkScopes
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
        """
        super().__init__()
        self.temp_config = tempscore_config
        self.column_config = column_config
        self._productions_benchmarks = production_benchmarks

    # Note that bencharmk production series are dimensionless.
    def _convert_benchmark_to_series(self, benchmark: IYOYBenchmark) -> pd.Series:
        """
        extracts the company projected intensity or production targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        return pd.Series({r.year: r.value for r in benchmark.projections}, name=(benchmark.region, benchmark.sector))

    # YOY production benchmarks are dimensionless.  S1S2 has nothing to do with any company data.
    # It's a label in the top-level of benchmark data.  Currently S1S2 is the only label with any data.
    def _get_projected_production(self, scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Converts IYOYBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: pd.DataFrame
        """
        result = []
        for bm in self._productions_benchmarks.dict()[str(scope)]['benchmarks']:
            result.append(self._convert_benchmark_to_series(IYOYBenchmark.parse_obj(bm)))
        df_bm = pd.DataFrame(result)
        df_bm.index.names = [self.column_config.REGION, self.column_config.SECTOR]

        return df_bm

    # This data is in production units, not energy intensity units.  S1S2 has nothing to do with any company data.
    # It's a label in the top-level of benchmark data.  Currently S1S2 is the only label with any data.
    # And it appears that GHG__SCOPE12 is actually a production number, not an emissions number.
    def get_company_projected_production(self, ghg_scope12: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for list of companies in ghg_scope12
        :param ghg_scope12: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: DataFrame of projected productions for [base_year - base_year + 50]
        """
        benchmark_production_projections = self.get_benchmark_projections(ghg_scope12)
        company_production = ghg_scope12[self.column_config.GHG_SCOPE12]
        return benchmark_production_projections.add(1).cumprod(axis=1).mul(
                    company_production, axis=0).astype('pint[MWh]')

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

        benchmark_projection = benchmark_projection.loc[list(zip(benchmark_regions, sectors)),
                                                        range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                              self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index

        return benchmark_projection


class BaseProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(self, EI_benchmarks: IEmissionIntensityBenchmarkScopes,
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
        ColumnsConfig.COMPANY_ID, ColumnsConfig.BASE_EI ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        intensity_benchmarks = self._get_intensity_benchmarks(company_info_at_base_year)
        decarbonization_paths = self._get_decarbonizations_paths(intensity_benchmarks)
        last_ei = intensity_benchmarks[self.temp_config.CONTROLS_CONFIG.target_end_year]
        ei_base = company_info_at_base_year[self.column_config.BASE_EI]

        df = decarbonization_paths.mul((ei_base - last_ei), axis=0)
        df = df.add(last_ei, axis=0).astype('pint[t CO2/MWh]')
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
        # This throws a warning when processing a NaN
        return intensity_benchmark_row.apply(lambda x: (x.m - last_ei.m) / (first_ei.m - last_ei.m))

    def _convert_benchmark_to_series(self, benchmark: IEIBenchmark) -> pd.Series:
        """
        extracts the company projected intensities or targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        return pd.Series({r.year: r.value for r in benchmark.projections}, name=(benchmark.region, benchmark.sector), dtype='pint[t CO2/MWh]')

    def _get_projected_intensities(self, scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Converts IEIBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: pd.DataFrame
        """
        result = []
        for bm in self._EI_benchmarks.dict()[str(scope)]['benchmarks']:
            result.append(self._convert_benchmark_to_series(IEIBenchmark.parse_obj(bm)))
        df_bm = pd.DataFrame(result)
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
        sectors = company_sector_region_info[self.column_config.SECTOR]
        regions = company_sector_region_info[self.column_config.REGION]
        benchmark_regions = regions.copy()
        mask = benchmark_regions.isin(benchmark_projection.reset_index()[self.column_config.REGION])
        benchmark_regions.loc[~mask] = "Global"

        benchmark_projection = benchmark_projection.loc[list(zip(benchmark_regions, sectors)),
                                                        range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                              self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index
        return benchmark_projection
