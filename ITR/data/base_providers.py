import pandas as pd
from typing import List, Type
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.data_providers import CompanyDataProvider, IntensityBenchmarkDataProvider, ProductionBenchmarkDataProvider
from ITR.interfaces import ICompanyData, EScope


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
        :param feature: PROJECTED_EI or PROJECTED_TARGETS
        :param scope: a scope
        :return: pd.Series
        """
        return pd.Series(
            {r['year']: r['value'] for r in company.dict()[feature][str(scope)]['projections']},
            name=company.company_id)

    def _get_company_intensity_at_year(self, year: int, company_ids: List[str]) -> pd.Series:
        """
        Returns projected intensities for a given set of companies and year
        :param year: calendar year
        :param company_ids: List of company ids
        :return: pd.Series with intensities for given company ids
        """
        return self.get_company_projected_intensities(company_ids)[year]

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
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and
        ColumnsConfig.REGION
        """
        df_fundamentals = self.get_company_fundamentals(company_ids)
        base_year = self.temp_config.CONTROLS_CONFIG.base_year
        company_info = df_fundamentals.loc[
            company_ids, [self.column_config.SECTOR, self.column_config.REGION,
                          self.column_config.GHG_SCOPE12]]
        ei_at_base = self._get_company_intensity_at_year(base_year, company_ids).rename(self.column_config.BASE_EI)
        return company_info.merge(ei_at_base, left_index=True, right_index=True)

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company
        """
        return pd.DataFrame.from_records(
            [ICompanyData.parse_obj(c).dict() for c in self.get_company_data(company_ids)],
            exclude=['projected_targets', 'projected_intensities']).set_index(self.column_config.COMPANY_ID)

    def get_company_projected_intensities(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected intensities per company
        """
        return pd.DataFrame(
            [self._convert_projections_to_series(c, self.column_config.PROJECTED_EI) for c in
             self.get_company_data(company_ids)])

    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected targets per company
        """
        return pd.DataFrame(
            [self._convert_projections_to_series(c, self.column_config.PROJECTED_TARGETS) for c in
             self.get_company_data(company_ids)])


class BaseProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(self, excel_path: str, benchmark_temperature: float,
                 benchmark_global_budget: float, is_AFOLU_included: bool,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__(benchmark_temperature, benchmark_global_budget, is_AFOLU_included)
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self.temp_config = tempscore_config
        self.column_config = column_config


class BaseProviderProductionBenchmark(ProductionBenchmarkDataProvider):
    def __init__(self, excel_path: str, column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__()
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self.temp_config = tempscore_config
        self.column_config = column_config
