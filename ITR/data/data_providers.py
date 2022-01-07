from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np

from ITR.configs import ProjectionConfig
from ITR.interfaces import ICompanyData


class CompanyDataProvider(ABC):
    """
    Company data provider super class.
    Data container for company specific data. It expects both Fundamental (e.g. Company revenue, marktetcap etc) and
    emission and target data per company.

    Initialized CompanyDataProvider is required when setting up a data warehouse instance.
    """

    def __init__(self, **kwargs):
        """
        Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        pass

    @abstractmethod
    def get_company_data(self, company_ids: List[str]) -> List[ICompanyData]:
        """
        Get all relevant data for a list of company ids (ISIN). This method should return a list of ICompanyData
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        Gets the value of a variable for a list of companies idss
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_intensity_and_production_at_base_year(self, company_ids: List[str]) -> pd.DataFrame:
        """
        Get the emission intensity and the production for a list of companies at the base year.
        :param: company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and
        ColumnsConfig.REGION
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_projected_intensities(self, company_ids: List[str]) -> pd.DataFrame:
        """
        Gets the emission intensities for a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected intensities for each company in company_ids
        """
        raise NotImplementedError


    @abstractmethod
    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        Gets the projected targets for a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected targets for each company in company_ids
        """
        raise NotImplementedError


class ProductionBenchmarkDataProvider(ABC):
    """
    Production projecton data provider super class.

    This Data Container contains Production data on benchmark level. Data has a regions and sector indices.
    Initialized ProductionBenchmarkDataProvider is required when setting up a data warehouse instance.
    """

    def __init__(self, **kwargs):
        """
        Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        pass

    @abstractmethod
    def get_company_projected_production(self, ghg_scope12: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for all companies in ghg_scope12
        :param ghg_scope12: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: Dataframe of projected productions for [base_year - base_year + 50]
        """
        raise NotImplementedError

    @abstractmethod
    def get_benchmark_projections(self, company_secor_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        get the sector emissions for a list of companies.
        If there is no data for the sector, then it will be replaced by the global value
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        raise NotImplementedError


class IntensityBenchmarkDataProvider(ABC):
    """
    Production intensity data provider super class.
    This Data Container contains emission intensity data on benchmark level. Data has a regions and sector indices.
    Initialized IntensityBenchmarkDataProvider is required when setting up a data warehouse instance.
    """
    AFOLU_CORRECTION_FACTOR = 0.76  # AFOLU -> Acronym of agriculture, forestry and other land use

    def __init__(self, benchmark_temperature: float, benchmark_global_budget: float, is_AFOLU_included: bool,
                 **kwargs):
        """
        Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        self.AFOLU_CORRECTION_FACTOR = 0.76
        self._benchmark_temperature = benchmark_temperature
        self._is_AFOLU_included = is_AFOLU_included
        self._benchmark_global_budget = benchmark_global_budget

    @property
    def is_AFOLU_included(self) -> bool:
        """
        :return: if AFOLU is included in the benchmarks global budget
        """
        return self._is_AFOLU_included

    @is_AFOLU_included.setter
    def is_AFOLU_included(self, value):
        self._is_AFOLU_included = value

    @property
    def benchmark_temperature(self) -> float:
        """
        :return: assumed temperature for the benchmark. for OECM 1.5C for example
        """
        return self._benchmark_temperature

    @property
    def benchmark_global_budget(self) -> float:
        """
        :return: Benchmark provider assumed global budget. if AFOLU is not included global budget is divided by 0.76
        """
        return self._benchmark_global_budget if self.is_AFOLU_included else (
                self._benchmark_global_budget / self.AFOLU_CORRECTION_FACTOR)

    @benchmark_global_budget.setter
    def benchmark_global_budget(self, value):
        self._benchmark_global_budget = value

    @abstractmethod
    def _get_intensity_benchmarks(self, company_sector_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        raise NotImplementedError

    @abstractmethod
    def get_SDA_intensity_benchmarks(self, company_sector_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        raise NotImplementedError


class EmissionIntensityProjector(ABC):
    """
    This class projects emission intensities on company level based on historic data on:
    - A company's emission history (in t CO2)
    - A company's production history (units depend on industry, e.g. TWh for electricity)
    """

    def __init__(self, historic_data: pd.DataFrame):
        self.historic_data = historic_data
        self.projection_data = None
        self.historic_years = [column for column in self.historic_data.columns if type(column) == int]
        self.projection_years = range(max(self.historic_years), ProjectionConfig.TARGET_YEAR)

    def project(self) -> pd.DataFrame:
        # TODO: Input should be a List[ICompanyData], Output should be a List[ICompanyData]
        self._validate_historic_data()
        # TODO: Check if emission intensities are supplied
        # TODO: If they are not: compute intensities by emissions / production
        # TODO: Keep only S1S2 and add comment to separate further in the future
        # TODO: Work back from end result: projected_ei_trajectories: Optional[ICompanyEIProjectionsScopes] = None

        historic_intensities: pd.DataFrame = self.historic_data[self.historic_years]
        standardized_intensities = self._standardize(historic_intensities)
        intensity_trends = self._get_trends(standardized_intensities)
        return self._extrapolate(intensity_trends)

    def _validate_historic_data(self):
        # TODO: Check that data contains at least 2 values that are not NaNs
        # TODO: Check that data is indeed historic
        # TODO: Throw error or continue with valid companies + log invalid company IDs
        pass

    def get_emission_intensities(self):
        # TODO: Separate S1 and S2
        # TODO: Separate Steel Electricity in results - units are different
        # TODO: Get historic emissions
        # TODO: Get historic production
        # TODO: Compute EIs for each scope S1, S2, S1+S2
        pass

    def _standardize(self, intensities: pd.DataFrame) -> pd.DataFrame:
        winsorized_intensities: pd.DataFrame = self._winsorize(intensities)
        standardized_intensities: pd.DataFrame = self._interpolate(winsorized_intensities)
        return standardized_intensities

    def _winsorize(self, historic_intensities: pd.DataFrame) -> pd.DataFrame:
        winsorized: pd.DataFrame = historic_intensities.clip(
            lower=historic_intensities.quantile(q=ProjectionConfig.LOWER_PERCENTILE, axis='columns', numeric_only=True),
            upper=historic_intensities.quantile(q=ProjectionConfig.UPPER_PERCENTILE, axis='columns', numeric_only=True),
            axis='index'
        )
        return winsorized

    def _interpolate(self, historic_intensities: pd.DataFrame) -> pd.DataFrame:
        # Interpolate NaNs surrounded by values, and extrapolate NaNs with last known value
        interpolated = historic_intensities.interpolate(method='linear', axis='columns', inplace=False,
                                                        limit_direction='forward')
        return interpolated

    def _get_trends(self, intensities: pd.DataFrame):
        # Compute year-on-year growth ratios of emission intensities
        ratios: pd.DataFrame = intensities.rolling(window=2, axis='columns', closed='right') \
            .apply(func=self._year_on_year_ratio, raw=True)

        trends: pd.DataFrame = ratios.median(axis='columns', skipna=True).clip(
            lower=ProjectionConfig.LOWER_DELTA,
            upper=ProjectionConfig.UPPER_DELTA,
        )
        return trends

    def _extrapolate(self, trends: pd.DataFrame) -> pd.DataFrame:
        projected_intensities = self.historic_data.copy()
        for year in self.projection_years:
            projected_intensities[year + 1] = projected_intensities[year] * (1 + trends)
        return projected_intensities

    def _year_on_year_ratio(self, arr: np.ndarray) -> float:
        return (arr[1] / arr[0]) - 1.0