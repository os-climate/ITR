from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from ITR.interfaces import ICompanyData


class CompanyDataProvider(ABC):
    """
    Company data provider super class.
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
        get the value of a variable of a list of companies
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_intensity_and_production_at_base_year(self, company_ids: List[str], base_year: int) -> pd.DataFrame:
        """
        overrides subclass method
        :param: company_ids: list of company ids
        :param: base year: int
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and
        ColumnsConfig.REGION
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_projected_intensities(self, company_ids: List[str]) -> pd.DataFrame:
        """
        get the value of a variable of a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected intensities for each company in company_ids
        """
        raise NotImplementedError

    def _get_company_intensity_at_year(self, year: int, company_ids: List[str]) -> pd.Series:
        return self.get_company_projected_intensities(company_ids)[year]

    @abstractmethod
    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        get the value of a variable of a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected targets for each company in company_ids
        """
        raise NotImplementedError

    @abstractmethod
    def _unit_of_measure_correction(self, company_ids: List[str], projected_emission: pd.DataFrame) -> pd.DataFrame:
        """
        corrects the projection emissions for the configured sectors with a temperature correction from the
        TempScoreConfig
        :param company_ids: list of company ids
        :param projected_emission: series of projected emissions
        :return: series of projected emissions corrected for unit of measure
        """


class CompanyNotFoundException(Exception):
    """
    This exception occurs when a company is not found.
    """
    pass


class ProductionBenchmarkDataProvider(ABC):
    """
    Production projecton data provider super class.
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
    """
    AFOLU_CORRECTION_FACTOR = 0.76

    def __init__(self, benchmark_temperature: float, benchmark_global_budget: float, AFOLU_included: bool = False,
                 **kwargs):
        """
        Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        self.AFOLU_CORRECTION_FACTOR = 0.76
        self._benchmark_temperature = benchmark_temperature
        self._AFOLU_included = AFOLU_included
        self._benchmark_global_budget = benchmark_global_budget

    @property
    def AFOLU_included(self) -> bool:
        """
        :return: if AFOLU is included in the benchmarks global budget
        """
        return self._AFOLU_included

    @AFOLU_included.setter
    def AFOLU_included(self, value):
        self._AFOLU_included = value

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
        return self._benchmark_global_budget if self.AFOLU_included else (
                self._benchmark_global_budget / self.AFOLU_CORRECTION_FACTOR)

    @benchmark_global_budget.setter
    def benchmark_global_budget(self, value):
        self._benchmark_global_budget = value

    @abstractmethod
    def _get_intensity_benchmarks(self, company_secor_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        raise NotImplementedError

    @abstractmethod
    def get_SDA_intensity_benchmarks(self, company_secor_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        raise NotImplementedError
