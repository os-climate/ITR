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
    def get_projected_intensities(self, company_ids: List[str]) -> pd.DataFrame:
        """
        get the value of a variable of a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected intensities for each company in company_ids
        """
        raise NotImplementedError

    @abstractmethod
    def get_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        get the value of a variable of a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected targets for each company in company_ids
        """
        raise NotImplementedError

    @abstractmethod
    def _unit_of_measure_correction(self, company_ids: List[str], projected_emission: pd.DataFrame) -> pd.DataFrame:
        """
        corrects the projection emissions for the configured sectors with a temperature correction from the TempScoreConfig
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
    def get_projected_production_per_company(self, ghg_scope12: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for list of companies
        :param ghg_scope12: Pandas Dataframe with ghg values, sector and region indexed by company_id
        :return: Dataframe of projected productions for [base_year - base_year + 50]
        """
        raise NotImplementedError

    @abstractmethod
    def get_benchmark_projections(self, ghg_scope12: pd.DataFrame) -> pd.DataFrame:
        """
        get the sector emissions for a list of companies.
        If there is no data for the sector, then it will be replaced by the global value
        :param company_ids: list of company ids
        :param feature: name of the projected feature
        :return: series of projected emissions for the sector
        """
        raise NotImplementedError


class IntensityBenchmarkDataProvider(ABC):
    """
    Production intensity data provider super class.
    """

    def __init__(self, **kwargs):
        """
        Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        pass

    @abstractmethod
    def get_intensity_benchmarks(self, company_secor_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        Get all relevant data for a list of company ids (ISIN). This method should return a list of ICompanyData
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        raise NotImplementedError
