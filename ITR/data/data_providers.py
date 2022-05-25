from abc import ABC, abstractmethod
from typing import List, Dict, Union
import pandas as pd

import numpy as np

from ITR.configs import TabsConfig, ColumnsConfig, VariablesConfig, TemperatureScoreConfig
from ITR.interfaces import ICompanyData, EScope, IHistoricData, IProductionRealization, IHistoricEmissionsScopes, \
    IHistoricEIScopes, ICompanyEIProjection, ICompanyEIProjectionsScopes, ICompanyEIProjections

import pint
from pint import Quantity
from ITR.data.osc_units import ureg
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
    def get_company_projected_trajectories(self, company_ids: List[str]) -> pd.DataFrame:
        """
        Gets the emission intensities for a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected intensity trajectories for each company in company_ids
        """
        raise NotImplementedError


    @abstractmethod
    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        Gets the projected targets for a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected intensity targets for each company in company_ids
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

    def __init__(self, benchmark_temperature: Quantity['delta_degC'], benchmark_global_budget: Quantity['CO2'], is_AFOLU_included: bool,
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
    def benchmark_temperature(self) -> Quantity['delta_degC']:
        """
        :return: assumed temperature for the benchmark. for OECM 1.5C for example
        """
        return self._benchmark_temperature

    @property
    def benchmark_global_budget(self) -> Quantity['CO2']:
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
