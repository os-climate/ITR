from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Type

import pandas as pd

from ..configs import ColumnsConfig, ProjectionControls  # noqa F401
from ..data.osc_units import EmissionsQuantity, delta_degC_Quantity
from ..interfaces import (  # noqa F401
    EScope,
    ICompanyData,
    ICompanyEIProjection,
    ICompanyEIProjections,
    ICompanyEIProjectionsScopes,
    IHistoricData,
    IHistoricEIScopes,
    IHistoricEmissionsScopes,
    IProductionRealization,
)


class CompanyDataProvider(ABC):
    """Company data provider super class.
    Data container for company specific data. It expects both Fundamental (e.g. Company revenue, marktetcap etc) and
    emission and target data per company.

    Initialized CompanyDataProvider is required when setting up a data warehouse instance.
    """

    def __init__(self, **kwargs):
        """Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        pass

    @property
    @abstractmethod
    def column_config(self) -> Type[ColumnsConfig]:
        """Return the ColumnsConfig associated with this Data Provider"""
        raise NotImplementedError

    @property
    @abstractmethod
    def own_data(self) -> bool:
        """Return True if this object contains its own data; false if data housed elsewhere"""
        raise NotImplementedError

    @abstractmethod
    def get_projection_controls(self) -> ProjectionControls:
        """Return the ProjectionControls associated with this CompanyDataProvider."""
        raise NotImplementedError

    @abstractmethod
    def get_company_ids(self) -> List[str]:
        """Return the list of Company IDs of this CompanyDataProvider"""
        raise NotImplementedError

    @abstractmethod
    def get_company_data(
        self, company_ids: Optional[List[str]] = None
    ) -> List[ICompanyData]:
        """Get all relevant data for a list of company ids (ISIN), or all company data if `company_ids` is None.
        This method should return a list of ICompanyData instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """Gets the value of a variable for a list of companies ids
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_intensity_and_production_at_base_year(
        self, company_ids: List[str]
    ) -> pd.DataFrame:
        """Get the emission intensity and the production for a list of companies at the base year.
        :param: company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and
        ColumnsConfig.REGION
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_projected_trajectories(
        self, company_ids: List[str]
    ) -> pd.DataFrame:
        """Gets the emission intensities for a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected intensity trajectories for each company in company_ids
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """Gets the projected targets for a list of companies
        :param company_ids: list of company ids
        :return: dataframe of projected intensity targets for each company in company_ids
        """
        raise NotImplementedError

    @abstractmethod
    def _allocate_emissions(
        self,
        new_companies: List[ICompanyData],
        ei_benchmarks: IntensityBenchmarkDataProvider,
        projection_controls: ProjectionControls,
    ):
        """Use benchmark data from `ei_benchmarks` to allocate sector-level emissions from aggregated emissions.
        For example, a Utility may supply both Electricity and Gas to customers, reported separately.
        When we split the company into Electricity and Gas lines of business, we can allocate Scope emissions
        to the respective lines of business using benchmark averages to guide the allocation.
        """
        raise NotImplementedError

    @abstractmethod
    def _validate_projected_trajectories(
        self, companies: List[ICompanyData], ei_bm: IntensityBenchmarkDataProvider
    ):
        """Called when benchmark data is first known, or when projection control parameters or benchmark data changes.
        COMPANY_IDS are a list of companies with historic data that need to be projected.
        EI_BENCHMARKS are the benchmarks for all sectors, regions, and scopes
        In previous incarnations of this function, no benchmark data was needed for any reason.
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_target_projections(
        self,
        production_bm: ProductionBenchmarkDataProvider,
        ei_bm: IntensityBenchmarkDataProvider,
    ):
        """Use benchmark data to calculate target projections"""
        raise NotImplementedError


class ProductionBenchmarkDataProvider(ABC):
    """Production projecton data provider super class.

    This Data Container contains Production data on benchmark level. Data has a regions and sector indices.
    Initialized ProductionBenchmarkDataProvider is required when setting up a data warehouse instance.
    """

    def __init__(self, **kwargs):
        """Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        self._own_data = False

    @property
    def own_data(self) -> bool:
        """:return: True if this object contains its own data; false if data housed elsewhere"""
        return self._own_data

    @abstractmethod
    def benchmark_changed(
        self, production_benchmark: ProductionBenchmarkDataProvider
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_projected_production(
        self, scope: EScope = EScope.AnyScope
    ) -> pd.DataFrame:
        """Converts IProductionBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: a pint[dimensionless] pd.DataFrame
        """
        raise NotImplementedError

    @abstractmethod
    def get_company_projected_production(
        self, ghg_scope12: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the projected productions for all companies in ghg_scope12
        :param ghg_scope12: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: Dataframe of projected productions for [base_year - base_year + 50]
        """
        raise NotImplementedError


class IntensityBenchmarkDataProvider(ABC):
    """Production intensity data provider super class.
    This Data Container contains emission intensity data on benchmark level. Data has a regions and sector indices.
    Initialized IntensityBenchmarkDataProvider is required when setting up a data warehouse instance.
    """

    AFOLU_CORRECTION_FACTOR = (
        0.76  # AFOLU -> Acronym of agriculture, forestry and other land use
    )

    def __init__(
        self,
        benchmark_temperature: delta_degC_Quantity,
        benchmark_global_budget: EmissionsQuantity,
        is_AFOLU_included: bool,
        **kwargs,
    ):
        """Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        self.AFOLU_CORRECTION_FACTOR = 0.76
        self._benchmark_temperature = benchmark_temperature
        self._is_AFOLU_included = is_AFOLU_included
        self._benchmark_global_budget = benchmark_global_budget
        self._own_data = False

    @property
    def own_data(self) -> bool:
        """:return: True if this object contains its own data; false if data housed elsewhere"""
        return self._own_data

    @abstractmethod
    def get_scopes(self) -> List[EScope]:
        raise NotImplementedError

    @abstractmethod
    def benchmarks_changed(self, ei_benchmarks: IntensityBenchmarkDataProvider) -> bool:
        raise NotImplementedError

    @abstractmethod
    def prod_centric_changed(
        self, ei_benchmarks: IntensityBenchmarkDataProvider
    ) -> bool:
        raise NotImplementedError

    @property
    def is_AFOLU_included(self) -> bool:
        """:return: if AFOLU is included in the benchmarks global budget"""
        return self._is_AFOLU_included

    @is_AFOLU_included.setter
    def is_AFOLU_included(self, value):
        self._is_AFOLU_included = value

    @property
    def benchmark_temperature(self) -> delta_degC_Quantity:
        """:return: assumed temperature for the benchmark. for OECM 1.5C for example"""
        return self._benchmark_temperature

    @property
    def benchmark_global_budget(self) -> EmissionsQuantity:
        """:return: Benchmark provider assumed global budget. if AFOLU is not included global budget is divided by 0.76"""
        return (
            self._benchmark_global_budget
            if self.is_AFOLU_included
            else (self._benchmark_global_budget / self.AFOLU_CORRECTION_FACTOR)
        )

    @benchmark_global_budget.setter
    def benchmark_global_budget(self, value):
        self._benchmark_global_budget = value

    @abstractmethod
    def _get_intensity_benchmarks(
        self, company_sector_region_info: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        raise NotImplementedError

    @abstractmethod
    def get_SDA_intensity_benchmarks(
        self, company_sector_region_info: pd.DataFrame
    ) -> pd.DataFrame:
        """Returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        raise NotImplementedError

    @abstractmethod
    def is_production_centric(self) -> bool:
        """Returns True if benchmark is "production_centric" (as defined by OECM)"""
        raise NotImplementedError
