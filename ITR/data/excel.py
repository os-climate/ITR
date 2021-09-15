from typing import Type, List
from pydantic import ValidationError
import logging

import pandas as pd
from ITR.data.data_provider import DataProvider
from ITR.configs import ColumnsConfig, TabsConfig, TemperatureScoreConfig, SectorsConfig
from ITR.interfaces import IDataProviderCompany


class ExcelProvider(DataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param config: A dictionary containing a "path" field that leads to the path of the CSV file
    """

    def __init__(self, company_path: str, sector_path: str, config: Type[ColumnsConfig] = ColumnsConfig):
        super().__init__()
        self.company_data = pd.read_excel(company_path, sheet_name=None, skiprows=0)
        self._check_company_data()
        self.sector_data = pd.read_excel(sector_path, sheet_name=None, skiprows=0)
        self.c = config

    def _check_company_data(self) -> None:
        """
        Checks if the company data excel contains the data in the right format

        :return: None
        """
        assert pd.Series([TabsConfig.FUNDAMENTAL, TabsConfig.PROJECTED_TARGET, TabsConfig.PROJECTED_EI]).isin(
            self.company_data.keys()).all(), "some tabs are missing in the company data excel"

    def _check_sector_data(self) -> None:
        """
        Checks if the sector data excel contains the data in the right format

        :return: None
        """
        assert pd.Series([TabsConfig.PROJECTED_PRODUCTION, TabsConfig.PROJECTED_EI]).isin(
            self.sector_data.keys()).all(), "some tabs are missing in the sector data excel"

    def _unit_of_measure_correction(self, company_ids: List[str], projected_emission: pd.DataFrame) -> pd.DataFrame:
        """

        :param company_ids: list of company ids
        :param projected_emission: series of projected emissions
        :return: series of projected emissions corrected for unit of measure
        """
        projected_emission.loc[self.get_value(company_ids, ColumnsConfig.SECTOR).isin(SectorsConfig.CORRECTION_SECTORS),
                               range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                     TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)] *= \
            TemperatureScoreConfig.CONTROLS_CONFIG.energy_unit_conversion_factor
        return projected_emission

    def _get_projection(self, company_ids: List[str], feature: str) -> pd.DataFrame:
        """
        get the projected emissions for list of companies
        :param company_ids: list of company ids
        :param feature: name of the projected feature
        :return: series of projected emissions
        """
        projected_emissions = self.company_data[feature]
        projected_emissions = projected_emissions.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        assert all(company_id in projected_emissions.index for company_id in company_ids), \
            f"company ids missing in {feature}"

        projected_emissions = projected_emissions.loc[company_ids, :]
        projected_emissions = self._unit_of_measure_correction(company_ids, projected_emissions)

        projected_emissions = projected_emissions.loc[:, range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                               TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)]
        projected_emissions_s1s2 = projected_emissions.groupby(level=0, sort=False).sum()  # add scope 1 and 2

        return projected_emissions_s1s2

    def _get_projected_production(self, ghg_scope12: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for list of companies
        :param ghg_scope12: Pandas Dataframe with ghg values indexed by company_id
        :return: Dataframe of projected productions for [base_year - base_year + 50]
        """
        company_ids = ghg_scope12.index
        benchmark_production_projections = self._get_benchmark_projections(company_ids, TabsConfig.PROJECTED_PRODUCTION)
        return benchmark_production_projections.add(1).cumprod(axis=1).mul(ghg_scope12.values, axis=0)

    def _get_benchmark_projections(self, company_ids: List[str], feature: str) -> pd.DataFrame:
        """
        get the sector emissions for a list of companies.
        If there is no data for the sector, then it will be replaced by the global value
        :param company_ids: list of company ids
        :param feature: name of the projected feature
        :return: series of projected emissions for the sector
        """
        benchmark_projection = self.sector_data[feature]
        sectors = self.get_value(company_ids, ColumnsConfig.SECTOR)
        regions = self.get_value(company_ids, ColumnsConfig.REGION)
        regions.loc[~regions.isin(benchmark_projection[ColumnsConfig.REGION])] = "Global"
        benchmark_projection = benchmark_projection.reset_index().set_index(
            [ColumnsConfig.SECTOR, ColumnsConfig.REGION])

        benchmark_projection = benchmark_projection.loc[list(zip(sectors, regions)),
                                                        range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                              TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = company_ids

        return benchmark_projection

    def _get_cumulative_emission(self, projected_emission_intensity: pd.DataFrame, projected_production: pd.DataFrame
                                 ) -> pd.Series:
        """
        get the weighted sum of the projected emission times the projected production
        :param projected_emission_intensity: series of projected emissions
        :param projected_production: series of projected production series
        :return: weighted sum of production and emission
        """
        return projected_emission_intensity.reset_index(drop=True).multiply(projected_production.reset_index(
            drop=True)).sum(axis=1)

    def get_company_data(self, company_ids: List[str]) -> List[IDataProviderCompany]:
        """
        Get all relevant data for a list of company ids (ISIN). This method should return a list of IDataProviderCompany
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        data_company = self.company_data[TabsConfig.FUNDAMENTAL]

        assert pd.Series(company_ids).isin(data_company.loc[:, ColumnsConfig.COMPANY_ID]).all(), \
            "some of the company ids are not included in the fundamental data"

        data_company = data_company.loc[data_company.loc[:, ColumnsConfig.COMPANY_ID].isin(company_ids), :]
        ghg_scope12 = data_company[[ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_SCOPE12]].set_index(
            ColumnsConfig.COMPANY_ID)
        projected_production = self._get_projected_production(ghg_scope12)

        data_company.loc[:, ColumnsConfig.CUMULATIVE_TRAJECTORY] = self._get_cumulative_emission(
            projected_emission_intensity=self._get_projection(company_ids, TabsConfig.PROJECTED_EI),
            projected_production=projected_production).to_numpy()

        data_company.loc[:, ColumnsConfig.CUMULATIVE_TARGET] = self._get_cumulative_emission(
            projected_emission_intensity=self._get_projection(company_ids, TabsConfig.PROJECTED_TARGET),
            projected_production=projected_production).to_numpy()

        data_company.loc[:, ColumnsConfig.CUMULATIVE_BUDGET] = self._get_cumulative_emission(
            projected_emission_intensity=self._get_benchmark_projections(company_ids, TabsConfig.PROJECTED_EI),
            projected_production=projected_production).to_numpy()

        companies = data_company.to_dict(orient="records")

        model_companies: List[IDataProviderCompany] = [IDataProviderCompany.parse_obj(company) for company in companies]

        return model_companies

    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        get the value of a variable of a list of companies
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        company_data = self.company_data[TabsConfig.FUNDAMENTAL]
        company_data = company_data.reset_index().set_index(ColumnsConfig.COMPANY_ID)
        return company_data.loc[company_ids, variable_name]
