import pandas as pd
import numpy as np
from typing import List, Type
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, SectorsConfig
from ITR.data import CompanyDataProvider
from ITR.interfaces import ICompanyData


class JsonProviderCompany(CompanyDataProvider):
    """
    Data provider skeleton for JSON files. This class serves primarily for connecting to the ITR tool via API.

    :param portfolio: A list of ICompanyData objects that each contain fundamental company data
    :param projected_ei: A pandas DataFrame containing the projected emission intensities per company
    :param projected_target: A pandas DataFrame containing the projected target emission intensities per company
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
    """

    def __init__(self,
                 portfolio: List[ICompanyData],
                 projected_ei: pd.DataFrame,
                 projected_target: pd.DataFrame,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__()
        self.portfolio = portfolio
        self.company_data = pd.DataFrame.from_records([c.dict(exclude_none=True) for c in portfolio])\
                                        .set_index(column_config.COMPANY_ID)
        self.projected_ei = projected_ei.set_index(column_config.COMPANY_ID)
        self.projected_target = projected_target.set_index(column_config.COMPANY_ID)
        self.column_config = column_config
        self.temp_config = tempscore_config
        self.ENERGY_UNIT_CONVERSION_FACTOR = 3.6

    def get_company_data(self, company_ids: List[str]) -> List[ICompanyData]:
        return [company for company in self.portfolio if company.company_id in company_ids]

    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        return self.company_data.loc[company_ids, variable_name]

    def get_company_intensity_and_production_at_base_year(self, company_ids: List[str]) -> pd.DataFrame:
        """
        overrides subclass method
        :param: company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        """
        missing_ids = [company_id for company_id in company_ids if company_id not in self.company_data.index]
        assert not missing_ids, f"Company IDs not found in fundamental data: {missing_ids}"

        base_year = self.temp_config.CONTROLS_CONFIG.base_year
        company_info = self.company_data.loc[company_ids, [self.column_config.SECTOR, self.column_config.REGION,
                                                           self.column_config.GHG_SCOPE12]]
        ei_at_base = self._get_company_intensity_at_year(base_year, company_ids).rename(self.column_config.BASE_EI)
        return company_info.merge(ei_at_base, left_index=True, right_index=True)

    def _get_company_projections(self, company_ids: List[str], projections: pd.DataFrame) -> pd.DataFrame:
        """
        Get projections from given data for list of companies
        :param company_ids: List of company IDs
        :param projections: A pandas DataFrame containing the projection data
        :return: A pandas DataFrame of projections for the supplied company IDs
        """
        missing_ids = [company_id for company_id in company_ids if company_id not in projections.index]
        assert not missing_ids, f"No projection data found for companies with ID: {missing_ids}"

        projected_emissions = projections.loc[company_ids, :]
        projected_emissions = self._unit_of_measure_correction(company_ids, projected_emissions)

        projected_emissions = projected_emissions.loc[:, range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                               self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]

        # Due to bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero workaround below:
        projected_emissions = projected_emissions.fillna(np.inf)
        projected_emissions_s1s2 = projected_emissions.groupby(level=0, sort=False).sum()  # add scope 1 and 2
        projected_emissions_s1s2 = projected_emissions_s1s2.replace(np.inf, np.nan)

        return projected_emissions_s1s2

    def get_company_projected_intensities(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected intensities per company
        """
        return self._get_company_projections(company_ids, self.projected_ei)

    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected targets per company
        """
        return self._get_company_projections(company_ids, self.projected_target)

    def _unit_of_measure_correction(self, company_ids: List[str], projected_emission: pd.DataFrame) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :param projected_emission: A pandas DataFrame of projected emissions
        :return: A pandas DataFrame of projected emissions corrected for unit of measure
        """
        projected_emission.loc[
            self.get_value(company_ids, self.column_config.SECTOR).isin(SectorsConfig.CORRECTION_SECTORS),
            range(self.temp_config.CONTROLS_CONFIG.base_year,
                  self.temp_config.CONTROLS_CONFIG.target_end_year + 1)] *= \
            self.ENERGY_UNIT_CONVERSION_FACTOR
        return projected_emission
