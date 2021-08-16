from typing import Type, List
from pydantic import ValidationError
import logging

import pandas as pd
from ITR.data.data_provider import DataProvider
from ITR.configs import ColumnsConfig, TabsConfig, ControlsConfig, SectorsConfig
from ITR.interfaces import IDataProviderCompany, IDataProviderTarget


class ExcelProvider(DataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param config: A dictionary containing a "path" field that leads to the path of the CSV file
    """

    def __init__(self, company_path: str, sector_path: str, config: Type[ColumnsConfig] = ColumnsConfig):
        super().__init__()
        self.company_data = pd.read_excel(company_path, sheet_name=None, skiprows=0)
        self.check_company_data()
        self.sector_data = pd.read_excel(sector_path, sheet_name=None, skiprows=0)
        self.c = config

    def check_company_data(self) -> None:
        """
        Checks if the company data excel contains the data in the right format

        :return: None
        """
        assert pd.Series([TabsConfig.FUNDAMENTAL, TabsConfig.TARGET, TabsConfig.PROJECTED_TARGET,
                          TabsConfig.PROJECTED_PRODUCTION, TabsConfig.PROJECTED_EI]).isin(
            self.company_data.keys()).all(), "some tabs are missing in the company data excel"

    def check_sector_data(self) -> None:
        """
        Checks if the sector data excel contains the data in the right format

        :return: None
        """
        assert pd.Series([TabsConfig.PROJECTED_PRODUCTION, TabsConfig.PROJECTED_EI]).isin(
            self.sector_data.keys()).all(), "some tabs are missing in the sector data excel"

    def get_targets(self, company_ids: List[str]) -> List[IDataProviderTarget]:
        """
        Get all relevant targets for a list of company ids (ISIN). This method should return a list of
        IDataProviderTarget instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the targets
        """
        model_targets = self._target_df_to_model(self.company_data['target_data'])
        model_targets = [target for target in model_targets if target.company_id in company_ids]
        return model_targets

    def _target_df_to_model(self, df_targets: pd.DataFrame) -> List[IDataProviderTarget]:
        """
        transforms target Dataframe into list of IDataProviderTarget instances

        :param df_targets: pandas Dataframe with targets
        :return: A list containing the targets
        """
        logger = logging.getLogger(__name__)
        targets = df_targets.to_dict(orient="records")
        model_targets: List[IDataProviderTarget] = []
        for target in targets:
            try:
                model_targets.append(IDataProviderTarget.parse_obj(target))
            except ValidationError as e:
                logger.warning(
                    "(one of) the target(s) of company %s is invalid and will be skipped" % target[self.c.COMPANY_NAME])
                pass
        return model_targets

    def _unit_of_measure_correction(self, company_ids: List[str], projected_values: pd.Series) -> pd.Series:
        """

        :param company_ids: list of company ids
        :param projected_values: series of projected values
        :return: series of projected values corrected for unit of measure
        """
        projected_values.loc[self.get_value(company_ids, ColumnsConfig.SECTOR) == SectorsConfig.ELECTRICITY,
                             range(ControlsConfig.BASE_YEAR, ControlsConfig.TARGET_END_YEAR + 1)] *= 3.6
        return projected_values

    def get_projected_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        get the projected value of a variable for list of companies
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of projected values
        """
        projected_values = self.company_data[variable_name]
        projected_values = projected_values.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        assert all(company_id in projected_values.index for company_id in company_ids), \
            f"company ids missing in {TabsConfig.PROJECTED_EI}"

        projected_values = projected_values.loc[company_ids, :]

        if variable_name == TabsConfig.PROJECTED_TARGET or variable_name == TabsConfig.PROJECTED_EI:
            projected_values = self._unit_of_measure_correction(company_ids, projected_values)

        projected_values = projected_values.loc[:, range(ControlsConfig.BASE_YEAR, ControlsConfig.TARGET_END_YEAR + 1)]

        if variable_name == TabsConfig.PROJECTED_TARGET or variable_name == TabsConfig.PROJECTED_EI:
            projected_values = projected_values.groupby(level=0, sort=False).sum()

        return projected_values

    def get_benchmark_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        get the benchmark value for a list of companies. The benchmark corresponds to the projected value of the sector.
        If there is no data for the sector, then it will be replaced by the global value
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of projected values for the benchmark
        """
        projected_benchmark = self.sector_data[variable_name]
        sectors = self.get_value(company_ids, ColumnsConfig.SECTOR)
        regions = self.get_value(company_ids, ColumnsConfig.REGION)
        regions.loc[~regions.isin(projected_benchmark[ColumnsConfig.REGION])] = "Global"
        projected_benchmark = projected_benchmark.reset_index().set_index([ColumnsConfig.SECTOR, ColumnsConfig.REGION])

        return projected_benchmark.loc[list(zip(sectors, regions)),
                                       range(ControlsConfig.BASE_YEAR, ControlsConfig.TARGET_END_YEAR + 1)].to_numpy()

    def get_cumulative_value(self, projected_emission: pd.Series, projected_production: pd.Series) -> pd.Series:
        """
        get the weighted sum of the projected emission times the projected production
        :param projected_emission: series of projected emission values
        :param projected_production: series of projected production series
        :return: weighted sum of production and emission
        """
        return (projected_emission * projected_production).sum(axis=1)

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

        data_company.loc[:, ColumnsConfig.CUMULATIVE_TRAJECTORY] = self.get_cumulative_value(
            self.get_projected_value(company_ids, TabsConfig.PROJECTED_EI),
            self.get_projected_value(company_ids, TabsConfig.PROJECTED_PRODUCTION)).to_numpy()

        data_company.loc[:, ColumnsConfig.CUMULATIVE_TARGET] = self.get_cumulative_value(
            self.get_projected_value(company_ids, TabsConfig.PROJECTED_TARGET),
            self.get_projected_value(company_ids, TabsConfig.PROJECTED_PRODUCTION)).to_numpy()

        data_company.loc[:, ColumnsConfig.CUMULATIVE_BUDGET] = self.get_cumulative_value(
            self.get_benchmark_value(company_ids, variable_name=TabsConfig.PROJECTED_EI),
            self.get_projected_value(company_ids, TabsConfig.PROJECTED_PRODUCTION)).to_numpy()

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

    def get_sbti_targets(self, companies: list) -> list:
        """
        For each of the companies, get the status of their target (Target set, Committed or No target) as it's known to
        the SBTi.

        :param companies: A list of companies. Each company should be a dict with a "company_name" and "company_id"
                            field.
        :return: The original list, enriched with a field called "sbti_target_status"
        """
        raise NotImplementedError
