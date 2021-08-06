from typing import Type, List
from pydantic import ValidationError
import logging

import pandas as pd
import numpy as np
from ITR.data.data_provider import DataProvider
from ITR.configs import ColumnsConfig, TabsConfig, ControlsConfig, SectorsConfig
from ITR.interfaces import IDataProviderCompany, IDataProviderTarget


class ExcelProvider(DataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param config: A dictionary containing a "path" field that leads to the path of the CSV file
    """

    def __init__(self, path: str, config: Type[ColumnsConfig] = ColumnsConfig):
        super().__init__()
        self.data = pd.read_excel(path, sheet_name=None, skiprows=0)
        self.c = config

    def get_targets(self, company_ids: List[str]) -> List[IDataProviderTarget]:
        """
        Get all relevant targets for a list of company ids (ISIN). This method should return a list of
        IDataProviderTarget instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the targets
        """
        model_targets = self._target_df_to_model(self.data['target_data'])
        model_targets = [target for target in model_targets if target.company_id in company_ids]
        return model_targets

    def _target_df_to_model(self, df_targets):
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

    def get_projected_ei(self, company_ids: List[str]) -> pd.DataFrame:
        """
        """
        projected_ei = self.data[TabsConfig.PROJECTED_EI]
        projected_ei = projected_ei.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        assert all(company_id in projected_ei.index for company_id in company_ids), \
            f"company ids missing in {TabsConfig.PROJECTED_EI}"

        projected_ei = projected_ei.loc[company_ids, :]
        projected_ei.loc[projected_ei.loc[:, ColumnsConfig.SECTOR] == SectorsConfig.ELECTRICITY,
                         range(ControlsConfig.BASE_YEAR, ControlsConfig.TARGET_END_YEAR + 1)] *= 3.6

        projected_ei = projected_ei.loc[:, range(ControlsConfig.BASE_YEAR, ControlsConfig.TARGET_END_YEAR + 1)]
        projected_ei = projected_ei.groupby(level=0, sort=False).sum()
        return projected_ei

    def get_projected_production(self, company_ids: List[str]) -> pd.DataFrame:
        """
        """
        projected_production = self.data[TabsConfig.PROJECTED_PRODUCTION]
        projected_production = projected_production.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        assert all(company_id in projected_production.index for company_id in company_ids), \
            f"company ids missing in {TabsConfig.PROJECTED_PRODUCTION}"

        projected_production = projected_production.loc[company_ids,
                                                        range(ControlsConfig.BASE_YEAR, ControlsConfig.TARGET_END_YEAR + 1)]
        return projected_production

    def get_cumulative_targets(self, company_ids: List[str]):
        """
        TODO
        """
        return (self.get_projected_ei(company_ids) *
                self.get_projected_production(company_ids)).sum(axis=1).to_numpy()

    def get_company_data(self, company_ids: List[str]) -> List[IDataProviderCompany]:
        """
        Get all relevant data for a list of company ids (ISIN). This method should return a list of IDataProviderCompany
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        data_company = self.data[TabsConfig.FUNDAMENTAL]

        data_company = data_company.loc[data_company.loc[:, ColumnsConfig.COMPANY_ID].isin(company_ids), :]

        data_company.loc[:, ColumnsConfig.CUMULATIVE_TRAJECTORY] = self.get_cumulative_targets(company_ids)

        companies = data_company.to_dict(orient="records")

        model_companies: List[IDataProviderCompany] = [IDataProviderCompany.parse_obj(company) for company in companies]

        return model_companies

    def get_sbti_targets(self, companies: list) -> list:
        """
        For each of the companies, get the status of their target (Target set, Committed or No target) as it's known to
        the SBTi.

        :param companies: A list of companies. Each company should be a dict with a "company_name" and "company_id"
                            field.
        :return: The original list, enriched with a field called "sbti_target_status"
        """
        raise NotImplementedError
