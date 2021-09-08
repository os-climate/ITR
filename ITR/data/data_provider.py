from abc import ABC, abstractmethod
from typing import List

from ITR.interfaces import IDataProviderCompany


class DataProvider(ABC):
    """
    General data provider super class.
    """

    def __init__(self, **kwargs):
        """
        Create a new data provider instance.

        :param config: A dictionary containing the configuration parameters for this data provider.
        """
        pass



    @abstractmethod
    def get_company_data(self, company_ids: List[str]) -> List[IDataProviderCompany]:
        """
        Get all relevant data for a list of company ids (ISIN). This method should return a list of IDataProviderCompany
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        raise NotImplementedError



class CompanyNotFoundException(Exception):
    """
    This exception occurs when a company is not found.
    """
    pass
