from abc import ABC
from enum import Enum
import warnings # needed until apply behaves better with Pint quantities in arrays

from typing import Type

from pint import Quantity
from pint_pandas import PintArray

import pandas as pd
import pint
import pint_pandas

ureg = pint.get_application_registry()
Q_ = ureg.Quantity
PA_ = pint_pandas.PintArray

from .configs import PortfolioAggregationConfig, ColumnsConfig
from .interfaces import EScope


class PortfolioAggregationMethod(Enum):
    """
    The portfolio aggregation method determines how the temperature scores for the individual companies are aggregated
    into a single portfolio score.
    """
    WATS = 'WATS'
    TETS = 'TETS'
    MOTS = 'MOTS'
    EOTS = 'EOTS'
    ECOTS = 'ECOTS'
    AOTS = 'AOTS'
    ROTS = 'ROTS'

    @staticmethod
    def is_emissions_based(method: 'PortfolioAggregationMethod') -> bool:
        """
        Check whether a given method is emissions-based (i.e. it uses the emissions to calculate the aggregation).

        :param method: The method to check
        :return:
        """
        return method in [PortfolioAggregationMethod.MOTS, PortfolioAggregationMethod.EOTS,
                          PortfolioAggregationMethod.ECOTS, PortfolioAggregationMethod.AOTS,
                          PortfolioAggregationMethod.ROTS]

    @staticmethod
    def get_value_column(method: 'PortfolioAggregationMethod', column_config: Type[ColumnsConfig]) -> str:
        map_value_column = {
            PortfolioAggregationMethod.MOTS: column_config.COMPANY_MARKET_CAP,
            PortfolioAggregationMethod.EOTS: column_config.COMPANY_ENTERPRISE_VALUE,
            PortfolioAggregationMethod.ECOTS: column_config.COMPANY_EV_PLUS_CASH,
            PortfolioAggregationMethod.AOTS: column_config.COMPANY_TOTAL_ASSETS,
            PortfolioAggregationMethod.ROTS: column_config.COMPANY_REVENUE,
        }

        return map_value_column.get(method, column_config.COMPANY_MARKET_CAP)


class PortfolioAggregation(ABC):
    """
    This class is a base class that provides portfolio aggregation calculation.

    :param config: A class defining the constants that are used throughout this class. This parameter is only required
                    if you'd like to overwrite a constant. This can be done by extending the PortfolioAggregationConfig
                    class and overwriting one of the parameters.
    """

    def __init__(self, config: Type[PortfolioAggregationConfig] = PortfolioAggregationConfig):
        self.c = config

    def _check_column(self, data: pd.DataFrame, column: str):
        """
        Check if a certain column is filled for all companies. If not throw an error.

        :param data: The data to check
        :param column: The column to check
        :return:
        """
        missing_data = data[pd.isnull(data[column])][self.c.COLS.COMPANY_NAME].unique()
        if len(missing_data):
            raise ValueError("The value for {} is missing for the following companies: {}".format(
                column, ", ".join(missing_data)
            ))

    def _calculate_aggregate_score(self, data: pd.DataFrame, input_column: str,
                                   portfolio_aggregation_method: PortfolioAggregationMethod) -> pd.Series:
        """
        Aggregate the scores in a given column based on a certain portfolio aggregation method.

        :param data: The data to run the calculations on
        :param input_column: The input column (containing the scores)
        :param portfolio_aggregation_method: The method to use
        :return: The aggregates score as a pd.Series
        """
        if portfolio_aggregation_method == PortfolioAggregationMethod.WATS:
            total_investment_weight = data[self.c.COLS.INVESTMENT_VALUE].sum()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # See https://github.com/hgrecco/pint-pandas/issues/114
                    weights_series = pd.Series(data.apply(
                        lambda row: row[self.c.COLS.INVESTMENT_VALUE] * row[input_column] / total_investment_weight,
                        axis=1))
                    return weights_series
            except ZeroDivisionError:
                raise ValueError("The portfolio weight is not allowed to be zero")

        # Total emissions weighted temperature score (TETS)
        elif portfolio_aggregation_method == PortfolioAggregationMethod.TETS:
            use_S1S2 = data[self.c.COLS.SCOPE].isin([EScope.S1S2, EScope.S1S2S3])
            use_S3 = data[self.c.COLS.SCOPE].isin([EScope.S3, EScope.S1S2S3])
            if use_S3.any():
                self._check_column(data, self.c.COLS.GHG_SCOPE3)
            if use_S1S2.any():
                self._check_column(data, self.c.COLS.GHG_SCOPE12)
            # Calculate the total emissions of all companies
            emissions = data.loc[use_S1S2, self.c.COLS.GHG_SCOPE12].sum() + data.loc[use_S3, self.c.COLS.GHG_SCOPE3].sum()
            try:
                weights_series = pd.Series((data[self.c.COLS.GHG_SCOPE12].where(use_S1S2,0) + data[self.c.COLS.GHG_SCOPE3].where(use_S3, 0)) \
                            / emissions * data[input_column])
                return weights_series

            except ZeroDivisionError:
                raise ValueError("The total emissions should be higher than zero")

        elif PortfolioAggregationMethod.is_emissions_based(portfolio_aggregation_method):
            # These four methods only differ in the way the company is valued.
            value_column = PortfolioAggregationMethod.get_value_column(portfolio_aggregation_method, self.c.COLS)

            # Calculate the total owned emissions of all companies
            try:
                self._check_column(data, self.c.COLS.INVESTMENT_VALUE)
                self._check_column(data, value_column)
                use_S1S2 = data[self.c.COLS.SCOPE].isin([EScope.S1S2, EScope.S1S2S3])
                use_S3 = data[self.c.COLS.SCOPE].isin([EScope.S3, EScope.S1S2S3])
                if use_S1S2.any():
                    self._check_column(data, self.c.COLS.GHG_SCOPE12)
                if use_S3.any():
                    self._check_column(data, self.c.COLS.GHG_SCOPE3)
                data[self.c.COLS.OWNED_EMISSIONS] = (data[self.c.COLS.INVESTMENT_VALUE] / data[value_column]) * (
                        data[self.c.COLS.GHG_SCOPE12].where(use_S1S2, 0) + data[self.c.COLS.GHG_SCOPE3].where(use_S3, 0))
            except ZeroDivisionError:
                raise ValueError("To calculate the aggregation, the {} column may not be zero".format(value_column))
            owned_emissions = data[self.c.COLS.OWNED_EMISSIONS].sum()

            try:
                # Calculate the MOTS value per company
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = data.apply(
                        lambda row: (row[self.c.COLS.OWNED_EMISSIONS] / owned_emissions) * row[input_column],
                        axis=1)
                return result
            except ZeroDivisionError:
                raise ValueError("The total owned emissions can not be zero")
        else:
            raise ValueError("The specified portfolio aggregation method is invalid")
