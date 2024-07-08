import logging
import warnings  # needed until apply behaves better with Pint quantities in arrays
from abc import ABC
from enum import Enum
from typing import Type

import numpy as np
import pandas as pd

import ITR

from .configs import ColumnsConfig, LoggingConfig, PortfolioAggregationConfig
from .data import PintType
from .data.osc_units import PA_, asPintSeries
from .interfaces import EScope

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class PortfolioAggregationMethod(Enum):
    """The portfolio aggregation method determines how the temperature scores for the individual companies are aggregated
    into a single portfolio score.
    """

    WATS = "WATS"
    TETS = "TETS"
    MOTS = "MOTS"
    EOTS = "EOTS"
    ECOTS = "ECOTS"
    AOTS = "AOTS"
    ROTS = "ROTS"

    @staticmethod
    def is_emissions_based(method: "PortfolioAggregationMethod") -> bool:
        """Check whether a given method is emissions-based (i.e. it uses the emissions to calculate the aggregation).

        :param method: The method to check
        :return:
        """
        return method in [
            PortfolioAggregationMethod.MOTS,
            PortfolioAggregationMethod.EOTS,
            PortfolioAggregationMethod.ECOTS,
            PortfolioAggregationMethod.AOTS,
            PortfolioAggregationMethod.ROTS,
        ]

    @staticmethod
    def get_value_column(
        method: "PortfolioAggregationMethod", column_config: Type[ColumnsConfig]
    ) -> str:
        map_value_column = {
            PortfolioAggregationMethod.MOTS: column_config.COMPANY_MARKET_CAP,
            PortfolioAggregationMethod.EOTS: column_config.COMPANY_ENTERPRISE_VALUE,
            PortfolioAggregationMethod.ECOTS: column_config.COMPANY_EV_PLUS_CASH,
            PortfolioAggregationMethod.AOTS: column_config.COMPANY_TOTAL_ASSETS,
            PortfolioAggregationMethod.ROTS: column_config.COMPANY_REVENUE,
            # The test case tells us these should be correct
            PortfolioAggregationMethod.WATS: column_config.COMPANY_MARKET_CAP,
            PortfolioAggregationMethod.TETS: column_config.COMPANY_MARKET_CAP,
        }

        try:
            return map_value_column[method]
        except KeyError:
            logger.warning(
                f"method '{method}' not found (type({method}) = {type(method)}; defaulting to COMPANY_MARKET_CAP"
            )
        return column_config.COMPANY_MARKET_CAP


class PortfolioAggregation(ABC):
    """This class is a base class that provides portfolio aggregation calculation.

    :param config: A class defining the constants that are used throughout this class. This parameter is only required
                    if you'd like to overwrite a constant. This can be done by extending the PortfolioAggregationConfig
                    class and overwriting one of the parameters.
    """

    def __init__(
        self, config: Type[PortfolioAggregationConfig] = PortfolioAggregationConfig
    ):
        self.c = config

    def _check_column(self, data: pd.DataFrame, column: str):
        """Check if a certain column is filled for all companies. If not log an error.
        The aggregation treats missing values as zeroes.

        :param data: The data to check
        :param column: The column to check
        :return:
        """
        missing_data = data[ITR.isna(data[column])][self.c.COLS.COMPANY_NAME].unique()
        if len(missing_data):
            logger.error(
                f"The value for {column} is missing for the following companies: {', '.join(missing_data)}"
            )

    def _calculate_aggregate_score(
        self,
        data: pd.DataFrame,
        input_column: str,
        portfolio_aggregation_method: PortfolioAggregationMethod,
    ) -> pd.Series:
        """Aggregate the scores in a given column based on a certain portfolio aggregation method.

        :param data: The data to run the calculations on
        :param input_column: The input column (containing the scores)
        :param portfolio_aggregation_method: The method to use
        :return: The aggregates score as a pd.Series
        """
        # Used to test against data[input_column].dtype.kind in ['f', 'i']
        assert isinstance(data[input_column].dtype, PintType)
        if portfolio_aggregation_method == PortfolioAggregationMethod.WATS:
            # Used to test against data[self.c.COLS.INVESTMENT_VALUE].dtype.kind in ['f', 'i']
            assert isinstance(data[self.c.COLS.INVESTMENT_VALUE].dtype, PintType)
            total_investment_weight = data[self.c.COLS.INVESTMENT_VALUE].sum()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # See https://github.com/hgrecco/pint-pandas/issues/114
                    weights_series = (
                        data[input_column]
                        * data[self.c.COLS.INVESTMENT_VALUE]
                        / total_investment_weight
                    )
                    return weights_series
            except ZeroDivisionError:
                raise ValueError("The portfolio weight is not allowed to be zero")

        # Total emissions weighted temperature score (TETS)
        elif portfolio_aggregation_method == PortfolioAggregationMethod.TETS:
            use_S1S2 = data[self.c.COLS.SCOPE].isin(
                [EScope.S1, EScope.S2, EScope.S1S2, EScope.S1S2S3]
            )
            use_S3 = data[self.c.COLS.SCOPE].isin([EScope.S3, EScope.S1S2S3])
            assert isinstance(data[self.c.COLS.GHG_SCOPE12].dtype, PintType)
            assert isinstance(data[self.c.COLS.GHG_SCOPE3].dtype, PintType)
            if use_S3.any():
                self._check_column(data, self.c.COLS.GHG_SCOPE3)
                use_S3 = use_S3 & ~ITR.isna(data[self.c.COLS.GHG_SCOPE3])
            if use_S1S2.any():
                self._check_column(data, self.c.COLS.GHG_SCOPE12)
                use_S1S2 = use_S1S2 & ~ITR.isna(data[self.c.COLS.GHG_SCOPE12])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Calculate the total emissions of all companies.
                    # https://github.com/pandas-dev/pandas/issues/50564 explains why we need fillna(0) to make sum work
                    emissions = (
                        asPintSeries(data.loc[use_S1S2, self.c.COLS.GHG_SCOPE12])
                        .fillna(0)
                        .sum()
                        + asPintSeries(data.loc[use_S3, self.c.COLS.GHG_SCOPE3])
                        .fillna(0)
                        .sum()
                    )
                    # See https://github.com/hgrecco/pint-pandas/issues/130
                    weights_dtype = f"pint[{emissions.u}]"
                    # df_z works around fact that we cannot just insert quantities willy-nilly using Pandas where function
                    df_z = pd.Series(
                        PA_(np.zeros(len(data.index)), dtype="Mt CO2e"),
                        index=data.index,
                    )
                    weights_series = (
                        (
                            data[self.c.COLS.GHG_SCOPE12].where(use_S1S2, df_z)
                            + data[self.c.COLS.GHG_SCOPE3].where(use_S3, df_z)
                        ).astype(weights_dtype)
                        / emissions
                        * data[input_column]
                    )
                    return weights_series

            except ZeroDivisionError:
                raise ValueError("The total emissions should be higher than zero")

        elif PortfolioAggregationMethod.is_emissions_based(
            portfolio_aggregation_method
        ):
            # These four methods only differ in the way the company is valued.
            value_column = PortfolioAggregationMethod.get_value_column(
                portfolio_aggregation_method, self.c.COLS
            )
            # Used to check data[value_column].dtype.kind in ['f', 'i']
            assert isinstance(data[value_column].dtype, PintType)

            # Calculate the total owned emissions of all companies
            try:
                self._check_column(data, self.c.COLS.INVESTMENT_VALUE)
                self._check_column(data, value_column)
                use_S1S2 = data[self.c.COLS.SCOPE].isin(
                    [EScope.S1, EScope.S2, EScope.S1S2, EScope.S1S2S3]
                )
                use_S3 = data[self.c.COLS.SCOPE].isin([EScope.S3, EScope.S1S2S3])
                if use_S1S2.any():
                    self._check_column(data, self.c.COLS.GHG_SCOPE12)
                    use_S1S2 = use_S1S2 & ~ITR.isna(data[self.c.COLS.GHG_SCOPE12])
                if use_S3.any():
                    self._check_column(data, self.c.COLS.GHG_SCOPE3)
                    use_S3 = use_S3 & ~ITR.isna(data[self.c.COLS.GHG_SCOPE3])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_z = pd.Series(
                        PA_(np.zeros(len(data.index)), dtype="Mt CO2e"),
                        index=data.index,
                    )
                    data[self.c.COLS.OWNED_EMISSIONS] = (
                        data[self.c.COLS.INVESTMENT_VALUE] / data[value_column]
                    ) * (
                        data[self.c.COLS.GHG_SCOPE12].where(use_S1S2, df_z)
                        + data[self.c.COLS.GHG_SCOPE3].where(use_S3, df_z)
                    ).astype("pint[Mt CO2e]")
            except ZeroDivisionError:
                raise ValueError(
                    "To calculate the aggregation, the {} column may not be zero".format(
                        value_column
                    )
                )

            try:
                # Calculate the MOTS value per company
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    assert isinstance(data[self.c.COLS.OWNED_EMISSIONS].dtype, PintType)
                    owned_emissions = asPintSeries(data[self.c.COLS.OWNED_EMISSIONS])
                    # https://github.com/pandas-dev/pandas/issues/50564 explains why we need fillna(0) to make sum work
                    total_emissions = owned_emissions.fillna(0).sum()
                    result = data[input_column] * owned_emissions / total_emissions
                return result
            except ZeroDivisionError:
                raise ValueError("The total owned emissions can not be zero")
        else:
            raise ValueError("The specified portfolio aggregation method is invalid")
