from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import ITR
from ITR.data.osc_units import ureg
import pint

from .interfaces import PortfolioCompany, EScope, ETimeFrames, ScoreAggregations, TemperatureScoreControls
from .configs import ColumnsConfig, TemperatureScoreConfig, LoggingConfig, logger
from .data.data_warehouse import DataWarehouse
from .portfolio_aggregation import PortfolioAggregationMethod
from .temperature_score import TemperatureScore

# If this file is moved, the computation of get_project_root may also need to change
def get_project_root() -> Path:
    return Path(__file__).parent


def _flatten_user_fields(record: PortfolioCompany):
    """
    Flatten the user fields in a portfolio company and return it as a dictionary.

    :param record: The record to flatten
    :return:
    """
    record_dict = record.dict(exclude_none=True)
    if record.user_fields is not None:
        for key, value in record_dict["user_fields"].items():
            record_dict[key] = value
        del record_dict["user_fields"]

    return record_dict


def _make_isin_map(df_portfolio: pd.DataFrame) -> dict:
    """
    Create a mapping from company_id to ISIN

    :param df_portfolio: The complete portfolio
    :return: A mapping from company_id to ISIN
    """
    return {company_id: company[ColumnsConfig.COMPANY_ISIN]
            for company_id, company in df_portfolio[[ColumnsConfig.COMPANY_ID, ColumnsConfig.COMPANY_ISIN]]
                .set_index(ColumnsConfig.COMPANY_ID)
                .to_dict(orient='index').items()}


def dataframe_to_portfolio(df_portfolio: pd.DataFrame) -> List[PortfolioCompany]:
    """
    Convert a data frame to a list of portfolio company objects.

    :param df_portfolio: The data frame to parse. The column names should align with the attribute names of the
    PortfolioCompany model.
    :return: A list of portfolio companies
    """
    # Adding some non-empty checks for portfolio upload
    if df_portfolio[ColumnsConfig.INVESTMENT_VALUE].isnull().any():
        error_message = f"Investment values are missing for one or more companies in the input file."
        logger.error(error_message)
        raise ValueError(error_message)
    if df_portfolio[ColumnsConfig.COMPANY_ISIN].isnull().any():
        error_message = f"Company ISINs are missing for one or more companies in the input file."
        logger.error(error_message)
        raise ValueError(error_message)
    if df_portfolio[ColumnsConfig.COMPANY_ID].isnull().any():
        error_message = f"Company IDs are missing for one or more companies in the input file."
        logger.error(error_message)
        raise ValueError(error_message)

    return [PortfolioCompany.parse_obj(company) for company in df_portfolio.to_dict(orient="records")]


def get_data(data_warehouse: DataWarehouse, portfolio: List[PortfolioCompany]) -> pd.DataFrame:
    """
    Get the required data from the data provider(s) and return a 9-box grid for each company.

    :param data_warehouse: DataWarehouse instances
    :param portfolio: A list of PortfolioCompany models
    :return: A data frame containing the relevant company data
    """
    df_portfolio = pd.DataFrame.from_records([_flatten_user_fields(c) for c in portfolio])

    company_data = data_warehouse.get_preprocessed_company_data(df_portfolio[ColumnsConfig.COMPANY_ID].to_list())

    if len(company_data) == 0:
        raise ValueError("None of the companies in your portfolio could be found by the data providers")

    df_company_data = pd.DataFrame.from_records([c.dict() for c in company_data])
    portfolio_data = pd.merge(left=df_company_data, right=df_portfolio.drop("company_name", axis=1), how="left",
                              on=["company_id"])

    return portfolio_data


def calculate(portfolio_data: pd.DataFrame, fallback_score: pint.Quantity['delta_degC'], aggregation_method: PortfolioAggregationMethod,
              grouping: Optional[List[str]], time_frames: List[ETimeFrames],
              scopes: List[EScope], anonymize: bool, aggregate: bool = True,
              controls: Optional[TemperatureScoreControls] = None) -> Tuple[pd.DataFrame,
                                                                            Optional[ScoreAggregations]]:
    """
    Calculate the different parts of the temperature score (actual scores, aggregations, column distribution).

    :param portfolio_data: The portfolio data, already processed by the target validation module
    :param fallback_score: The fallback score to use while calculating the temperature score
    :param aggregation_method: The aggregation method to use
    :param time_frames: The time frames that the temperature scores should be calculated for  (None to calculate all)
    :param scopes: The scopes that the temperature scores should be calculated for (None to calculate all)
    :param grouping: The names of the columns to group on
    :param anonymize: Whether to anonymize the resulting data set or not
    :param aggregate: Whether to aggregate the scores or not
    :return: The scores, the aggregations and the column distribution (if a
    """

    config = TemperatureScoreConfig
    if controls:
        TemperatureScoreConfig.CONTROLS_CONFIG = controls
    ts = TemperatureScore(time_frames=time_frames, scopes=scopes, fallback_score=fallback_score,
                          grouping=grouping, aggregation_method=aggregation_method, config=config)

    scores = ts.calculate(portfolio_data)
    aggregations = None
    if aggregate:
        aggregations = ts.aggregate_scores(scores)

    if anonymize:
        scores = ts.anonymize_data_dump(scores)

    return scores, aggregations


# https://stackoverflow.com/a/74137209/1291237
def umean(quantified_data):
    """
    Assuming Gaussian statistics, uncertainties stem from Gaussian parent distributions. In such a case,
    it is standard to weight the measurements (nominal values) by the inverse variance.

    This function uses error propagation on the to get an uncertainty of the weighted average.
    :param: A set of uncertainty values
    :return: The weighted mean of the values, with a freshly calculated error term
    """
    values = np.array(list(map(lambda v: v.m if isinstance(v.m, ITR.UFloat) else ITR.ufloat(v.m, 0), quantified_data)))
    epsilon = 1e-7
    wavg = ITR.ufloat(sum([v.n/(v.s**2+epsilon) for v in values])/sum([1/(v.s**2+epsilon) for v in values]), 
                      np.sqrt(len(values)/sum([1/(v.s**2+epsilon) for v in values])))
    if wavg.s==0.0:
        # Uncertainties of zero can unpromote back to floats
        return wavg.n
    elif wavg.s < epsilon:
        logger.debug(f"Casting out small uncertainty {wavg.s} from {wavg}; epsilon = {epsilon}.")
        wavg = wavg.n

    return wavg
