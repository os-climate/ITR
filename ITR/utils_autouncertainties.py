from __future__ import annotations

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import ITR
from ITR.data.osc_units import ureg, Q_, asPintSeries
import pint

from .interfaces import PortfolioCompany, EScope, ETimeFrames, ScoreAggregations
from .configs import ColumnsConfig, TemperatureScoreControls, TemperatureScoreConfig, LoggingConfig

import logging
logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)

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

    :param df_portfolio: The data frame to parse. The column names should align with the attribute names of the PortfolioCompany model.
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
    :return: A data frame containing the relevant company data indexed by (COMPANY_ID, SCOPE)
    """
    df_portfolio = pd.DataFrame.from_records([_flatten_user_fields(c) for c in portfolio
                                              if c.company_id not in data_warehouse.company_data.missing_ids])
    df_portfolio[ColumnsConfig.INVESTMENT_VALUE] = asPintSeries(df_portfolio[ColumnsConfig.INVESTMENT_VALUE])

    if ColumnsConfig.COMPANY_ID not in df_portfolio.columns:
        raise ValueError(f"Portfolio contains no company_id data")

    # This transforms a dataframe of portfolio data into model data just so we can transform that back into a dataframe?!
    # It does this for all scopes, not only the scopes of interest
    company_data = data_warehouse.get_preprocessed_company_data(df_portfolio[ColumnsConfig.COMPANY_ID].to_list())
    
    if len(company_data) == 0:
        raise ValueError("None of the companies in your portfolio could be found by the data providers")

    df_company_data = pd.DataFrame.from_records([dict(c) for c in company_data])
    # Until we have https://github.com/hgrecco/pint-pandas/pull/58...
    df_company_data.ghg_s1s2 = df_company_data.ghg_s1s2.astype('pint[Mt CO2e]')
    s3_data_invalid = df_company_data[ColumnsConfig.GHG_SCOPE3].isna()
    if len(s3_data_invalid[s3_data_invalid].index)>0:
        df_company_data.loc[s3_data_invalid, ColumnsConfig.GHG_SCOPE3] = df_company_data.loc[s3_data_invalid, ColumnsConfig.GHG_SCOPE3].map(
            lambda x: Q_(np.nan, 'Mt CO2e'))
    for col in [ColumnsConfig.GHG_SCOPE3, ColumnsConfig.CUMULATIVE_BUDGET, ColumnsConfig.CUMULATIVE_SCALED_BUDGET,
                ColumnsConfig.CUMULATIVE_TARGET, ColumnsConfig.CUMULATIVE_TRAJECTORY]:
        df_company_data[col] = df_company_data[col].astype('pint[Mt CO2e]')
    for col in [ColumnsConfig.COMPANY_REVENUE, ColumnsConfig.COMPANY_MARKET_CAP, ColumnsConfig.COMPANY_ENTERPRISE_VALUE, ColumnsConfig.COMPANY_EV_PLUS_CASH, ColumnsConfig.COMPANY_TOTAL_ASSETS, ColumnsConfig.COMPANY_CASH_EQUIVALENTS]:
        df_company_data[col] = asPintSeries(df_company_data[col])
    df_company_data[ColumnsConfig.BENCHMARK_TEMP] = df_company_data[ColumnsConfig.BENCHMARK_TEMP].astype('pint[delta_degC]')
    df_company_data[ColumnsConfig.BENCHMARK_GLOBAL_BUDGET] = df_company_data[ColumnsConfig.BENCHMARK_GLOBAL_BUDGET].astype('pint[Gt CO2e]')
    portfolio_data = pd.merge(left=df_company_data, right=df_portfolio.drop(ColumnsConfig.COMPANY_NAME, axis=1), how="left",
                              on=[ColumnsConfig.COMPANY_ID]).set_index([ColumnsConfig.COMPANY_ID, ColumnsConfig.SCOPE])
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
def umean(unquantified_data):
    """
    Assuming Gaussian statistics, uncertainties stem from Gaussian parent distributions. In such a case,
    it is standard to weight the measurements (nominal values) by the inverse variance.

    Following the pattern of np.mean, this function is really nan_mean, meaning it calculates based on non-NaN values.
    If there are no such, it returns np.nan, just like np.mean does with an empty array.

    This function uses error propagation on the to get an uncertainty of the weighted average.
    :param: A set of uncertainty values
    :return: The weighted mean of the values, with a freshly calculated error term
    """
    from math import sqrt

    values = np.array(list(map(lambda v: v if isinstance(v, ITR.UFloat) else ITR.ufloat(v, 0), unquantified_data)), dtype="object")
    values_notnan = [v for v in values if not ITR.isnan(v)]
    if ITR.HAS_AUTOUNCERTAINTIES:
        if sum([v._nom for v in values_notnan])==0:
            if len(values_notnan)==0:
                return np.nan
            return sum([v._nom for v in values_notnan]) / len(values_notnan)
        minval = min([abs(v._nom) for v in values_notnan])
        if minval==0:
            epsilon = 1e-12
        else:
            epsilon = 1e-12 * minval
        wavg = ITR.ufloat(sum([v._nom/(v._err**2+epsilon) for v in values])/sum([1/(v._err**2+epsilon) for v in values]), 
                          np.sqrt(len(values)/sum([1/(v._err**2+epsilon) for v in values])))
        if wavg._err <= sqrt(epsilon):
            logger.debug(f"Casting out small uncertainty {wavg._err} from {wavg}; epsilon = {epsilon}.")
            wavg = wavg._nom
    else:
        if sum([v.s for v in values_notnan])==0:
            if len(values_notnan)==0:
                return np.nan
            return sum([v.n for v in values_notnan]) / len(values_notnan)
        minval = min([abs(v.n) for v in values_notnan])
        if minval==0:
            epsilon = 1e-12
        else:
            epsilon = 1e-12 * minval
        wavg = ITR.ufloat(sum([v.n/(v.s**2+epsilon) for v in values])/sum([1/(v.s**2+epsilon) for v in values]), 
                          np.sqrt(len(values)/sum([1/(v.s**2+epsilon) for v in values])))
        if wavg.s <= sqrt(epsilon):
            logger.debug(f"Casting out small uncertainty {wavg.s} from {wavg}; epsilon = {epsilon}.")
            wavg = wavg.n

    return wavg


def uround(u, ndigits):
    """
    Round an uncertainty to ndigits.
    """
    if ITR.HAS_AUTOUNCERTAINTIES:
        breakpoint()
        return u.round(ndigits)
    if np.isnan(u.n):
        return ITR._ufloat_nan
    if np.isnan(u.s):
        return ITR.ufloat(round(u.n, ndigits), u.s)
    return ITR.ufloat(round(u.n, ndigits), round(u.s, ndigits))

try:
    import uncertainties
    uncertainties.UFloat.__round__ = uround
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


# From https://goshippo.com/blog/measure-real-size-any-python-object/
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
