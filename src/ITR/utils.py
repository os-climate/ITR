from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import ITR

from .configs import (
    ColumnsConfig,
    LoggingConfig,
    TemperatureScoreConfig,
    TemperatureScoreControls,
)
from .data.data_warehouse import DataWarehouse
from .data.osc_units import Q_, delta_degC_Quantity, ureg
from .interfaces import EScope, ETimeFrames, PortfolioCompany, ScoreAggregations
from .portfolio_aggregation import PortfolioAggregationMethod
from .temperature_score import TemperatureScore

from pint_pandas import PintArray, PintType

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


# If this file is moved, the computation of get_project_root may also need to change
def get_project_root() -> Path:
    return Path(__file__).parent


# If DF_COL contains Pint quantities (because it is a PintArray or an array of Pint Quantities),
# return a two-column dataframe of magnitudes and units.
# If DF_COL contains no Pint quanities, return it unchanged.
def dequantify_column(df_col: pd.Series) -> pd.DataFrame:
    if isinstance(df_col.values, PintArray):
        return pd.DataFrame(
            {
                df_col.name: df_col.values.quantity.m,
                df_col.name + "_units": str(df_col.values.dtype.units),
            },
            index=df_col.index,
        )
    elif df_col.size == 0:
        return df_col
    elif isinstance(df_col.iloc[0], Quantity):  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
            m, u = list(zip(*df_col.map(lambda x: (np.nan, "dimensionless") if pd.isna(x) else (x.m, str(x.u)))))
            return pd.DataFrame({df_col.name: m, df_col.name + "_units": u}, index=df_col.index).convert_dtypes()
    else:
        return df_col


# Rewrite dataframe DF so that columns containing Pint quantities are represented by a column for the Magnitude and column for the Units.
# The magnitude column retains the original column name and the units column is renamed with a _units suffix.
def dequantify_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([dequantify_column(df[col]) for col in df.columns], axis=1)


# Because this DF comes from reading a Trino table, and because columns must be unqiue, we don't have to enumerate to ensure we properly handle columns with duplicated names
def requantify_df(df: pd.DataFrame, typemap={}) -> pd.DataFrame:
    units_col = None
    columns_not_found = [k for k in typemap.keys() if k not in df.columns]
    if columns_not_found:
        logger.error(f"columns {columns_not_found} not found in DataFrame")
        raise ValueError
    columns_reversed = reversed(df.columns)
    for col in columns_reversed:
        if col.endswith("units") and col[-6] in "_ ":
            if units_col:
                logger.error(f"Column {units_col} follows {col} without intervening value column")
                # We expect _units column to follow a non-units column
                raise ValueError
            units_col = col
            continue
        if units_col:
            if [col, "units"] != [units_col[:-6], units_col[-5:]]:
                logger.error(f"Excpecting column name {col}_units but saw {units_col} instead")
                raise ValueError
            if (df[units_col] == df[units_col].iloc[0]).all():
                # We can make a PintArray since column is of homogeneous type
                # ...and if the first valid index matches all, we can take first row as good
                new_col = PintArray(df[col], dtype=f"pint[{ureg(df[units_col].iloc[0]).u}]")
            else:
                # Make a pd.Series of Quantity in a way that does not throw UnitStrippedWarning
                if df[col].map(lambda x: x is None).any():
                    # breakpoint()
                    raise
                new_col = pd.Series(data=df[col], name=col) * pd.Series(
                    data=df[units_col].map(
                        lambda x: typemap.get(col, ureg("dimensionless").u) if pd.isna(x) else ureg(x).u
                    ),
                    name=col,
                )
            if col in typemap.keys():
                new_col = new_col.astype(f"pint[{typemap[col]}]")
            df = df.drop(columns=units_col)
            df[col] = new_col
            units_col = None
        elif col in typemap.keys():
            df[col] = df[col].astype(f"pint[{typemap[col]}]")
    return df


def asPintSeries(series: pd.Series, name=None, errors="ignore", inplace=False) -> pd.Series:
    """
    :param series : pd.Series possibly containing Quantity values, not already in a PintArray.
    :param name : the name to give to the resulting series
    :param errors : { 'raise', 'ignore' }, default 'ignore'
    :param inplace : bool, default False.  If True, perform operation in-place.

    :return: If there is only one type of unit in the series, a PintArray version of the series, replacing NULL values with Quantity (np.nan, unit_type).

    Raises ValueError if there are more than one type of units in the series.
    Silently returns series if no conversion needed to be done.
    """

    # FIXME: Errors in the imput template can trigger this assertion
    if isinstance(series, pd.DataFrame):
        assert len(series) == 1
        series = series.iloc[0]

    if series.dtype != "O":
        if errors == "ignore":
            return series
        if name:
            raise ValueError(f"'{name}' not dtype('O')")
        elif series.name:
            raise ValueError(f"Series '{series.name}' not dtype('O')")
        else:
            raise ValueError("Series not dtype('O')")
    # NA_VALUEs are true NaNs, missing units
    na_values = ITR.isna(series)
    units = series[~na_values].map(lambda x: x.u if isinstance(x, Quantity) else None)  # type: ignore
    unit_first_idx = units.first_valid_index()
    if unit_first_idx is None:
        if errors != "ignore":
            raise ValueError(f"No value units in series: {series}")
        return series
    # Arbitrarily pick first of the most popular units, as promised
    unit = units.mode()[0]
    if inplace:
        new_series = series
    else:
        new_series = series.copy()
    if name:
        new_series.name = name
    na_index = na_values[na_values].index
    if len(na_index) > 0:
        new_series.loc[na_index] = new_series.loc[na_index].map(lambda x: PintType(unit).na_value)
    return new_series.astype(f"pint[{unit}]")


def asPintDataFrame(df: pd.DataFrame, errors="ignore", inplace=False) -> pd.DataFrame:
    """
    :param df : pd.DataFrame with columns to be converted into PintArrays where possible.
    :param errors : { 'raise', 'ignore' }, default 'ignore'
    :param inplace : bool, default False.  If True, perform operation in-place.

    :return: A pd.DataFrame with columns converted to PintArrays where possible.

    Raises ValueError if there are more than one type of units in any of the columns.
    """
    if inplace:
        new_df = df
    else:
        new_df = pd.DataFrame()
    for col in df.columns:
        new_df[col] = asPintSeries(df[col], name=col, errors=errors, inplace=inplace)
    new_df.index = df.index
    # When DF.COLUMNS is a MultiIndex, the naive column-by-column construction replaces MultiIndex values
    # with the anonymous tuple of the MultiIndex and DF.COLUMNS becomes just an Index of tuples.
    # We need to restore the MultiIndex or lose information.
    new_df.columns = df.columns
    return new_df


def requantify_df_from_columns(df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    :param df: pd.DataFrame
    :param inplace: bool, default False.  If True, perform operation in-place.

    :return: A pd.DataFrame with columns originally matching the pattern COLUMN_NAME [UNITS] renamed to COLUMN_NAME and replaced with a PintArray with dtype=ureg(UNITS) (aka 'pint[UNITS]')
    """
    p = re.compile(r"^(.*)\s*\[(.*)\]\s*$")
    if not inplace:
        df = df.copy()
    for column in df.columns:
        m = p.match(column)
        if m:
            col = m.group(1).strip()
            unit = m.group(2).strip()
            df.rename(columns={column: col}, inplace=True)
            df[col] = pd.Series(PA_(df[col], unit))
    return df


def _flatten_user_fields(record: PortfolioCompany):
    """
    Flatten the user fields in a portfolio company and return it as a dictionary.

    :param record: The record to flatten
    :return:
    """
    record_dict = record.model_dump(exclude_none=True)
    if record.user_fields is not None:
        for key, value in record_dict["user_fields"].items():
            record_dict[key] = value
        del record_dict["user_fields"]

    return record_dict


def dataframe_to_portfolio(df_portfolio: pd.DataFrame) -> List[PortfolioCompany]:
    """
    Convert a data frame to a list of portfolio company objects.

    :param df_portfolio: The data frame to parse. The column names should align with the attribute names of the PortfolioCompany model.
    :return: A list of portfolio companies
    """
    # Adding some non-empty checks for portfolio upload
    if df_portfolio[ColumnsConfig.INVESTMENT_VALUE].isnull().any():
        error_message = "Investment values are missing for one or more companies in the input file."
        logger.error(error_message)
        raise ValueError(error_message)
    if df_portfolio[ColumnsConfig.COMPANY_ISIN].isnull().any():
        error_message = "Company ISINs are missing for one or more companies in the input file."
        logger.error(error_message)
        raise ValueError(error_message)
    if df_portfolio[ColumnsConfig.COMPANY_ID].isnull().any():
        error_message = "Company IDs are missing for one or more companies in the input file."
        logger.error(error_message)
        raise ValueError(error_message)

    return [PortfolioCompany.model_validate(company) for company in df_portfolio.to_dict(orient="records")]


def get_data(data_warehouse: DataWarehouse, portfolio: List[PortfolioCompany]) -> pd.DataFrame:
    """
    Get the required data from the data provider(s) and return a 9-box grid for each company.

    :param data_warehouse: DataWarehouse instances
    :param portfolio: A list of PortfolioCompany models
    :return: A data frame containing the relevant company data indexed by (COMPANY_ID, SCOPE)
    """
    company_ids = set(data_warehouse.company_data.get_company_ids())
    df_portfolio = pd.DataFrame.from_records(
        [_flatten_user_fields(c) for c in portfolio if c.company_id in company_ids]
    )
    df_portfolio[ColumnsConfig.INVESTMENT_VALUE] = asPintSeries(df_portfolio[ColumnsConfig.INVESTMENT_VALUE])

    if ColumnsConfig.COMPANY_ID not in df_portfolio.columns:
        raise ValueError("Portfolio contains no company_id data")

    # This transforms a dataframe of portfolio data into model data just so we can transform that back into a dataframe?!
    # It does this for all scopes, not only the scopes of interest
    company_data = data_warehouse.get_preprocessed_company_data(df_portfolio[ColumnsConfig.COMPANY_ID].to_list())

    if len(company_data) == 0:
        raise ValueError("None of the companies in your portfolio could be found by the data providers")

    df_company_data = pd.DataFrame.from_records([dict(c) for c in company_data])
    # Until we have https://github.com/hgrecco/pint-pandas/pull/58...
    df_company_data.ghg_s1s2 = df_company_data.ghg_s1s2.astype("pint[Mt CO2e]")
    s3_data_invalid = df_company_data[ColumnsConfig.GHG_SCOPE3].isna()
    if len(s3_data_invalid[s3_data_invalid].index) > 0:
        df_company_data.loc[s3_data_invalid, ColumnsConfig.GHG_SCOPE3] = df_company_data.loc[
            s3_data_invalid, ColumnsConfig.GHG_SCOPE3
        ].map(lambda x: Q_(np.nan, "Mt CO2e"))
    for col in [
        ColumnsConfig.GHG_SCOPE3,
        ColumnsConfig.CUMULATIVE_BUDGET,
        ColumnsConfig.CUMULATIVE_SCALED_BUDGET,
        ColumnsConfig.CUMULATIVE_TARGET,
        ColumnsConfig.CUMULATIVE_TRAJECTORY,
    ]:
        df_company_data[col] = df_company_data[col].astype("pint[Mt CO2e]")
    for col in [
        ColumnsConfig.COMPANY_REVENUE,
        ColumnsConfig.COMPANY_MARKET_CAP,
        ColumnsConfig.COMPANY_ENTERPRISE_VALUE,
        ColumnsConfig.COMPANY_EV_PLUS_CASH,
        ColumnsConfig.COMPANY_TOTAL_ASSETS,
        ColumnsConfig.COMPANY_CASH_EQUIVALENTS,
    ]:
        df_company_data[col] = asPintSeries(df_company_data[col])
    df_company_data[ColumnsConfig.BENCHMARK_TEMP] = df_company_data[ColumnsConfig.BENCHMARK_TEMP].astype(
        "pint[delta_degC]"
    )
    df_company_data[ColumnsConfig.BENCHMARK_GLOBAL_BUDGET] = df_company_data[
        ColumnsConfig.BENCHMARK_GLOBAL_BUDGET
    ].astype("pint[Gt CO2e]")
    portfolio_data = pd.merge(
        left=df_company_data,
        right=df_portfolio.drop(ColumnsConfig.COMPANY_NAME, axis=1),
        how="left",
        on=[ColumnsConfig.COMPANY_ID],
    ).set_index([ColumnsConfig.COMPANY_ID, ColumnsConfig.SCOPE])
    return portfolio_data


def get_benchmark_projections(
    prod_df: pd.DataFrame, company_sector_region_scope: Optional[pd.DataFrame] = None, scope: EScope = EScope.AnyScope
) -> pd.DataFrame:
    """
    :param prod_df: DataFrame of production statistics by sector, region, scope (and year)
    :param company_sector_region_scope: DataFrame indexed by ColumnsConfig.COMPANY_ID
    with at least the following columns: ColumnsConfig.SECTOR, ColumnsConfig.REGION, and ColumnsConfig.SCOPE
    :param scope: a scope
    :return: A pint[dimensionless] DataFrame with partial production benchmark data per calendar year per row, indexed by company.
    """

    if company_sector_region_scope is None:
        return prod_df

    # We drop the meaningless S1S2/AnyScope from the production benchmark and replace it with the company's scope.
    # This is needed to make indexes align when we go to multiply production times intensity for a scope.
    prod_df_anyscope = prod_df.droplevel(ColumnsConfig.SCOPE)
    df = (
        company_sector_region_scope[[ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE]]
        .reset_index()
        .drop_duplicates()
        .set_index(["company_id", ColumnsConfig.SCOPE])
    )
    # We drop the meaningless S1S2/AnyScope from the production benchmark and replace it with the company's scope.
    # This is needed to make indexes align when we go to multiply production times intensity for a scope.
    company_benchmark_projections = df.merge(
        prod_df_anyscope,
        left_on=[ColumnsConfig.SECTOR, ColumnsConfig.REGION],
        right_index=True,
        how="left",
    )
    # If we don't get a match, then the projections will be `nan`.  Look at the last year's column to find them.
    mask = company_benchmark_projections.iloc[:, -1].isna()
    if mask.any():
        # Patch up unknown regions as "Global"
        global_benchmark_projections = (
            df[mask]
            .drop(columns=ColumnsConfig.REGION)
            .merge(
                prod_df_anyscope.loc[(slice(None), "Global"), :].droplevel([ColumnsConfig.REGION]),
                left_on=[ColumnsConfig.SECTOR],
                right_index=True,
                how="left",
            )
        )
        combined_benchmark_projections = pd.concat(
            [
                company_benchmark_projections[~mask].drop(columns=ColumnsConfig.REGION),
                global_benchmark_projections,
            ]
        )
        return combined_benchmark_projections.drop(columns=ColumnsConfig.SECTOR)
    return company_benchmark_projections.drop(columns=[ColumnsConfig.SECTOR, ColumnsConfig.REGION])


def calculate(
    portfolio_data: pd.DataFrame,
    fallback_score: delta_degC_Quantity,
    aggregation_method: PortfolioAggregationMethod,
    grouping: Optional[List[str]],
    time_frames: List[ETimeFrames],
    scopes: List[EScope],
    anonymize: bool,
    aggregate: bool = True,
    controls: Optional[TemperatureScoreControls] = None,
) -> Tuple[pd.DataFrame, Optional[ScoreAggregations]]:
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
    ts = TemperatureScore(
        time_frames=time_frames,
        scopes=scopes,
        fallback_score=fallback_score,
        grouping=grouping,
        aggregation_method=aggregation_method,
        config=config,
    )

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
    arr = np.array(
        [v if isinstance(v, ITR.UFloat) else ITR.ufloat(v, 0) for v in unquantified_data if not ITR.isnan(v)]
    )
    N = len(arr)
    if N == 0:
        return np.nan
    if N == 1:
        return arr[0]
    nominals = ITR.nominal_values(arr)
    if any(ITR.std_devs(arr) == 0):
        # We cannot mix and match "perfect" measurements with uncertainties
        # Instead compute the mean and return the "standard error" as the uncertainty
        # e.g. ITR.umean([100, 200]) = 150 +/- 50
        w_mean = sum(nominals) / N
        w_std = np.std(nominals) / np.sqrt(N - 1)
    else:
        # Compute the "uncertainty of the weighted mean", which apparently
        # means ignoring whether or not there are large uncertainties
        # that should be created by elements that disagree
        # e.g. ITR.umean([100+/-1, 200+/-1]) = 150.0+/-0.7 (!)
        w_sigma = 1 / sum([1 / (v.s**2) for v in arr])
        w_mean = sum([v.n / (v.s**2) for v in arr]) * w_sigma
        w_std = w_sigma * np.sqrt(sum([1 / (v.s**2) for v in arr]))
    result = ITR.ufloat(w_mean, w_std)
    return result


def uround(u, ndigits):
    """
    Round an uncertainty to ndigits.
    """
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
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
