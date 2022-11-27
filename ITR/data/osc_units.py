"""
This module handles initialization of pint functionality
"""

import numpy as np
import pandas as pd
from pint import get_application_registry, Quantity

ureg = get_application_registry()

Q_ = ureg.Quantity
M_ = ureg.Measurement

# FIXME: delay loading of pint_pandas until after we've initialized ourselves
from pint_pandas import PintType, PintArray
PintType.ureg = ureg
PA_ = PintArray

ureg.define("CO2e = CO2 = CO2eq = CO2_eq")
ureg.define("LNG = 3.44 / 2.75 CH4")
# with ureg.context("CH4_conversions"):
#     print(ureg("t LNG").to("t CO2"))
# will print 3.44 t CO2

ureg.define("Fe = [iron] = Steel")
ureg.define("iron = Fe")
ureg.define("Al = [aluminum] = Aluminum")
ureg.define("aluminium = Al")
ureg.define("Cement = [cement]")
ureg.define("cement = Cement")
ureg.define("Cu = [copper] = Copper")
ureg.define("Paper = [paper] = Pulp")
ureg.define("Paperboard = Paper")

# For reports that use 10,000 t instead of 1e3 or 1e6
ureg.define('myria- = 10000')

# These are for later
ureg.define('fraction = [] = frac')
ureg.define('percent = 1e-2 frac = pct = percentage')
ureg.define('ppm = 1e-6 fraction')

ureg.define("USD = [currency]")
ureg.define("EUR = nan USD")
ureg.define("JPY = nan USD")

ureg.define("btu = Btu")
ureg.define("mmbtu = 1e6 btu")
# ureg.define("boe = 5.712 GJ")
ureg.define("boe = 6.1178632 GJ")
ureg.define("mboe = 1e3 boe")
ureg.define("mmboe = 1e6 boe")

# Transportation activity

ureg.define("vehicle = [vehicle] = v")
ureg.define("passenger = [passenger] = p = pass")
ureg.define("vkm = vehicle * kilometer")
ureg.define("pkm = passenger * kilometer")
ureg.define("tkm = tonne * kilometer")

ureg.define('hundred = 1e2')
ureg.define('thousand = 1e3')
ureg.define('million = 1e6')
ureg.define('billion = 1e9')
ureg.define('trillion = 1e12')
ureg.define('quadrillion = 1e15')

# Backward compatibility
ureg.define("Fe_ton = t Steel")



# These are for later still
# ureg.define("HFC = [ HFC_emissions ]")
# ureg.define("PFC = [ PFC_emissions ]")
# ureg.define("mercury = Hg = Mercury")
# ureg.define("mercure = Hg = Mercury")
# ureg.define("PM10 = [ PM10_emissions ]")

def asPintSeries(series: pd.Series, name=None, errors='ignore', inplace=False) -> pd.Series:
    """
    Parameters
    ----------
    series : pd.Series possibly containing Quantity values, not already in a PintArray.
    name : the name to give to the resulting series
    errors : { 'raise', 'ignore' }, default 'ignore'
    inplace : bool, default False
             If True, perform operation in-place.

    Returns
    -------
    If there is only one type of unit in the series, a PintArray version of the series,
    replacing NULL values with Quantity (np.nan, unit_type).

    Raises ValueError if there are more than one type of units in the series.
    Silently series if no conversion needed to be done.
    """
    if series.dtype != 'O':
        if errors == 'ignore':
            return series
        if name:
            raise ValueError ("'{name}' not dtype('O')")
        elif series.name:
            raise ValueError ("Series '{series.name}' not dtype('O')")
        else:
            raise ValueError ("Series not dtype('O')")
    na_values = series.isna()
    units = series[~na_values].map(lambda x: x.u if isinstance(x, Quantity) else None)
    unit_first_idx = units.first_valid_index()
    if unit_first_idx is not None and len(set(units.values.tolist()))==1:
        first_unit = units[unit_first_idx]
        if inplace:
            new_series = series
        else:
            new_series = series.copy()
        if name:
            new_series.name = name
        na_index = na_values[na_values].index
        new_series.loc[na_index] = pd.Series(Q_(np.nan, first_unit), index=na_index)
        return new_series.astype(f"pint[{first_unit}]")
    if errors != 'ignore':
        raise ValueError(f"Element types not homogeneously ({first_unit}): {series}")
    return series

def asPintDataFrame(df: pd.DataFrame, errors='ignore', inplace=False) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame with columns to be converted into PintArrays where possible.
    errors : { 'raise', 'ignore' }, default 'ignore'
    inplace : bool, default False
             If True, perform operation in-place.

    Returns
    -------
    A pd.DataFrame with columns converted to PintArrays where possible.
    Raises ValueError if there are more than one type of units in any of the columns.
    """
    if inplace:
        new_df = df
    else:
        new_df = pd.DataFrame()
    for col in df.columns:
        new_df[col] = asPintSeries(df[col], name=col, errors=errors, inplace=inplace)
    new_df.index = df.index
    return new_df
