"""
This module handles initialization of pint functionality
"""

import numpy as np
import pandas as pd
from pint import get_application_registry

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

def asPintSeries(series: pd.Series, name=None, errors='ignore') -> pd.Series:
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
    units = series[~na_values].map(lambda x: x.u)
    first_unit = units[units.first_valid_index()]
    if len(set(units.values.tolist()))==1:
        new_series = series.copy()
        # FIXME: new_series.loc[na_values[na_values].index] = Q_(np.nan, first_unit) doesn't work, so...
        for idx in na_values[na_values].index:
            new_series.loc[idx] = Q_(np.nan, first_unit)
        return new_series.astype(f"pint[{first_unit}]")
    if errors != 'ignore':
        raise ValueError(f"Element types not homogeneously ({first_unit})")
    return series

def asPintDataFrame(df: pd.DataFrame, errors='ignore') -> pd.DataFrame:
    new_df = pd.DataFrame()
    for col in df.columns:
        new_df[col] = asPintSeries(df[col], name=col, errors=errors)
    return new_df
