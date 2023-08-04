"""
This package helps companies and financial institutions to assess the temperature alignment of investment and lending
portfolios.
"""
import warnings
import pandas as pd
import numpy as np
import json
from .data import osc_units
from .interfaces import EScope
from . import data
from . import utils
from . import temperature_score
import pint
from pint_pandas import PintType, PintArray

try:
    # Even if we have uncertainties available as a module, we don't have the right version of pint
    if hasattr(pint.compat, 'tokenizer'):
        raise AttributeError
    from uncertainties import ufloat, UFloat
    from uncertainties.unumpy import uarray, isnan, nominal_values, std_devs
    from .utils import umean
    _ufloat_nan = ufloat(np.nan, 0.0)
    pint.pint_eval.tokenizer = pint.pint_eval.uncertainty_tokenizer
    HAS_UNCERTAINTIES = True
except (ImportError, ModuleNotFoundError, AttributeError):
    HAS_UNCERTAINTIES = False
    from numpy import isnan
    from statistics import mean

    def nominal_values(x):
        return x

    def std_devs(x):
        if hasattr(x, '__len__'):
            return [0] * len(x)
        return 0

    def uarray(nom_vals, std_devs):
        return nom_vals

    def umean(unquantified_data):
        return mean(unquantified_data)

def isna(x):
    '''
    True if X is either a NaN-like Quantity or otherwise NA-like
    '''
    # This function simplifies dealing with NA vs. NaN quantities and magnitudes inside and outside of PintArrays
    if isinstance(x, pint.Quantity):
        x = x.m
    if HAS_UNCERTAINTIES and isinstance(x, UFloat):
        return isnan(x)
    return pd.isna(x)

def Q_m_as(value, units, inplace=False):
    '''
    Convert VALUE from a string to a Quantity.
    If the Quanity is not already in UNITS, then convert in place.
    Returns the MAGNITUDE of the (possibly) converted value.
    '''
    x = value
    if type(value)==str:
        x = pint.Quantity(value)
    if x.u==units:
        return x.m
    if inplace:
        x.ito(units)
        return x.m
    return x.to(units).m


def recombine_nom_and_std(nom: pd.Series, std: pd.Series) -> pd.Series:
    '''
    A Pandas-friendly way to combine nominal and error terms for uncertainties
    '''
    assert HAS_UNCERTAINTIES
    if std.sum()==0:
        return nom
    return pd.Series(data=uarray(nom.values, np.where(nom.notna(), std.values, 0)), index=nom.index, name=nom.name)


def JSONEncoder(q):
    if isinstance(q, pint.Quantity):
        if pd.isna(q.m):
            return f"nan {q.u}"
        return f"{q:.5f}"
    elif isinstance(q, EScope):
        return q.name
    elif isinstance(q, pd.Series):
        # Inside the map function NA values become float64 nans and lose their units
        res = pd.DataFrame(q.map(lambda x: f"nan {q.pint.u}" if isna(x) else f"{x:.5f}"), columns=['value']).reset_index().to_dict('records')
        return res
    else:
        return str(q)
