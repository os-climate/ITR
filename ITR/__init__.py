"""
This package helps companies and financial institutions to assess the temperature alignment of investment and lending
portfolios.
"""
import pandas as pd
import numpy as np
from .data import osc_units
from . import data
from . import utils
from . import temperature_score
import pint

try:
    from uncertainties import ufloat, UFloat
    from uncertainties.unumpy import uarray, isnan, nominal_values, std_devs
    from .utils import umean
    HAS_UNCERTAINTIES = True
    _ufloat_nan = ufloat(np.nan, 0.0)
    pint.pint_eval.tokenizer = pint.pint_eval.uncertainty_tokenizer
except (ImportError, ModuleNotFoundError):
    HAS_UNCERTAINTIES = False
    from numpy import isnan
    from statistics import mean

    def nominal_values(x):
        return x

    def std_devs(x):
        if isinstance(x, float):
            return 0
        return [0] * len(x)

    def uarray(nom_vals, std_devs):
        return nom_vals

    def umean(unquantified_data):
        return mean(unquantified_data)

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
    assert HAS_UNCERTAINTIES
    if std.sum()==0:
        return nom
    assert not std.isna().any()
    return pd.Series(data=uarray(nom.values, std.values), index=nom.index, name=nom.name)

