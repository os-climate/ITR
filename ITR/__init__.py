"""
This package helps companies and financial institutions to assess the temperature alignment of investment and lending
portfolios.
"""
from .data import osc_units
from . import data
from . import utils
from . import temperature_score

try:
    from uncertainties import ufloat, UFloat, uarray
    from uncertainties.unumpy import isnan, nominal_values, std_devs
    HAS_UNCERTAINTIES = True
    _ufloat_nan = ufloat(np.nan, 0.0)
except ModuleNotFoundError:
    HAS_UNCERTAINTIES = False
    from numpy import isnan
