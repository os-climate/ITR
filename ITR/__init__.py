"""
This package helps companies and financial institutions to assess the temperature alignment of investment and lending
portfolios.
"""
from .data import osc_units
from . import data
from . import utils
from . import temperature_score

try:
    import numpy as np
    from uncertainties import ufloat, UFloat
    from uncertainties.unumpy import uarray, isnan, nominal_values, std_devs
    from .utils import umean
    HAS_UNCERTAINTIES = True
    _ufloat_nan = ufloat(np.nan, 0.0)
except ModuleNotFoundError:
    HAS_UNCERTAINTIES = False
    from numpy import isnan
