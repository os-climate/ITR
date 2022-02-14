"""
This module handles initialization of pint functionality
"""

from pint import set_application_registry
from pint_pandas import PintArray, PintType
from openscm_units import unit_registry
PintType.ureg = unit_registry
ureg = unit_registry
set_application_registry(ureg)
Q_ = ureg.Quantity
PA_ = PintArray

ureg.define("CO2e = CO2 = CO2eq = CO2_eq")
ureg.define("Fe_ton = [produced_ton]")

# These are for later
ureg.define('fraction = [] = frac')
ureg.define('percent = 1e-2 frac = pct = percentage')
ureg.define('ppm = 1e-6 fraction')

ureg.define("USD = [currency]")
ureg.define("EUR = nan USD")
ureg.define("JPY = nan USD")

ureg.define("btu = Btu")
ureg.define("boe = 5.712 GJ")

# These are for later still
# ureg.define("HFC = [ HFC_emissions ]")
# ureg.define("PFC = [ PFC_emissions ]")
# ureg.define("mercury = Hg = Mercury")
# ureg.define("mercure = Hg = Mercury")
# ureg.define("PM10 = [ PM10_emissions ]")
