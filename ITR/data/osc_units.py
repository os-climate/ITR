"""
This module handles initialization of pint functionality
"""

from pint import set_application_registry
from pint_pandas import PintArray, PintType
from openscm_units import unit_registry

# openscm_units doesn't make it easy to set preprocessors.  This is one way to do it.
unit_registry.preprocessors=[
     lambda s1: s1.replace('BoE', 'boe'),
]

PintType.ureg = unit_registry
ureg = unit_registry
set_application_registry(ureg)
Q_ = ureg.Quantity
PA_ = PintArray

ureg.define("CO2e = CO2 = CO2eq = CO2_eq")
ureg.define("LNG = 3.44 / 2.75 CH4")
# with ureg.context("CH4_conversions"):
#     print(ureg("t LNG").to("t CO2"))
# will print 3.44 t CO2

ureg.define("Fe_ton = [produced_ton]")
ureg.define("passenger = [passenger_unit]")

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

# These are for later still
# ureg.define("HFC = [ HFC_emissions ]")
# ureg.define("PFC = [ PFC_emissions ]")
# ureg.define("mercury = Hg = Mercury")
# ureg.define("mercure = Hg = Mercury")
# ureg.define("PM10 = [ PM10_emissions ]")
