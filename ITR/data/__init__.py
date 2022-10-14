"""
This module contains classes that create connections to data providers and initializes our system of units
"""

import pint
from pint import set_application_registry
from openscm_units import unit_registry
import re

# openscm_units doesn't make it easy to set preprocessors.  This is one way to do it.
unit_registry.preprocessors=[
    lambda s1: re.sub(r'passenger.km', 'pkm', s1),
    lambda s2: s2.replace('BoE', 'boe'),
]

ureg = unit_registry
set_application_registry(ureg)

# Overwrite what pint/pint/__init__.py initalizes
# # Default Quantity, Unit and Measurement are the ones
# # build in the default registry.
# Quantity = UnitRegistry.Quantity
# Unit = UnitRegistry.Unit
# Measurement = UnitRegistry.Measurement
# Context = UnitRegistry.Context

pint.Quantity = ureg.Quantity
pint.Unit = ureg.Unit
pint.Measurement = ureg.Measurement
pint.Context = ureg.Context

# FIXME: delay loading of pint_pandas until after we've initialized ourselves
from pint_pandas import PintType
PintType.ureg = ureg
