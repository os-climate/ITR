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

pint.Quantity = ureg.Quantity
pint.Measurement = ureg.Measurement

# FIXME: delay loading of pint_pandas until after we've initialized ourselves
from pint_pandas import PintType
PintType.ureg = unit_registry
