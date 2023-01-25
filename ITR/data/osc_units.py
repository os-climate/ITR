"""
This module handles initialization of pint functionality
"""

import numpy as np
import pandas as pd
from pint import get_application_registry, Context, Quantity, DimensionalityError
import ITR

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

ureg.define("Alloys = [alloys]")
ureg.define("Al = [aluminum] = Aluminum")
ureg.define("aluminium = Al")
ureg.define("Biofuel = [biofuel]")
ureg.define("Cement = [cement]")
ureg.define("Coal = [coal]")
ureg.define("Cu = [copper] = Copper")
ureg.define("Paper = [paper] = Pulp")
ureg.define("Paperboard = Paper")
ureg.define("Petrochemicals = [petrochemicals]")
ureg.define("Petroleum = [petroleum]")
ureg.define("Fe = [iron] = Steel")

# For reports that use 10,000 t instead of 1e3 or 1e6
ureg.define('myria- = 10000')

# These are for later
ureg.define('fraction = [] = frac')
ureg.define('percent = 1e-2 frac = pct = percentage')
ureg.define('ppm = 1e-6 fraction')

# USD are the reserve currency of the ITR tool
ureg.define("USD = [currency] = $")
for currency_symbol, currency_abbrev in ITR.data.currency_dict.items():
    ureg.define(f"{currency_abbrev} = nan USD = {currency_symbol}")
# Currencies that don't have symbols are added one by one
ureg.define("CHF = nan USD")
ureg.define("MXN = nan USD") # $ abbreviation is ambiguous

ureg.define("btu = Btu")
ureg.define("mmbtu = 1e6 btu")
# ureg.define("boe = 5.712 GJ")
ureg.define("boe = 6.1178632 GJ = BoE")
ureg.define("mboe = 1e3 boe")
ureg.define("mmboe = 1e6 boe")
ureg.define("MMbbl = 1e6 bbl")

ureg.define("scf = ft**3")
ureg.define("mscf = 1000 scf = Mscf")
ureg.define("mmscf = 1000000 scf = MMscf")
ureg.define("bscf = 1000000000 scf")
ureg.define("bcm = 1000000000 m**3")
# ureg.define("bcm = 38.2 PJ") # Also bcm = 17 Mt CO2e, but that wrecks CO2e emissions intensities (which would devolve to dimensionless numbers)

# A reminder for those who come looking for transformation rules: by default,
# Pint's normal behavior is to do all manner of conversions...when units have the same dimensionality
# i.e., m/s, feet/minute, furlongs/fortnight.  But when dimensionality is different
# (such as converting a volume of CH4 to a mass equivalent of CO2), it is the
# caller's responsibility to do the conversion (so `src OP dst` is coded as `src.to(dst_units) OP dst`).

# Source: https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator-calculations-and-references
# 0.1 mmbtu equals one therm (EIA 2019).
# The average carbon coefficient of pipeline natural gas burned in 2018 is 14.43 kg carbon per mmbtu (EPA 2021).
# The fraction oxidized to CO2 is assumed to be 100 percent (IPCC 2006).
# 0.1 mmbtu/1 therm × 14.43 kg C/mmbtu × 44 kg CO2/12 kg C × 1 metric ton/1,000 kg = 0.005291 metric tons CO2/therm
# 0.1 mmbtu/1 therm × 14.43 kg C/mmbtu × 16 kg CH4/12 kg C × 1 metric ton/1,000 kg = 0.001924 metric tons CO2/therm
# 0.005291 metric tons CO2/therm x 10.37 therms/Mcf = 0.05487 metric tons CO2/Mcf
# 0.001924 metric tons CH4/therm x 10.37 therms/Mcf = 0.01995 metric tons CH4/Mcf

# One thousand cubic feet (Mcf) of natural gas equals 1.037 MMBtu, or 10.37 therms (EIA: https://www.eia.gov/tools/faqs/faq.php?id=45&t=8)
# Note that natural gas has a higher specific energy than CH4, which we use as the name for natural gas.  

NG_DENS = 0.7046 * ureg('kg CH4/(m**3 CH4)') # 0.657 
NG_SE = 54.84 * ureg('MJ/(kg CH4)')                  # specific energy (energy per mass); range is 50-55
ng = Context('ngas')
ng.add_transformation('[volume] CH4', '[mass] CH4', lambda ureg, x: x * NG_DENS)
ng.add_transformation('[mass] CH4', '[volume] CH4', lambda ureg, x: x / NG_DENS)
ng.add_transformation('[volume] CH4 ', '[energy]', lambda ureg, x: x * NG_DENS * NG_SE)
ng.add_transformation('[energy]', '[volume] CH4', lambda ureg, x: x / (NG_DENS * NG_SE))
ng.add_transformation('[carbon] * [length] * [methane] * [time] ** 2', '[carbon] * [mass]', lambda ureg, x: x * NG_DENS * NG_SE)
ng.add_transformation('[carbon] * [mass] / [volume] / [methane]', '[carbon] * [mass] / [energy]', lambda ureg, x: x / (NG_DENS * NG_SE))
ng.add_transformation('Mscf CH4', 'kg CO2e', lambda ureg, x: x * ureg('54.87 kg CO2e / (Mscf CH4)'))
ng.add_transformation('g CH4', 'g CO2e', lambda ureg, x: x * ureg('44 g CO2e / (16 g CH4)'))
ureg.add_context(ng)
ureg.enable_contexts('ngas')

def time_dimension(unit, exp):
    return ureg(unit).is_compatible_with("s") # and exp == -1


def convert_to_annual(x, errors='ignore'):
    """
    For a quantity X that has units of [time], reduce the time dimension, leaving an "implictly annual" metric.
    If X has no time dimension, or if it cannot be reduced to zero in a single step, raise a DimensionalityError.
    If ERRORS=='ignore', allow time dimension to be reduced one step towards zero rather than only to zero.
    Returns the reduced quantity, or the original quantity if reduction would result in an error being raised.
    """
    import pint
    unit_ct = pint.util.to_units_container(x)
    # print(unit_ct)
    # <UnitsContainer({'day': -1, 'kilogram': 1})>
    x_implied_annual = x
    try:
        time_unit, exp = next((pint.Unit(unit), exp) for unit, exp in unit_ct.items() if time_dimension(unit, exp))
        time_unit_str = str(time_unit)
        if exp == -1:
            x_implied_annual = Q_(x * ureg('a').to(time_unit), unit_ct.remove([time_unit_str]))
        elif exp == 1:
            x_implied_annual = Q_(x / ureg(time_unit_str).to('a'), unit_ct.remove([time_unit_str]))
        else:
            if errors=='ignore':
                if exp < 0:
                    x_implied_annual = Q_(x * ureg('a').to(time_unit), unit_ct.remove([time_unit_str]).add(time_unit_str, exp+1))
                else:
                    x_implied_annual = Q_(x / ureg(time_unit_str).to('a'), unit_ct.remove([time_unit_str]).add(time_unit_str, exp-1))
            raise DimensionalityError (x, '', extra_msg=f"; dimensionality must contain [time] or 1/[time], not [time]**{exp}")
    except StopIteration:
        if errors!='ignore':
            raise DimensionalityError (x, '', extra_msg=f"; dimensionality must contain [time] or 1/[time]")
    return x_implied_annual

def dimension_as(x, dim_unit):
    import pint
    unit_ct = pint.util.to_units_container(x)
    # print(unit_ct)
    # <UnitsContainer({'day': -1, 'kilogram': 1})>
    try:
        unit, exp = next((pint.Unit(unit), exp) for unit, exp in unit_ct.items() if ureg(unit).is_compatible_with(dim_unit))
        orig_dim_unit = ureg(str(unit))
        return (x * orig_dim_unit.to(dim_unit) / orig_dim_unit).to_reduced_units()
    except StopIteration:
        raise DimensionalityError (x, dim_unit, extra_msg=f"; no compatible dimension not found")

oil = Context('oil')
oil.add_transformation('[carbon] * [mass] ** 2 / [length] / [time] ** 2', '[carbon] * [mass]',
                      lambda ureg, x: x * ureg('bbl/boe').to_reduced_units())
oil.add_transformation('[carbon] * [mass] ** 2 / [length] / [time] ** 3', '[carbon] * [mass]',
                      lambda ureg, x: convert_to_annual(x) * ureg('bbl/boe').to_reduced_units())
# oil.add_transformation('boe', 'kg CO2e', lambda ureg, x: x * ureg('431.87 kg CO2e / boe')
oil.add_transformation('bbl', 'boe', lambda ureg, x: x * ureg('boe') / ureg('bbl'))
oil.add_transformation('boe', 'bbl', lambda ureg, x: x * ureg('bbl') / ureg('boe'))
oil.add_transformation('[carbon] * [mass] / [time]', '[carbon] * [mass]', lambda ureg, x: convert_to_annual(x))
# Converting intensity t CO2/bbl -> t CO2/boe
oil.add_transformation('[carbon] * [mass] / [length] ** 3', '[carbon] * [time] ** 2 / [length] ** 2', lambda ureg, x: (x * ureg('bbl/boe')).to_reduced_units())
oil.add_transformation('[carbon] * [time] ** 2 / [length] ** 2', '[carbon] * [mass] / [length] ** 3', lambda ureg, x: (x * ureg('boe/bbl')).to_reduced_units())
ureg.add_context(oil)
ureg.enable_contexts('oil')

# Transportation activity

ureg.define("vehicle = [vehicle] = v")
ureg.define("passenger = [passenger] = p = pass")
ureg.define("vkm = vehicle * kilometer")
ureg.define("pkm = passenger * kilometer")
ureg.define("tkm = tonne * kilometer")

ureg.define('hundred = 1e2 = Hundreds')
ureg.define('thousand = 1e3 = Thousands')
ureg.define('million = 1e6 = Millions')
ureg.define('billion = 1e9 = Billions')
ureg.define('trillion = 1e12 = Trillions')
ureg.define('quadrillion = 1e15')

# Backward compatibility
ureg.define("Fe_ton = t Steel")



# These are for later still
# ureg.define("HFC = [ HFC_emissions ]")
# ureg.define("PFC = [ PFC_emissions ]")
# ureg.define("mercury = Hg = Mercury")
# ureg.define("mercure = Hg = Mercury")
# ureg.define("PM10 = [ PM10_emissions ]")

# List of all the production units we know
_production_units = [ "Wh", "pkm", "tkm", "bcm CH4", "bbl", "boe", 't Alloys', "t Aluminum", "t Cement", "t Coal", "t Copper",
                      "t Paper", "t Steel", "USD", "m**2", 't Biofuel', 't Petrochemicals', 't Petroleum' ]
_ei_units = [f"t CO2/({pu})" if ' ' in pu else f"t CO2/{pu}" for pu in _production_units]

class ProductionMetric(str):
    """
    Valid production metrics accepted by ITR tool
    """

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            examples=_production_units,
        )

    @classmethod
    def validate(cls, units):
        if not isinstance(units, str):
            raise TypeError('string required')
        qty = ureg(units)
        for pu in _production_units:
            if qty.is_compatible_with(pu):
                return cls(units)
            qty_as_annual = convert_to_annual(qty, errors='ignore')
            if qty_as_annual.is_compatible_with(pu):
                return cls(str(qty_as_annual.u))
        raise ValueError(f"{qty} not relateable to {_production_units}")

    def __repr__(self):
        return f"ProductionMetric({super().__repr__()})"

class EmissionsMetric(str):
    """
    Valid production metrics accepted by ITR tool
    """

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            examples=['g CO2', 'kg CO2', 't CO2', 'Mt CO2'],
        )

    @classmethod
    def validate(cls, units):
        if not isinstance(units, str):
            raise TypeError('string required')
        qty = ureg(units)
        if qty.is_compatible_with('t CO2'):
            return cls(units)
        raise ValueError(f"{units} not relateable to 't CO2'")

    def __repr__(self):
        return f'EmissionsMetric({super().__repr__()})'

class EI_Metric(str):
    """
    Valid production metrics accepted by ITR tool
    """

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            examples=['g CO2/pkm', 'kg CO2/tkm', 't CO2/(t Steel)', 'Mt CO2/TWh'],
        )

    @classmethod
    def validate(cls, units):
        if not isinstance(units, str):
            raise TypeError('string required')
        qty = ureg(units)
        for ei_u in _ei_units:
            if qty.is_compatible_with(ei_u):
                return cls(units)
        raise ValueError(f"{units} not relateable to {_ei_units}")

    def __repr__(self):
        return f'EI_Metric({super().__repr__()})'

class BenchmarkMetric(str):
    """
    Valid benchmark metrics accepted by ITR tool
    """

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            examples=['dimensionless', 'g CO2/pkm', 'kg CO2/tkm', 't CO2/(t Steel)', 'Mt CO2/TWh'],
        )

    @classmethod
    def validate(cls, units):
        if not isinstance(units, str):
            raise TypeError('string required')
        if units=='dimensionless':
            return cls(units)
        qty = ureg(units)
        for ei_u in _ei_units:
            if qty.is_compatible_with(ei_u):
                return cls(units)
        raise ValueError(f"{units} not relateable to 'dimensionless' or {_ei_units}")

    def __repr__(self):
        return f'BenchmarkMetric({super().__repr__()})'

# Borrowed from https://github.com/hgrecco/pint/issues/1166
registry = ureg

schema_extra = dict(definitions=[
    dict(
        Quantity=dict(type="string"),
        EmissionsQuantity=dict(type="string"),
        ProductionQuantity=dict(type="string"),
        EI_Quantity=dict(type="string"),
        BenchmarkQuantity=dict(type="string"),
    )
])

# Dimensionality is something like `[mass]`, not `t CO2`.  And this returns a TYPE, not a Quantity

def quantity(dimensionality: str) -> type:
    """A method for making a pydantic compliant Pint quantity field type."""

    try:
        if isinstance(dimensionality, Quantity):
            registry.get_dimensionality(dimensionality)
    except KeyError:
        raise ValueError(f"{dimensionality} is not a valid dimensionality in pint!")

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if isinstance(value, str):
            try:
                q = Q_(value)
            except ValueError:
                # breakpoint()
                raise ValueError(f"cannot convert '{value}' to quantity")
            quantity = q
        elif isinstance(value, Quantity):
            quantity = value
        else:
            raise TypeError (f"quantity takes either a Q_ value or a string fully expressing a quantified value; got {value}")
        if quantity.is_compatible_with(dimensionality):
            return quantity
        assert quantity.check(cls.dimensionality), f"Dimensionality of {quantity} incompatible with {cls.dimensionality}"
        return quantity

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            {"$ref": "#/definitions/Quantity"}
        )
    
    return type(
        "Quantity",
        (Quantity,),
        dict(
            __get_validators__=__get_validators__,
            __modify_schema__=__modify_schema__,
            dimensionality=dimensionality,
            validate=validate,
        ),
    )
# end of borrowing


class EmissionsQuantity(Quantity):
    """A method for making a pydantic compliant Pint emissions quantity."""

    def __new__(cls, value, units=None):
        # Re-used the instance we are passed.  Do we need to copy?
        return value

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            # simplified regex here for brevity, see the wikipedia link above
            dimensionaltiy='t CO2',
            # some example postcodes
            examples=['g CO2', 'kg CO2', 't CO2', 'Mt CO2'],
        )

    @classmethod
    def validate(cls, quantity):
        if quantity is None:
            raise ValueError
        if isinstance(quantity, str):
            v, u = quantity.split(' ', 1)
            try:
                q = Q_(float(v), u)
            except ValueError:
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        if not isinstance(quantity, Quantity):
            raise TypeError(f"pint.Quantity required ({quantity}, type = {type(quantity)})")
        if quantity.is_compatible_with('t CO2'):
            return quantity
        raise DimensionalityError (quantity, 't CO2', dim1='', dim2='', extra_msg=f"Dimensionality must be compatible with 't CO2'")

    def __repr__(self):
        return f'EmissionsQuantity({super().__repr__()})'

    class Config:
        validate_assignment = True
        schema_extra = schema_extra
        json_encoders = {
            Quantity: str,
        }


class ProductionQuantity(str):
    """A method for making a pydantic compliant Pint production quantity."""

    def __new__(cls, value, units=None):
        # Re-used the instance we are passed.  Do we need to copy?
        return value

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            # simplified regex here for brevity, see the wikipedia link above
            dimensionaltiy='[production_units]',
            # some example postcodes
            examples=_production_units,
        )

    @classmethod
    def validate(cls, quantity):
        if quantity is None:
            raise ValueError
        if isinstance(quantity, str):
            v, u = quantity.split(' ', 1)
            try:
                q = Q_(float(v), u)
            except ValueError:
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        if not isinstance(quantity, Quantity):
            raise TypeError('pint.Quantity required')
        for pu in _production_units:
            if quantity.is_compatible_with(pu):
                return quantity
            quantity_as_annual = convert_to_annual(quantity, errors='ignore')
            if quantity_as_annual.is_compatible_with(pu):
                return quantity_as_annual
        raise DimensionalityError (quantity, str(_production_units), dim1='', dim2='', extra_msg=f"Dimensionality must be compatible with [{_production_units}]")

    def __repr__(self):
        return f'ProductionQuantity({super().__repr__()})'

    class Config:
        validate_assignment = True
        schema_extra = schema_extra
        json_encoders = {
            Quantity: str,
        }


class EI_Quantity(str):
    """A method for making a pydantic compliant Pint Emissions Intensity quantity."""

    def __new__(cls, value, units=None):
        # Re-used the instance we are passed.  Do we need to copy?
        return value

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            # simplified regex here for brevity, see the wikipedia link above
            dimensionaltiy='ei units',
            # some example postcodes
            examples=_ei_units,
        )

    @classmethod
    def validate(cls, quantity):
        if quantity is None:
            raise ValueError
        if isinstance(quantity, str):
            v, u = quantity.split(' ', 1)
            try:
                q = Q_(float(v), u)
            except ValueError:
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        if not isinstance(quantity, Quantity):
            raise TypeError('pint.Quantity required')
        for ei_u in _ei_units:
            if quantity.is_compatible_with(ei_u):
                return quantity
            quantity_as_annual = convert_to_annual(quantity, errors='ignore')
            if quantity_as_annual.is_compatible_with(ei_u):
                return quantity_as_annual
        raise DimensionalityError (quantity, str(_ei_units), dim1='', dim2='', extra_msg=f"Dimensionality must be compatible with [{_ei_units}]")

    def __repr__(self):
        return f'EI_Quantity({super().__repr__()})'

    class Config:
        validate_assignment = True
        schema_extra = schema_extra
        json_encoders = {
            Quantity: str,
        }


class BenchmarkQuantity(str):
    """A method for making a pydantic compliant Pint Benchmark quantity (which includes dimensionless production growth)."""

    def __new__(cls, value, units=None):
        # Re-used the instance we are passed.  Do we need to copy?
        return value

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            # simplified regex here for brevity, see the wikipedia link above
            dimensionaltiy='ei units',
            # some example postcodes
            examples=_ei_units,
        )

    @classmethod
    def validate(cls, quantity):
        if isinstance(quantity, str):
            v, u = quantity.split(' ', 1)
            try:
                q = Q_(float(v), u)
            except ValueError:
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        if not isinstance(quantity, Quantity):
            raise TypeError('pint.Quantity required')
        if str(quantity.u) == 'dimensionless':
            return quantity
        for ei_u in _ei_units:
            if quantity.is_compatible_with(ei_u):
                return quantity
        raise DimensionalityError (quantity, str(_ei_units), dim1='', dim2='', extra_msg=f"Dimensionality must be 'dimensionless' or compatible with [{_ei_units}]")

    def __repr__(self):
        return f'BenchmarkQuantity({super().__repr__()})'

    class Config:
        validate_assignment = True
        schema_extra = schema_extra
        json_encoders = {
            Quantity: str,
        }


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
    # NA_VALUEs are true NaNs, missing units
    na_values = series.isna()
    units = series[~na_values].map(lambda x: x.u if isinstance(x, Quantity) else None)
    unit_first_idx = units.first_valid_index()
    if unit_first_idx is None:
        if errors != 'ignore':
            raise ValueError(f"No value units in series: {series}")
        return series
    # Arbitrarily pick first of the most popular units, as promised
    unit = units.mode()[0]
    if inplace:
        new_series = series
    else:
        new_series = series.copy()
    if name:
        new_series.name = name
    na_index = na_values[na_values].index
    new_series.loc[na_index] = pd.Series(Q_(np.nan, unit), index=na_index)
    return new_series.astype(f"pint[{unit}]")

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
    # When DF.COLUMNS is a MultiIndex, the naive column-by-column construction replaces MultiIndex values
    # with the anonymous tuple of the MultiIndex and DF.COLUMNS becomes just an Index of tuples.
    # We need to restore the MultiIndex or lose information.
    new_df.columns = df.columns
    return new_df
