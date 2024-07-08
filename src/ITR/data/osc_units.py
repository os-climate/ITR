"""This module handles initialization of pint functionality"""

import re
from typing import Annotated, Any, List, Union

import pandas as pd
import pint
from pint import Context, DimensionalityError
from pydantic import GetJsonSchemaHandler
from pydantic.functional_validators import AfterValidator, BeforeValidator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from typing_extensions import TypeAlias

import ITR

from ..data import PA_, Q_, PintType, ureg

Quantity: TypeAlias = ureg.Quantity

ureg.define("CO2e = CO2 = CO2eq = CO2_eq")
# openscm_units does this for all gas species...we just have to keep up.
ureg.define("tCO2e = t CO2e")

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
ureg.define("myria- = 10000")

# These are for later
ureg.define("fraction = [] = frac")
ureg.define("percent = 1e-2 frac = pct = percentage")
ureg.define("ppm = 1e-6 fraction")

# By default, USD are the reserve currency of the ITR tool.  But data template can change that
base_currency_unit = "USD"
ureg.define("USD = [currency] = $")
for currency_symbol, currency_abbrev in ITR.data.currency_dict.items():
    ureg.define(f"{currency_abbrev} = nan USD = {currency_symbol}")
# Currencies that don't have symbols are added one by one
ureg.define("CHF = nan USD")
ureg.define("MXN = nan USD")  # $ abbreviation is ambiguous

fx_ctx = Context("FX")

ureg.define("btu = Btu")
ureg.define("mmbtu = 1e6 btu")
# ureg.define("boe = 5.712 GJ")
ureg.define("boe = 6.1178632 GJ = BoE")
ureg.define("mboe = 1e3 boe")
ureg.define("mmboe = 1e6 boe")
ureg.define("Mbbl = 1e3 bbl")
ureg.define("MMbbl = 1e6 bbl")

ureg.define("scf = ft**3")
ureg.define("mscf = 1000 scf = Mscf")
ureg.define("mmscf = 1000000 scf = MMscf")
ureg.define("bscf = 1000000000 scf = Bscf")
ureg.define("MMMscf = 1000000000 scf")
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

NG_DENS = 0.7046 * ureg("kg CH4/(m**3 CH4)")  # 0.657
NG_SE = 54.84 * ureg("MJ/(kg CH4)")  # specific energy (energy per mass); range is 50-55
ng = Context("ngas")
ng.add_transformation("[volume] CH4", "[mass] CH4", lambda ureg, x: x * NG_DENS)
ng.add_transformation("[mass] CH4", "[volume] CH4", lambda ureg, x: x / NG_DENS)
ng.add_transformation("[volume] CH4 ", "[energy]", lambda ureg, x: x * NG_DENS * NG_SE)
ng.add_transformation("[energy]", "[volume] CH4", lambda ureg, x: x / (NG_DENS * NG_SE))
ng.add_transformation(
    "[carbon] * [length] * [methane] * [time] ** 2",
    "[carbon] * [mass]",
    lambda ureg, x: x * NG_DENS * NG_SE,
)
ng.add_transformation(
    "[carbon] * [mass] / [volume] / [methane]",
    "[carbon] * [mass] / [energy]",
    lambda ureg, x: x / (NG_DENS * NG_SE),
)
ng.add_transformation(
    "[carbon] * [time] ** 2 / [length] ** 2",
    "[carbon] * [mass] / [length] ** 3 / [methane]",
    lambda ureg, x: x * NG_DENS * NG_SE,
)
ng.add_transformation(
    "[mass] / [length] / [methane] / [time] ** 2",
    "[]",
    lambda ureg, x: x / (NG_DENS * NG_SE),
)

ng.add_transformation(
    "Mscf CH4", "kg CO2e", lambda ureg, x: x * ureg("54.87 kg CO2e / (Mscf CH4)")
)
ng.add_transformation(
    "g CH4", "g CO2e", lambda ureg, x: x * ureg("44 g CO2e / (16 g CH4)")
)
ureg.add_context(ng)

COAL_SE = 29307.6 * ureg(
    "MJ/(t Coal)"
)  # specific energy (energy per mass); range is 50-55
coal = Context("coal")
coal.add_transformation("[mass] Coal", "[energy]", lambda ureg, x: x * COAL_SE)
coal.add_transformation("[energy]", "[mass] Coal", lambda ureg, x: x / COAL_SE)
coal.add_transformation(
    "g Coal", "g CO2e", lambda ureg, x: x * ureg("1.992 g CO2e / (1 g Coal)")
)
ureg.add_context(coal)

ureg.enable_contexts("ngas", "coal")


# from https://github.com/hgrecco/pint/discussions/1697
def direct_conversions(ureg, unit) -> List[str]:
    """Return a LIST of unit names that Pint can convert implicitly from/to UNIT.
    This does not include the list of additional unit names that can be explicitly
    converted by using the `Quantity.to` method.
    """

    def unit_dimensionality(ureg, name):
        unit = getattr(ureg, name, None)

        if unit is None or not isinstance(unit, pint.Unit):
            return {}

        return unit.dimensionality

    if isinstance(unit, str):
        unit = ureg.parse_units(unit)

    return [
        name
        for name in ureg
        if name != "%" and unit.dimensionality == unit_dimensionality(ureg, name)
    ]


# conversions = direct_conversions(ureg, "m / s ** 2")
# {name: getattr(ureg, name).dimensionality for name in conversions}


def time_dimension(unit, exp) -> bool:
    """True if UNIT can be converted to something related only to time."""
    return ureg(unit).is_compatible_with("s")  # and exp == -1


def convert_to_annual(x, errors="ignore"):
    """For a quantity X that has units of [time], reduce the time dimension, leaving an "implictly annual" metric.
    If X has no time dimension, or if it cannot be reduced to zero in a single step, raise a DimensionalityError.
    If ERRORS=='ignore', allow time dimension to be reduced one step towards zero rather than only to zero.
    Returns the reduced quantity, or the original quantity if reduction would result in an error being raised.
    """
    unit_ct = pint.util.to_units_container(x)
    # print(unit_ct)
    # <UnitsContainer({'day': -1, 'kilogram': 1})>
    x_implied_annual = x
    try:
        time_unit, exp = next(
            (pint.Unit(unit), exp)
            for unit, exp in unit_ct.items()
            if time_dimension(unit, exp)
        )
        time_unit_str = str(time_unit)
        if exp == -1:
            x_implied_annual = Q_(
                x * ureg("a").to(time_unit), unit_ct.remove([time_unit_str])
            )
        elif exp == 1:
            x_implied_annual = Q_(
                x / ureg(time_unit_str).to("a"), unit_ct.remove([time_unit_str])
            )
        else:
            if errors == "ignore":
                if exp < 0:
                    x_implied_annual = Q_(
                        x * ureg("a").to(time_unit),
                        unit_ct.remove([time_unit_str]).add(time_unit_str, exp + 1),
                    )
                else:
                    x_implied_annual = Q_(
                        x / ureg(time_unit_str).to("a"),
                        unit_ct.remove([time_unit_str]).add(time_unit_str, exp - 1),
                    )
            raise DimensionalityError(
                x,
                "",
                extra_msg=f"; dimensionality must contain [time] or 1/[time], not [time]**{exp}",
            )
    except StopIteration:
        if errors != "ignore":
            raise DimensionalityError(
                x, "", extra_msg="; dimensionality must contain [time] or 1/[time]"
            )
    return x_implied_annual


def dimension_as(x, dim_unit):
    unit_ct = pint.util.to_units_container(x)
    # print(unit_ct)
    # <UnitsContainer({'day': -1, 'kilogram': 1})>
    try:
        unit, exp = next(
            (pint.Unit(unit), exp)
            for unit, exp in unit_ct.items()
            if ureg(unit).is_compatible_with(dim_unit)
        )
        orig_dim_unit = ureg(str(unit))
        return (x * orig_dim_unit.to(dim_unit) / orig_dim_unit).to_reduced_units()
    except StopIteration:
        raise DimensionalityError(
            x, dim_unit, extra_msg="; no compatible dimension not found"
        )


def align_production_to_bm(prod_series: pd.Series, ei_bm: pd.Series) -> pd.Series:
    """A timeseries of production unit values can be aligned with a timeseries of Emissions Intensity (EI)
    metrics that uses different units of production.  For example, the production timeseries might be
    `bbl` (`Blue Barrels of Oil`) but the EI might be `t CO2e/GJ` (`CO2e * metric_ton / gigajoule`).
    By converting the production series to gigajoules up front, there are no complex conversions
    needed later (such as trying to convert `t CO2e * metric_ton / bbl` to
    `CO2e * metric_ton / gigajoule`, which is not straightfowrard, as the former is
    `[mass] / [length]**3` whereas the latter is `[seconds] ** 2 / [length] **2`.
    """
    if ureg(f"t CO2e/({prod_series.iloc[0].units})") == ei_bm.iloc[0]:
        return prod_series
    # Convert the units of production into the denominator of the EI units
    ei_units = str(ei_bm.iloc[0].units)
    (ei_unit_top, ei_unit_bottom) = ei_units.split("/", 1)
    if "/" in ei_unit_bottom:
        # Fix reciprocals: t CO2e / CH4 / bcm -> t CO2e / (CH4 * bcm)
        (bottom_unit_num, bottom_unit_denom) = ei_unit_bottom.split("/", 1)
        ei_unit_bottom = f"{bottom_unit_num} {bottom_unit_denom}"
    # We might need to add mass dimension back in if it was simplified out (t CO2e / t Fe, for example)
    if "[mass]" not in ureg.parse_units(ei_unit_top).dimensionality:
        try:
            mass_unit = [
                unit
                for unit, exp in pint.util.to_units_container(
                    prod_series.iloc[0].to_base_units()
                ).items()
                if exp == 1 and ureg(unit).is_compatible_with("kg")
            ][0]
            ei_unit_bottom = f"{mass_unit} {ei_unit_bottom}"
        except IndexError:
            # If no mass term in prod_series, likely a dimensional mismatch between prod_series and ei_unit_bottom
            raise DimensionalityError(
                prod_series.iloc[0],
                "",
                dim1=str(prod_series.dtype.units),
                dim2=ei_unit_bottom,
                extra_msg="cannot align units",
            )
    return asPintSeries(prod_series).pint.to(ei_unit_bottom)


oil = Context("oil")
oil.add_transformation(
    "[carbon] * [mass] ** 2 / [length] / [time] ** 2",
    "[carbon] * [mass]",
    lambda ureg, x: x * ureg("bbl/boe").to_reduced_units(),
)
oil.add_transformation(
    "[carbon] * [mass] ** 2 / [length] / [time] ** 3",
    "[carbon] * [mass]",
    lambda ureg, x: convert_to_annual(x) * ureg("bbl/boe").to_reduced_units(),
)
# oil.add_transformation('boe', 'kg CO2e', lambda ureg, x: x * ureg('431.87 kg CO2e / boe')
oil.add_transformation("bbl", "boe", lambda ureg, x: x * ureg("boe") / ureg("bbl"))
oil.add_transformation("boe", "bbl", lambda ureg, x: x * ureg("bbl") / ureg("boe"))
oil.add_transformation(
    "[carbon] * [mass] / [time]",
    "[carbon] * [mass]",
    lambda ureg, x: convert_to_annual(x),
)
# Converting intensity t CO2/bbl -> t CO2/boe
oil.add_transformation(
    "[carbon] * [mass] / [length] ** 3",
    "[carbon] * [time] ** 2 / [length] ** 2",
    lambda ureg, x: (x * ureg("bbl/boe")).to_reduced_units(),
)
oil.add_transformation(
    "[carbon] * [time] ** 2 / [length] ** 2",
    "[carbon] * [mass] / [length] ** 3",
    lambda ureg, x: (x * ureg("boe/bbl")).to_reduced_units(),
)
ureg.add_context(oil)
ureg.enable_contexts("oil")

# Transportation activity

ureg.define("vehicle = [vehicle] = v")
ureg.define("passenger = [passenger] = p = pass")
ureg.define("vkm = vehicle * kilometer")
ureg.define("pkm = passenger * kilometer")
ureg.define("tkm = tonne * kilometer")

ureg.define("hundred = 1e2 = Hundreds")
ureg.define("thousand = 1e3 = Thousands")
ureg.define("million = 1e6 = Millions")
ureg.define("billion = 1e9 = Billions")
ureg.define("trillion = 1e12 = Trillions")
ureg.define("quadrillion = 1e15")

# Backward compatibility
ureg.define("Fe_ton = t Steel")


# These are for later still
# ureg.define("HFC = [ HFC_emissions ]")
# ureg.define("PFC = [ PFC_emissions ]")
# ureg.define("mercury = Hg = Mercury")
# ureg.define("mercure = Hg = Mercury")
# ureg.define("PM10 = [ PM10_emissions ]")

# List of all the production units we know
_production_units = [
    "Wh",
    "pkm",
    "tkm",
    "bcm CH4",
    "bbl",
    "boe",
    "t Alloys",
    "t Aluminum",
    "t Cement",
    "t Coal",
    "t Copper",
    "t Paper",
    "t Steel",
    "USD",
    "m**2",
    "t Biofuel",
    "t Petrochemicals",
    "t Petroleum",
]
_ei_units = [
    f"t CO2e/({pu})" if " " in pu else f"t CO2e/{pu}" for pu in _production_units
]


def check_ProductionMetric(units: str) -> str:
    qty = ureg(units)
    for pu in _production_units:
        if qty.is_compatible_with(pu):
            return units
        qty_as_annual = convert_to_annual(qty, errors="ignore")
        if qty_as_annual.is_compatible_with(pu):
            return str(qty_as_annual.u)
    raise ValueError(f"{qty} not relateable to {_production_units}")


ProductionMetric = Annotated[str, AfterValidator(check_ProductionMetric)]


def check_EmissionsMetric(units: str) -> str:
    qty = ureg(units)
    if qty.is_compatible_with("t CO2"):
        return units
    raise ValueError(f"{units} not relateable to 't CO2'")


EmissionsMetric = Annotated[str, AfterValidator(check_EmissionsMetric)]


def check_EI_Metric(units: str) -> str:
    qty = ureg(units)
    for ei_u in _ei_units:
        if qty.is_compatible_with(ei_u):
            return units
    raise ValueError(f"{units} not relateable to {_ei_units}")


EI_Metric = Annotated[str, AfterValidator(check_EI_Metric)]


def check_BenchmarkMetric(units: str) -> str:
    if units == "dimensionless":
        return units
    qty = ureg(units)
    for ei_u in _ei_units:
        if qty.is_compatible_with(ei_u):
            return units
    raise ValueError(f"{units} not relateable to 'dimensionless' or {_ei_units}")


BenchmarkMetric = Annotated[str, AfterValidator(check_BenchmarkMetric)]


def to_Quantity(quantity: Union[Quantity, str]) -> Quantity:
    if isinstance(quantity, str):
        try:
            v, u = quantity.split(" ", 1)
            if v == "nan" or "." in v or "e" in v:
                quantity = Q_(float(v), u)
            else:
                quantity = Q_(int(v), u)
        except ValueError:
            return ureg(quantity)
    elif not isinstance(quantity, Quantity):  # type: ignore
        raise ValueError(f"{quantity} is not a Quantity")
    return quantity


def check_EmissionsQuantity(quantity: Quantity) -> Quantity:
    if quantity.is_compatible_with("t CO2"):
        return quantity
    raise DimensionalityError(
        quantity,
        "t CO2",
        dim1="",
        dim2="",
        extra_msg="Dimensionality must be compatible with 't CO2'",
    )


EmissionsQuantity = Annotated[
    Quantity, BeforeValidator(to_Quantity), AfterValidator(check_EmissionsQuantity)
]


def check_ProductionQuantity(quantity: Quantity) -> Quantity:
    for pu in _production_units:
        if quantity.is_compatible_with(pu):
            return quantity
    try:
        quantity_as_annual = convert_to_annual(quantity, errors="ignore")
        for pu in _production_units:
            if quantity_as_annual.is_compatible_with(pu):
                return quantity_as_annual
    except DimensionalityError:
        pass
    raise DimensionalityError(
        quantity,
        str(_production_units),
        dim1="",
        dim2="",
        extra_msg=f"Dimensionality must be compatible with [{_production_units}]",
    )


ProductionQuantity = Annotated[
    Quantity, BeforeValidator(to_Quantity), AfterValidator(check_ProductionQuantity)
]


def check_EI_Quantity(quantity: Quantity) -> Quantity:
    for ei_u in _ei_units:
        if quantity.is_compatible_with(ei_u):
            return quantity
    try:
        quantity_as_annual = convert_to_annual(quantity, errors="raise")
        for ei_u in _ei_units:
            if quantity_as_annual.is_compatible_with(ei_u):
                return quantity_as_annual
    except DimensionalityError:
        pass
    raise DimensionalityError(
        quantity,
        str(_ei_units),
        dim1="",
        dim2="",
        extra_msg=f"Dimensionality must be compatible with [{_ei_units}]",
    )


EI_Quantity = Annotated[
    Quantity, BeforeValidator(to_Quantity), AfterValidator(check_EI_Quantity)
]


def check_BenchmarkQuantity(quantity: Quantity) -> Quantity:
    if quantity.u.dimensionless:
        return quantity
    for ei_u in _ei_units:
        if quantity.is_compatible_with(ei_u):
            return quantity
    raise DimensionalityError(
        quantity,
        str(_ei_units),
        dim1="",
        dim2="",
        extra_msg=f"Dimensionality must be 'dimensionless' or compatible with [{_ei_units}]",
    )


BenchmarkQuantity = Annotated[
    Quantity, BeforeValidator(to_Quantity), AfterValidator(check_BenchmarkQuantity)
]


def check_MonetaryQuantity(quantity: Quantity) -> Quantity:
    try:
        if quantity.is_compatible_with("USD"):
            return quantity
    except RecursionError:
        # breakpoint()
        raise
    for currency in ITR.data.currency_dict.values():
        if quantity.is_compatible_with(currency):
            return quantity
    raise DimensionalityError(
        quantity,
        "USD",
        dim1="",
        dim2="",
        extra_msg=f"Dimensionality must be 'dimensionless' or compatible with [{ITR.data.currency_dict.values()}]",
    )


MonetaryQuantity = Annotated[
    Quantity, BeforeValidator(to_Quantity), AfterValidator(check_MonetaryQuantity)
]


def check_delta_degC_Quantity(quantity: Quantity) -> Quantity:
    try:
        if quantity.is_compatible_with("delta_degC"):
            return quantity
    except RecursionError:
        # breakpoint()
        raise
    raise DimensionalityError(
        quantity,
        "delta_degC",
        dim1="",
        dim2="",
        extra_msg="Dimensionality must be compatible with delta_degC",
    )


delta_degC_Quantity = Annotated[
    Quantity, BeforeValidator(to_Quantity), AfterValidator(check_delta_degC_Quantity)
]


def check_percent_Quantity(quantity: Quantity) -> Quantity:
    try:
        if quantity.dimensionless:
            return quantity.to("percent")
    except RecursionError:
        # breakpoint()
        raise
    raise DimensionalityError(
        quantity,
        "percent",
        dim1="",
        dim2="",
        extra_msg=f"Quantity `{quantity}` must be dimensionless",
    )


percent_Quantity = Annotated[
    Quantity, BeforeValidator(to_Quantity), AfterValidator(check_percent_Quantity)
]


def Quantity_type(units: str) -> type:
    """A method for making a pydantic compliant Pint quantity field type."""

    def validate(value, units, info):
        quantity = to_Quantity(value)
        assert quantity.is_compatible_with(
            units
        ), f"Units of {value} incompatible with {units}"
        return quantity

    def __get_pydantic_core_schema__(source_type: Any) -> CoreSchema:
        return core_schema.with_info_plain_validator_function(
            lambda value, info: validate(value, units, info)
        )

    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema["$ref"] = "#/definitions/Quantity"
        return json_schema

    return type(
        "Quantity",
        (Quantity,),
        dict(
            __get_pydantic_core_schema__=__get_pydantic_core_schema__,
            __get_pydantic_json_schema__=__get_pydantic_json_schema__,
        ),
    )


def asPintSeries(
    series: pd.Series, name=None, errors="ignore", inplace=False
) -> pd.Series:
    """:param series : pd.Series possibly containing Quantity values, not already in a PintArray.
    :param name : the name to give to the resulting series
    :param errors : { 'raise', 'ignore' }, default 'ignore'
    :param inplace : bool, default False.  If True, perform operation in-place.

    :return: If there is only one type of unit in the series, a PintArray version of the series, replacing NULL values with Quantity (np.nan, unit_type).

    Raises ValueError if there are more than one type of units in the series.
    Silently returns series if no conversion needed to be done.
    """
    # FIXME: Errors in the imput template can trigger this assertion
    if isinstance(series, pd.DataFrame):
        assert len(series) == 1
        series = series.iloc[0]

    if series.dtype != "O":
        if errors == "ignore":
            return series
        if name:
            raise ValueError(f"'{name}' not dtype('O')")
        elif series.name:
            raise ValueError(f"Series '{series.name}' not dtype('O')")
        else:
            raise ValueError("Series not dtype('O')")
    # NA_VALUEs are true NaNs, missing units
    na_values = ITR.isna(series)
    units = series[~na_values].map(lambda x: x.u if isinstance(x, Quantity) else None)  # type: ignore
    unit_first_idx = units.first_valid_index()
    if unit_first_idx is None:
        if errors != "ignore":
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
    if len(na_index) > 0:
        new_series.loc[na_index] = new_series.loc[na_index].map(
            lambda x: PintType(unit).na_value
        )
    return new_series.astype(f"pint[{unit}]")


def asPintDataFrame(df: pd.DataFrame, errors="ignore", inplace=False) -> pd.DataFrame:
    """:param df : pd.DataFrame with columns to be converted into PintArrays where possible.
    :param errors : { 'raise', 'ignore' }, default 'ignore'
    :param inplace : bool, default False.  If True, perform operation in-place.

    :return: A pd.DataFrame with columns converted to PintArrays where possible.

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


def requantify_df_from_columns(df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """:param df: pd.DataFrame
    :param inplace: bool, default False.  If True, perform operation in-place.

    :return: A pd.DataFrame with columns originally matching the pattern
    COLUMN_NAME [UNITS] renamed to COLUMN_NAME and replaced with a PintArray
    with dtype=ureg(UNITS) (aka 'pint[UNITS]')
    """
    p = re.compile(r"^(.*)\s*\[(.*)\]\s*$")
    if not inplace:
        df = df.copy()
    for column in df.columns:
        m = p.match(column)
        if m:
            col = m.group(1).strip()
            unit = m.group(2).strip()
            df.rename(columns={column: col}, inplace=True)
            df[col] = pd.Series(PA_(df[col], unit))
    return df
