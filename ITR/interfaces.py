from __future__ import annotations

import numpy as np
import pandas as pd

from operator import add
from enum import Enum
from typing import Optional, Dict, List, Literal, Union
from typing import TYPE_CHECKING, Callable
from pydantic import BaseModel, parse_obj_as, validator, root_validator
from dataclasses import dataclass

import pint
import ITR
from ITR.data.osc_units import ureg, Q_, M_
from pint.errors import DimensionalityError


@dataclass
class ProjectionControls:
    LOWER_PERCENTILE: float = 0.1
    UPPER_PERCENTILE: float = 0.9

    LOWER_DELTA: float = -0.10
    UPPER_DELTA: float = +0.03

    # FIXME: Should agree with TemperatureScoreConfig.CONTROLS_CONFIG
    BASE_YEAR: int = 2019
    TARGET_YEAR: int = 2050
    TREND_CALC_METHOD: Callable[[pd.DataFrame], pd.DataFrame] = staticmethod(pd.DataFrame.median)


class PintModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

# List of all the production units we know
_production_units = [ "Wh", "pkm", "tkm", "boe", "t Aluminum", "t Cement", "t Copper", "t Paper", "t Steel", "USD", "m**2" ]
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
        raise ValueError(f"{v} not relateable to {_production_units}")

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
        return f'ProductionMetric({super().__repr__()})'

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
        return f'ProductionMetric({super().__repr__()})'

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
        return f'ProductionMetric({super().__repr__()})'

# Borrowed from https://github.com/hgrecco/pint/issues/1166
registry = ureg

schema_extra = dict(definitions=[
    dict(
        Quantity=dict(type="string"),
        # We don't need EmissionsQuantity separately because Quantity does all we nmeed
        ProductionQuantity=dict(type="List[str]"),
        EI_Quantity=dict(type="List[str]"),
    )
])


def quantity(dimensionality: str) -> type:
    """A method for making a pydantic compliant Pint quantity field type."""

    try:
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
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        elif isinstance(value, pint.Quantity):
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
        (pint.Quantity,),
        dict(
            __get_validators__=__get_validators__,
            __modify_schema__=__modify_schema__,
            dimensionality=dimensionality,
            validate=validate,
        ),
    )
# end of borrowing


class EmissionsQuantity(pint.Quantity):
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
    def validate(cls, quantity):
        if quantity is None:
            breakpoint()
        if isinstance(quantity, str):
            v, u = quantity.split(' ', 1)
            try:
                q = Q_(float(v), u)
            except ValueError:
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        if not isinstance(quantity, pint.Quantity):
            raise TypeError(f"pint.Quantity required ({quantity}, type = {type(quantity)})")
        if quantity.is_compatible_with('t CO2'):
            return quantity
        raise DimensionalityError (quantity, 't CO2', dim1='', dim2='', extra_msg=f"Dimensionality must be compatible with 't CO2'")

    distance: quantity("[length]")
    speed: quantity("[length]/[time]")

    class Config:
        validate_assignment = True
        schema_extra = schema_extra
        json_encoders = {
            pint.Quantity: str,
        }
# end of borrowing


def emissions_quantity(dimensionality: str) -> type:
    """A method for making a pydantic compliant Pint quantity field type."""

    def __new__(cls, value, units=None):
        # Re-used the instance we are passed.  Do we need to copy?
        return value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if isinstance(value, dict):
            quantity = Q_(value['units'])
        else:
            quantity = Q_(value)
        if quantity.is_compatible_with(cls.dimensionality):
            return quantity
        assert quantity.check(cls.dimensionality), f"Dimensionality must be {cls.dimensionality} {breakpoint()}"
        return quantity

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            {"$ref": "#/definitions/EmissionsQuantity"}
        )
    
    return type(
        "Quantity",
        (pint.Quantity,),
        dict(
            __get_validators__=__get_validators__,
            __modify_schema__=__modify_schema__,
            dimensionality=dimensionality,
            validate=validate,
        ),
    )

    @classmethod
    def validate(cls, quantity):
        if quantity is None:
            breakpoint()
        if isinstance(quantity, str):
            v, u = quantity.split(' ', 1)
            try:
                q = Q_(float(v), u)
            except ValueError:
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        if not isinstance(quantity, pint.Quantity):
            raise TypeError('pint.Quantity required')
        for pu in _production_units:
            if quantity.is_compatible_with(pu):
                return quantity
        raise DimensionalityError (quantity, str(_production_units), dim1='', dim2='', extra_msg=f"Dimensionality must be compatible with [{_production_units}]")

    @validator('dims_list')
    def units_must_be_registered(cls, v):
        for d in v:
            try:
                registry.get_dimensionality(d)
            except KeyError:
                raise ValueError(f"{d} is not a valid dimensionality in pint!")
        return v

    class Config:
        validate_assignment = True
        schema_extra = schema_extra
        json_encoders = {
            pint.Quantity: str,
        }

def production_quantity(dims_list: List[str]) -> type:
    """A method for making a pydantic compliant Pint production quantity."""

class EI_Quantity(str):
    """A method for making a pydantic compliant Pint Emissions Intensity quantity."""

    def __new__(cls, value, units=None):
        # Re-used the instance we are passed.  Do we need to copy?
        return value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if isinstance(value, dict):
            quantity = Q_(value['units'])
        else:
            quantity = Q_(value)
        for dimensionality in ProductionQuantity(dims_list=dims_list).dims_list:
            if quantity.check(dimensionality):
                return quantity
        raise DimensionalityError(value.units, f"in [{ProductionQuantity(dims_list).dims_list}]")

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            {"$ref": "#/definitions/ProductionQuantity"}
        )
    
    return type(
        "ProductionQuantity",
        (ProductionQuantity,),
        dict(
            __get_validators__=__get_validators__,
            __modify_schema__=__modify_schema__,
            dims_list=f"List[str] = {dims_list}",
            validate=validate,
        ),
    )

    @classmethod
    def validate(cls, quantity):
        if quantity is None:
            breakpoint()
        if isinstance(quantity, str):
            v, u = quantity.split(' ', 1)
            try:
                q = Q_(float(v), u)
            except ValueError:
                raise ValueError(f"cannot convert '{quantity}' to quantity")
            quantity = q
        if not isinstance(quantity, pint.Quantity):
            raise TypeError('pint.Quantity required')
        for ei_u in _ei_units:
            if quantity.is_compatible_with(ei_u):
                return quantity
        raise DimensionalityError (quantity, str(_ei_units), dim1='', dim2='', extra_msg=f"Dimensionality must be compatible with [{_ei_units}]")

class EI_Quantity(BaseModel):

    dims_list: List[str]

    @validator('dims_list')
    def units_must_be_registered(cls, v):
        for d in v:
            try:
                registry.get_dimensionality(d)
            except KeyError:
                raise ValueError(f"{d} is not a valid dimensionality in pint!")
        return v

    class Config:
        validate_assignment = True
        schema_extra = schema_extra
        json_encoders = {
            pint.Quantity: str,
        }

def ei_quantity(dims_list: List[str]) -> type:
    """A method for making a pydantic compliant Pint Emissions Intensity (EI) quantity."""

class BenchmarkQuantity(str):
    """A method for making a pydantic compliant Pint Benchmark quantity (which includes dimensionless production growth)."""

    def __new__(cls, value, units=None):
        # Re-used the instance we are passed.  Do we need to copy?
        return value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if isinstance(value, dict):
            quantity = Q_(value['units'])
        else:
            quantity = Q_(value)
        for dimensionality in EI_Quantity(dims_list=dims_list).dims_list:
            if quantity.check(dimensionality):
                return quantity
        raise DimensionalityError(value.units, f"in [{EI_Quantity(dims_list).dims_list}]")

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            {"$ref": "#/definitions/EI_Quantity"}
        )
    
    return type(
        "EI_Quantity",
        (EI_Quantity,),
        dict(
            __get_validators__=__get_validators__,
            __modify_schema__=__modify_schema__,
            dims_list=f"List[str] = {dims_list}",
            validate=validate,
        ),
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
        if not isinstance(quantity, pint.Quantity):
            raise TypeError('pint.Quantity required')
        if str(quantity.u) == 'dimensionless':
            return quantity
        for ei_u in _ei_units:
            if quantity.is_compatible_with(ei_u):
                return quantity
        raise DimensionalityError (quantity, str(_ei_units), dim1='', dim2='', extra_msg=f"Dimensionality must be 'dimensionless' or compatible with [{_ei_units}]")


class OSC_Metric(BaseModel):
    units: str 
    @validator('units')
    def units_must_be_OSC(cls, v):
        if v == 'dimensionless':
            return v
        try:
            if ProductionMetric.unit_must_be_production(v):
                return v
        except ValueError:
            try:
                if EmissionsMetric.units_must_be_tCO2(v):
                    return v
            except ValueError:
                try:
                    if IntensityMetric.units_must_be_EI(v):
                        return v
                except ValueError:
                    raise ValueError(f"cannot understand {v} as OSC_Metric")


class SortableEnum(Enum):
    def __str__(self):
        return self.name

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) >= order.index(other)
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) > order.index(other)
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) <= order.index(other)
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            order = list(self.__class__)
            return order.index(self) < order.index(other)
        return NotImplemented


class EScope(SortableEnum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S1S2 = "S1+S2"
    S1S2S3 = "S1+S2+S3"

    @classmethod
    def get_scopes(cls) -> List[str]:
        """
        Get a list of all scopes.
        :return: A list of EScope string values
        """
        return ['S1', 'S2', 'S3', 'S1S2', 'S1S2S3']

    @classmethod
    def get_result_scopes(cls) -> List['EScope']:
        """
        Get a list of scopes that should be calculated if the user leaves it open.

        :return: A list of EScope objects
        """
        return [cls.S1, cls.S1S2, cls.S3, cls.S1S2S3]


class ETimeFrames(SortableEnum):
    """
    TODO: add support for multiple timeframes. Long currently corresponds to 2050.
    """
    SHORT = "short"
    MID = "mid"
    LONG = "long"


class ECarbonBudgetScenario(Enum):
    P25 = "25 percentile"
    P75 = "75 percentile"
    MEAN = "Average"


class EScoreResultType(Enum):
    DEFAULT = "Default"
    TRAJECTORY_ONLY = "Trajectory only"
    COMPLETE = "Complete"


class AggregationContribution(PintModel):
    company_name: str
    company_id: str
    temperature_score: quantity('delta_degC')
    contribution_relative: Optional[quantity('percent')]
    contribution: Optional[quantity('delta_degC')]

    def __getitem__(self, item):
        return getattr(self, item)


class Aggregation(PintModel):
    score: quantity('delta_degC')
    # proportion is a number from 0..1
    proportion: float
    contributions: List[AggregationContribution]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregation(BaseModel):
    all: Aggregation
    influence_percentage: quantity('percent')
    grouped: Dict[str, Aggregation]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregationScopes(BaseModel):
    S1: Optional[ScoreAggregation]
    S1S2: Optional[ScoreAggregation]
    S3: Optional[ScoreAggregation]
    S1S2S3: Optional[ScoreAggregation]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregations(BaseModel):
    short: Optional[ScoreAggregationScopes]
    mid: Optional[ScoreAggregationScopes]
    long: Optional[ScoreAggregationScopes]

    def __getitem__(self, item):
        return getattr(self, item)


class PortfolioCompany(BaseModel):
    company_name: str
    company_id: str
    company_isin: Optional[str]
    investment_value: float
    user_fields: Optional[dict]


# U is Unquantified, which is presently how our benchmarks come in (production_metric comes in elsewhere)
class UProjection(BaseModel):
    year: int
    value: Optional[float]

class IProjection(BaseModel):
    year: int
    value: Optional[pint.Quantity]


class IBenchmark(BaseModel):
    sector: str
    region: str
    benchmark_metric: BenchmarkMetric
    projections_nounits: Optional[List[UProjection]]
    projections: Optional[List[IProjection]]
    base_year_production: Optional[ProductionQuantity] # FIXME: applies only to production benchmarks

    def __init__(self, benchmark_metric, projections_nounits=None, projections=None,
                 base_year_production=None, *args, **kwargs):
        # FIXME: Probably want to define `target_end_year` to be 2051, not 2050...
        super().__init__(benchmark_metric=benchmark_metric,
                         projections_nounits=projections_nounits,
                         projections=projections,
                         base_year_production=base_year_production,
                         *args, **kwargs)
        # Sadly we need to build the full projection range before cutting it down to size...
        # ...until Tiemann learns the bi-valence of dict and Model parameters
        if self.projections_nounits:
            if self.projections:
                # Check if we've already seen/processed these exact projections
                changed_projections = [p for p in self.projections if not any([n for n in self.projections_nounits if n.year==p.year and n.value==p.value.m])]
                if changed_projections:
                    breakpoint()
                return
            self.projections = [IProjection(year=p.year, value=BenchmarkQuantity(Q_(p.value, benchmark_metric))) for p in self.projections_nounits
                                if p.year in range(ProjectionControls.BASE_YEAR,
                                                   ProjectionControls.TARGET_YEAR+1)]
        elif not self.projections:
            logger.warning(f"Empty Benchmark for sector {sector}, region {region}")


    def __getitem__(self, item):
        return getattr(self, item)


class IBenchmarks(BaseModel):
    benchmarks: List[IBenchmark]
    production_centric = False

    def __getitem__(self, item):
        return getattr(self, item)


class IProductionBenchmarkScopes(BaseModel):
    AnyScope: Optional[IBenchmarks]
    S1: Optional[IBenchmarks]
    S1S2: Optional[IBenchmarks]
    S3: Optional[IBenchmarks]
    S1S2S3: Optional[IBenchmarks]


class IEIBenchmarkScopes(BaseModel):
    S1: Optional[IBenchmarks]
    S1S2: Optional[IBenchmarks]
    S3: Optional[IBenchmarks]
    S1S2S3: Optional[IBenchmarks]
    benchmark_temperature: quantity('delta_degC')
    benchmark_global_budget: quantity('Gt CO2')
    is_AFOLU_included: bool

    def __init__(self, benchmark_temperature, benchmark_global_budget, *args, **kwargs):
        super().__init__(benchmark_temperature=pint_ify(benchmark_temperature, 'delta_degC'),
                         benchmark_global_budget=pint_ify(benchmark_global_budget, 'Gt CO2'),
                         *args, **kwargs)

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjection(PintModel):
    year: int
    value: Optional[ei_quantity(_ei_units)]

    def add(self, o):
        assert self.year==o.year
        return IEIRealization(year=self.year,
                              value = self.value + 0 if ITR.isnan(o.value.m) else o.value)


class ICompanyEIProjections(BaseModel):
    ei_metric: IntensityMetric
    projections: List[ICompanyEIProjection]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjectionsScopes(BaseModel):
    S1: Optional[ICompanyEIProjections]
    S2: Optional[ICompanyEIProjections]
    S1S2: Optional[ICompanyEIProjections]
    S3: Optional[ICompanyEIProjections]
    S1S2S3: Optional[ICompanyEIProjections]

    def __getitem__(self, item):
        return getattr(self, item)


class IProductionRealization(BaseModel):
    year: int
    value: Optional[production_quantity(_production_units)]


class IEmissionRealization(PintModel):
    year: int
    value: Optional[emissions_quantity('t CO2')]

    def add(self, o):
        assert self.year==o.year
        return IEmissionRealization(year=self.year,
                                    value = self.value + 0 if ITR.isnan(o.value.m) else o.value)


class IHistoricEmissionsScopes(PintModel):
    S1: List[IEmissionRealization]
    S2: List[IEmissionRealization]
    S1S2: List[IEmissionRealization]
    S3: List[IEmissionRealization]
    S1S2S3: List[IEmissionRealization]


class IEIRealization(PintModel):
    year: int
    value: Optional[ei_quantity(_ei_units)]

    def add(self, o):
        assert self.year==o.year
        return IEIRealization(year=self.year,
                              value = self.value + 0 if ITR.isnan(o.value.m) else o.value)


class IHistoricEIScopes(PintModel):
    S1: List[IEIRealization]
    S2: List[IEIRealization]
    S1S2: List[IEIRealization]
    S3: List[IEIRealization]
    S1S2S3: List[IEIRealization]


class IHistoricData(PintModel):
    productions: Optional[List[IProductionRealization]]
    emissions: Optional[IHistoricEmissionsScopes]
    emissions_intensities: Optional[IHistoricEIScopes]


class ITargetData(PintModel):
    netzero_year: Optional[int]
    target_type: Union[Literal['intensity'], Literal['absolute'], Literal['Intensity'], Literal['Absolute']]
    target_scope: EScope
    target_start_year: Optional[int]
    target_base_year: int
    target_end_year: int

    target_base_year_qty: float
    target_base_year_unit: str
    target_reduction_pct: float

    @root_validator
    def must_be_greater_than_2022(cls, v):
        if v['target_end_year'] < 2023:
            raise ValueError(f"Scope {v['target_scope']}: Target end year ({v['target_end_year']}) must be greater than 2022")
        return v


class ICompanyData(PintModel):
    company_name: str
    company_id: str

    sector: str  # TODO: make SortableEnums
    region: str  # TODO: make SortableEnums
    target_probability: float = 0.5

    target_data: Optional[List[ITargetData]]
    historic_data: Optional[IHistoricData]

    country: Optional[str]

    emissions_metric: Optional[EmissionsMetric]    # Typically use t CO2 for MWh/GJ and Mt CO2 for TWh/PJ
    production_metric: Optional[ProductionMetric]
    
    # These three instance variables match against financial data below, but are incomplete as historic_data and target_data
    base_year_production: Optional[production_quantity(_production_units)]
    ghg_s1s2: Optional[emissions_quantity('t CO2')]
    ghg_s3: Optional[emissions_quantity('t CO2')]

    industry_level_1: Optional[str]
    industry_level_2: Optional[str]
    industry_level_3: Optional[str]
    industry_level_4: Optional[str]

    company_revenue: Optional[float]
    company_market_cap: Optional[float]
    company_enterprise_value: Optional[float]
    company_ev_plus_cash: Optional[float]
    company_total_assets: Optional[float]
    company_cash_equivalents: Optional[float]

    # Initialized later when we have benchmark information.  It is OK to initialize as None and fix later.
    # They will show up as {'S1S2': { 'projections': [ ... ] }}
    projected_targets: Optional[ICompanyEIProjectionsScopes]
    projected_intensities: Optional[ICompanyEIProjectionsScopes]

    # TODO: Do we want to do some sector inferencing here?
    
    def _fixup_year_value_list(self, ListType, u_list, metric, inferred_metric):
        # u_list is unprocessed; i_list is processed; r_list is returned list
        i_list = [ul.dict() if isinstance(ul, BaseModel)
                  # In Python 3.9, dictionary union of x, y is x | y
                  # In Python 3.8, it's {**x, **y}
                  else {**{'year':ul['year']}, **{'value':Q_(ul['value'])
                                                  # Make NaNs dimensionless for now...we will fixup below
                                                  if ul['value'] is not None else Q_(np.nan, 'dimensionless')}}
                  for ul in u_list]
        if not i_list:
            return []
        if metric is None:
            try:
                metric = next(str(x['value'].u) for x in i_list if str(x['value'].u) != 'dimensionless')
            except StopIteration as e:
                # TODO: If everything in the list is empty, why not NULL it out and return []?
                metric = inferred_metric
        elif isinstance(metric, dict):
            metric = metric['units']
        else:
            metric = metric.u
        for il in i_list:
            if str(il['value'].u) == 'dimensionless':
                il['value'] = Q_(il['value'].m, metric)
        r_list = UProjections_to_IProjections(ListType, i_list, {'units':metric})
        return r_list
    
    def _sector_to_production_units(self, sector, region="Global"):
        sector_unit_dict = {
            'Electricity Utilities': { 'North America':'MWh', 'Global': 'GJ' },
            'Gas Utilities': { 'Global': 'PJ' },
            'Utilities': { 'Global': 'PJ' },
            'Steel': { 'Global': 't Steel' },
            'Aluminum': { 'Global': 't Aluminum' },
            'Oil & Gas': { 'Global': 'mmboe' },
            'Autos': { 'Global': 'pkm' },
            'Trucking': { 'Global': 'tkm' },
            'Cement': { 'Global': 't Cement' },
            'Construction Buildings': { 'Global': 'billion USD' },
            'Residential Buildings': { 'Global': 'billion m**2' }, # Should it be 'built m**2' ?
            'Commercial Buildings': { 'Global': 'billion m**2' }, # Should it be 'built m**2' ?
            'Textiles': { 'Global': 'billion USD' },
            'Chemicals': { 'Global': 'billion USD' },
        }
        units = None
        if sector_unit_dict.get(sector):
            region_unit_dict = sector_unit_dict[sector]
            if region_unit_dict.get(region):
                units = region_unit_dict[region]
            else:
                units = region_unit_dict['Global']
        else:
            raise ValueError(f"No source of production metrics for {self.company_name}")
        return units        

    def _get_base_realization_from_historic(self, realized_values: List[BaseModel], units, base_year=None):
        valid_realizations = [rv for rv in realized_values if rv.value is not None and not ITR.isnan(rv.value.magnitude)]
        if not valid_realizations:
            retval = realized_values[0].copy()
            retval.year = None
            return retval
        valid_realizations.sort(key=lambda x:x.year, reverse=True)
        if base_year and valid_realizations[0].year != base_year:
            retval = realized_values[0].copy()
            retval.year = base_year
            # FIXME: Unless and until we accept uncertainties as input, rather than computed data, we don't need to make this a UFloat here
            retval.value = Q_(np.nan, units)
            return retval
        return valid_realizations[0]

    def __init__(self, emissions_metric=None, production_metric=None, base_year_production=None, ghg_s1s2=None, ghg_s3=None,
                 target_data=None, historic_data=None, *args, **kwargs):
        super().__init__(emissions_metric=emissions_metric,
                         production_metric=production_metric,
                         base_year_production=base_year_production,
                         ghg_s1s2=ghg_s1s2, ghg_s3=ghg_s3,
                         target_data=target_data,
                         historic_data=historic_data,
                         *args, **kwargs)
        # In-bound parameters are JSON (str, int, float, dict), which are converted to models by __super__ and stored as instance variables
        if production_metric is None:
            units = self._sector_to_production_units(self.sector, self.region)
            self.production_metric = ProductionMetric(units)
            if emissions_metric is None:
                self.emissions_metric = EmissionsMetric('t CO2')
        elif emissions_metric is None:
            if str(self.production_metric) in ['TWh', 'PJ', 'Mt Steel', 'megaFe_ton', 'mmboe']:
                self.emissions_metric = EmissionsMetric('Mt CO2')
            else:
                self.emissions_metric = parse_obj_as(EmissionsMetric, {'units': 't CO2'})
            # TODO: Should raise a warning here
        base_year = None
        if self.base_year_production:
            pass
        # Right now historic_data comes in via template.py ESG data
        elif self.historic_data and self.historic_data.productions:
            # TODO: This is a hack to get things going.
            base_realization = self._get_base_realization_from_historic(self.historic_data.productions, self.production_metric.units, base_year)
            base_year = base_realization.year
            self.base_year_production = base_realization.value
        else:
            raise ValueError(f"missing historic data for base_year_production for {self.company_name}")
            self.base_year_production = Q_(np.nan, str(self.production_metric))
        if self.ghg_s1s2 is None and self.historic_data and self.historic_data.emissions:
            if self.historic_data.emissions.S1S2:
                base_realization = self._get_base_realization_from_historic(self.historic_data.emissions.S1S2, self.emissions_metric.units, base_year)
                base_year = base_year or base_realization.year
                self.ghg_s1s2 = base_realization.value
            elif self.historic_data.emissions.S1 and self.historic_data.emissions.S2:
                base_realization_s1 = self._get_base_realization_from_historic(self.historic_data.emissions.S1, self.emissions_metric.units, base_year)
                base_realization_s2 = self._get_base_realization_from_historic(self.historic_data.emissions.S2, self.emissions_metric.units, base_year)
                base_year = base_year or base_realization_s1.year
                if base_realization_s1.value is not None and base_realization_s2.value is not None:
                    self.ghg_s1s2 = base_realization_s1.value + base_realization_s2.value
        if self.ghg_s1s2 is None and self.historic_data and self.historic_data.emissions_intensities:
            intensity_units = (Q_(1.0, self.emissions_metric.units) / Q_(1.0, self.production_metric.units)).units
            if self.historic_data.emissions_intensities.S1S2:
                base_realization = self._get_base_realization_from_historic(self.historic_data.emissions_intensities.S1S2, intensity_units, base_year)
                base_year = base_year or base_realization.year
                if base_realization.value is not None:
                    self.ghg_s1s2 = base_realization.value * self.base_year_production
            elif self.historic_data.emissions_intensities.S1 and self.historic_data.emissions_intensities.S2:
                base_realization_s1 = self._get_base_realization_from_historic(self.historic_data.emissions_intensities.S1, intensity_units, base_year)
                base_realization_s2 = self._get_base_realization_from_historic(self.historic_data.emissions_intensities.S2, intensity_units, base_year)
                base_year = base_year or base_realization_s1.year
                if base_realization_s1.value is not None and base_realization_s2.value is not None:
                    self.ghg_s1s2 = (base_realization_s1.value + base_realization_s2.value) * self.base_year_production
            else:
                raise ValueError(f"missing S1S2 historic intensity data for {self.company_name}")
        if self.ghg_s1s2 is None:
            raise ValueError(f"missing historic emissions or intensity data to calculate ghg_s1s2 for {self.company_name}")
        if self.ghg_s3 is None and self.historic_data and self.historic_data.emissions and self.historic_data.emissions.S3:
            base_realization_s3 = self._get_base_realization_from_historic(self.historic_data.emissions.S3, str(self.emissions_metric), base_year)
            self.ghg_s3 = base_realization_s3.value
        if self.ghg_s3 is None and self.historic_data and self.historic_data.emissions_intensities:
            if self.historic_data.emissions_intensities.S3:
                intensity_units = (Q_(1.0, self.emissions_metric.units) / Q_(1.0, self.production_metric.units)).units
                base_realization_s3 = self._get_base_realization_from_historic(self.historic_data.emissions_intensities.S3, intensity_units, base_year)
                if base_realization_s3.value is not None:
                    self.ghg_s3 = base_realization_s3.value * self.base_year_production


# These aggregate terms are all derived from the benchmark being used
class ICompanyAggregates(ICompanyData):
    cumulative_budget: emissions_quantity('t CO2')
    cumulative_trajectory: emissions_quantity('t CO2')
    cumulative_target: emissions_quantity('t CO2')
    benchmark_temperature: quantity('delta_degC')
    benchmark_global_budget: EmissionsQuantity
    scope: EScope

    # projected_production is computed but never saved, so computed at least 2x: initialiation/projection and cumulative budget
    # projected_targets: Optional[ICompanyEIProjectionsScopes]
    # projected_intensities: Optional[ICompanyEIProjectionsScopes]

class TemperatureScoreControls(BaseModel):
    base_year: int
    target_end_year: int
    projection_start_year: int
    projection_end_year: int
    tcre: quantity('delta_degC')
    carbon_conversion: emissions_quantity('t CO2')
    scenario_target_temperature: quantity('delta_degC')

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def tcre_multiplier(self) -> quantity('delta_degC/(t CO2)'):
        return self.tcre / self.carbon_conversion

# FIXME: Can somebody help sort out the circularities we have?
IEIRealization.update_forward_refs()
TemperatureScoreControls.update_forward_refs()
