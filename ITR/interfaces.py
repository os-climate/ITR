from __future__ import annotations

import numpy as np
import pandas as pd

from operator import add
from enum import Enum
from typing import Optional, Dict, List, Literal, Union
from typing import TYPE_CHECKING, Callable
from pydantic import BaseModel, parse_obj_as, validator, root_validator

import ITR
from ITR.logger import logger
from ITR.data.osc_units import ureg, Q_, M_, BenchmarkMetric, BenchmarkQuantity, ProductionMetric, ProductionQuantity, EmissionsMetric, EmissionsQuantity, EI_Metric, EI_Quantity, quantity
from ITR.configs import ProjectionControls

import pint
from pint.errors import DimensionalityError

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
        # FIXME: Should this also contain cls.S2 or no?
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

    @classmethod
    def get_result_types(cls) -> List[str]:
        """
        Get a list of all result types, ordered by priority (first << last priority).
        :return: A list of the EScoreResultType values
        """
        return [EScoreResultType.DEFAULT, EScoreResultType.TRAJECTORY_ONLY, EScoreResultType.COMPLETE]


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
    S2: Optional[IBenchmarks]
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
                              value = self.value + (0 if ITR.isnan(o.value.m) else o.value))


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
                                    value = self.value + (0 if ITR.isnan(o.value.m) else o.value))


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
                              value = self.value + (0 if ITR.isnan(o.value.m) else o.value))


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
    target_reduction_pct: float # This is actually a fraction, not a percentage.  1.0 = complete reduction to zero.

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
    historic_data: Optional[IHistoricData] # IHistoric data can contain None values; need to convert to Quantified NaNs

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

    def _normalize_historic_data(self, historic_data: IHistoricData, production_metric: ProductionMetric, emissions_metric: EmissionsMetric) -> IHistoricData:
        def _normalize(value, metric):
            if value is not None:
                return value.to(metric)
            return Q_(np.nan, metric)
        
        if historic_data is None:
            return None

        if historic_data.productions:
            historic_data.productions = [IProductionRealization(year=p.year, value=_normalize (p.value, production_metric))
                                         for p in historic_data.productions]
        ei_metric = f"{emissions_metric} / ({production_metric})"
        for scope_name in EScope.get_scopes():
            if historic_data.emissions:
                setattr(historic_data.emissions, scope_name, [IEmissionRealization(year=p.year, value=_normalize(p.value, emissions_metric))
                                                              for p in getattr(historic_data.emissions, scope_name)])
            if historic_data.emissions_intensities:
                setattr(historic_data.emissions_intensities, scope_name, [IEIRealization(year=p.year, value=_normalize(p.value, ei_metric))
                                                                          for p in getattr(historic_data.emissions_intensities, scope_name)])
        return historic_data

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
        self.historic_data = self._normalize_historic_data(self.historic_data, self.production_metric, self.emissions_metric)
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
            logger.warning(f"missing historic data for base_year_production for {self.company_name}")
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

