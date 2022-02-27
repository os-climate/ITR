from enum import Enum
from typing import Optional, Dict, List, Literal, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError, parse_obj_as

from pint import Quantity
from ITR.data.osc_units import ureg, Q_
import numpy as np
import pandas as pd


class PintModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


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


class AggregationContribution(PintModel):
    company_name: str
    company_id: str
    temperature_score: Quantity['delta_degC']
    contribution_relative: Optional[Quantity['delta_degC']]
    contribution: Optional[Quantity['delta_degC']]

    def __getitem__(self, item):
        return getattr(self, item)


class Aggregation(PintModel):
    score: Quantity['delta_degC']
    proportion: float
    contributions: List[AggregationContribution]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregation(BaseModel):
    all: Aggregation
    influence_percentage: float
    grouped: Dict[str, Aggregation]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregationScopes(BaseModel):
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


def pint_ify(x, units='error'):
    if 'units' in units:
        units = units['units']
    if x is None:
        return Q_(np.nan, units)
    if type(x) == str:
        if x.startswith('nan '):
            return Q_(np.nan, units)
        return ureg(x)
    if isinstance(x, Quantity):
        return x
    return Q_(x, units)


def UProjections_to_IProjections(ul, metric):
    if ul is None or ul is np.nan:
        return ul
    for x in ul:
        if isinstance(x, IProjection):
            return ul
    units = metric['units']
    if 'units' in units:
        units = units['units']
    pl = [dict(x) for x in ul]
    for x in pl:
        if x['value'] is None or x['value'] is np.nan:
            x['value'] = Q_(np.nan, units)
        else:
            x['value'] = pint_ify(x['value'], units)
    return pl


def UProjection_to_IProjection(u, metric):
    if u is None or u['value'] is np.nan:
        return pint_ify(np.nan, metric['units'])
    if not isinstance(u, dict):
        return u
    p = dict(u)
    p['value'] = pint_ify(p['value'], metric['units'])
    return p


def UScopes_to_IScopes(uscopes):
    if not isinstance(uscopes, dict):
        return uscopes
    iscopes = dict(uscopes)
    for skey, sval in iscopes.items():
        if iscopes[skey] is None:
            continue
        iscopes[skey] = ireports = dict(iscopes[skey])
        ireports['reports'] = u_2_i_list = ireports['reports'].copy()
        for i in range(len(u_2_i_list)):
            iscope = dict(u_2_i_list[i])
            iscope['projections'] = UProjections_to_IProjections(iscope['projections'], iscope['company_metric'])
            u_2_i_list[i] = iscope
    return iscopes


class PowerGenerationWh(BaseModel):
    units: Union[Literal['MWh'], Literal['GWh'], Literal['TWh']]


class PowerGenerationJ(BaseModel):
    units: Union[Literal['GJ'], Literal['gigajoule'], Literal['GP'], Literal['petajoule']]


PowerGeneration = Annotated[Union[PowerGenerationWh, PowerGenerationJ], Field(discriminator='units')]


class ManufactureSteel(BaseModel):
    units: Union[Literal['Fe_ton'], Literal['kiloFe_ton'], Literal['megaFe_ton']]


Manufacturing = Annotated[Union[ManufactureSteel], Field(discriminator='units')]

ProductionMetric = Annotated[Union[PowerGeneration, ManufactureSteel], Field(discriminator='units')]


class EmissionsCO2(BaseModel):
    units: Union[Literal['t CO2'], Literal['kt CO2'], Literal['Mt CO2'], Literal['Gt CO2']]


EmissionsMetric = Annotated[EmissionsCO2, Field(discriminator='units')]


class EmissionsIntensity(BaseModel):
    units: Union[
        Literal['t CO2/MWh'], Literal['t CO2/GWh'], Literal['t CO2/TWh'], Literal['t CO2/GJ'], Literal['t CO2/PJ'],
        Literal['t CO2/Fe_ton']]


class DimensionlessNumber(BaseModel):
    units: Literal['dimensionless']


OSC_Metric = Annotated[
    Union[ProductionMetric, EmissionsMetric, EmissionsIntensity, DimensionlessNumber], Field(discriminator='units')]


# U is Unquantified
class UProjection(BaseModel):
    year: int
    value: Optional[float]


class UBenchmark(BaseModel):
    sector: str
    region: str
    benchmark_metric: OSC_Metric
    projections: List[UProjection]

    def __getitem__(self, item):
        return getattr(self, item)


# I means we have quantified values.  Normally we'd need to __init__ this, but it's always handled in UProjection_to_IProjection
class IProjection(PintModel):
    year: int
    value: Optional[Quantity]


class IBenchmark(BaseModel):
    sector: str
    region: str
    benchmark_metric: OSC_Metric
    projections: List[IProjection]

    def __init__(self, benchmark_metric, projections, *args, **kwargs):
        super().__init__(benchmark_metric=benchmark_metric,
                         projections=UProjections_to_IProjections(projections, benchmark_metric),
                         *args, **kwargs)

    def __getitem__(self, item):
        return getattr(self, item)


class IBenchmarks(BaseModel):
    benchmarks: List[IBenchmark]

    def __getitem__(self, item):
        return getattr(self, item)


class IProductionBenchmarkScopes(BaseModel):
    S1S2: Optional[IBenchmarks]
    S3: Optional[IBenchmarks]
    S1S2S3: Optional[IBenchmarks]


class IEIBenchmarkScopes(PintModel):
    S1S2: Optional[IBenchmarks]
    S3: Optional[IBenchmarks]
    S1S2S3: Optional[IBenchmarks]
    benchmark_temperature: Quantity['delta_degC']
    benchmark_global_budget: Quantity['CO2']
    is_AFOLU_included: bool

    def __init__(self, benchmark_temperature, benchmark_global_budget, *args, **kwargs):
        super().__init__(benchmark_temperature=pint_ify(benchmark_temperature, 'delta_degC'),
                         benchmark_global_budget=pint_ify(benchmark_global_budget, 'Gt CO2'),
                         *args, **kwargs)

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjection(BaseModel):
    company_metric: OSC_Metric
    projections: List[IProjection]

    def __init__(self, projections, *args, **kwargs):
        super().__init__(projections=UProjections_to_IProjections(projections, company_metric),
                         *args, **kwargs)

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjection(PintModel):
    year: int
    value: Optional[Quantity[EmissionsIntensity]]

    def __init__(self, year, value):
        super().__init__(year=year, value=pint_ify(value, 't CO2/MWh'))

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjections(BaseModel):
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


class IProductionRealization(PintModel):
    year: int
    value: Optional[Quantity[ProductionMetric]]

    def __init__(self, year, value=None):
        super().__init__(year=year, value=value)
        if value is None:
            self.value = None


class IEmissionRealization(PintModel):
    year: int
    value: Optional[Quantity['CO2']]

    def __init__(self, year, value):
        super().__init__(year=year, value=pint_ify(value, 't CO2'))
        if value is None:
            self.value = None


class IHistoricEmissionsScopes(PintModel):
    S1: List[IEmissionRealization]
    S2: List[IEmissionRealization]
    S1S2: List[IEmissionRealization]
    S3: List[IEmissionRealization]
    S1S2S3: List[IEmissionRealization]


class IEIRealization(PintModel):
    year: int
    value: Optional[Quantity[EmissionsIntensity]]

    def __init__(self, year, value):
        super().__init__(year=year, value=value)
        if value is None:
            self.value = None


class IHistoricEIScopes(PintModel):
    S1: List[IEIRealization]
    S2: List[IEIRealization]
    S1S2: List[IEIRealization]
    S3: List[IEIRealization]
    S1S2S3: List[IEIRealization]


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
        return [cls.S1S2, cls.S3, cls.S1S2S3]


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


class IHistoricData(PintModel):
    productions: Optional[List[IProductionRealization]]
    emissions: Optional[IHistoricEmissionsScopes]
    emissions_intensities: Optional[IHistoricEIScopes]


class ITargetData(PintModel):
    netzero_year: Optional[int]
    target_type: Union[Literal['intensity'], Literal['absolute'], Literal['other']]
    target_scope: EScope
    start_year: Optional[int]
    base_year: int
    end_year: int

    target_base_qty: float
    target_base_unit: str
    target_reduction_pct: float


class ICompanyData(PintModel):
    company_name: str
    company_id: str

    region: str  # TODO: make SortableEnums
    sector: str  # TODO: make SortableEnums
    target_probability: float = 0.5

    target_data: Optional[List[ITargetData]]
    historic_data: Optional[IHistoricData]

    country: Optional[str]

    emissions_metric: Optional[EmissionsMetric]     # Typically use t CO2 for MWh/GJ and Mt CO2 for TWh/PJ
    production_metric: Optional[ProductionMetric] # Optional because it can be inferred from sector and region
    
    # These three instance variables match against financial data below, but are incomplete as historic_data and target_data
    base_year_production: Optional[Quantity[ProductionMetric]]
    ghg_s1s2: Optional[Quantity[EmissionsMetric]]
    ghg_s3: Optional[Quantity[EmissionsMetric]]

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
    def _fixup_historic_productions(self, historic_productions, production_metric):
        if historic_productions is None or production_metric is None:
            # We have absolutely no production data of any kind...too bad!
            return self.historic_data.productions
        return UProjections_to_IProjections(historic_productions, production_metric)

    def __init__(self, historic_data=None, projected_targets=None, projected_intensities=None, emissions_metric=None,
                 production_metric=None, base_year_production=None, ghg_s1s2=None, ghg_s3=None, *args, **kwargs):
        super().__init__(historic_data=historic_data,
                         projected_targets=projected_targets,
                         projected_intensities=projected_intensities,
                         emissions_metric=emissions_metric,
                         production_metric=production_metric,
                         *args, **kwargs)
        # In-bound parameters are dicts, which are converted to models by __super__ and stored as instance variables
        if production_metric is None:
            if self.sector == 'Electricity Utilities':
                units = 'MWh' if self.region == 'North America' else 'GJ'
            elif self.sector == 'Steel':
                units = 'Fe_ton'
            else:
                raise ValueError("No source of production metrics")
            self.production_metric = parse_obj_as(ProductionMetric, {'units': units})
            if emissions_metric is None:
                self.emissions_metric = parse_obj_as(EmissionsMetric, {'units': 't CO2'})
        elif emissions_metric is None:
            if self.production_metric.units in ['TWh', 'PJ']:
                self.emissions_metric = parse_obj_as(EmissionsMetric, {'units': 'Mt CO2'})
            else:
                self.emissions_metric = parse_obj_as(EmissionsMetric, {'units': 't CO2'})
            # TODO: Should raise a warning here
        if base_year_production:
            self.base_year_production = pint_ify(base_year_production, self.production_metric.units)
        elif self.historic_data and self.historic_data.productions:
            # TODO: This is a hack to get things going.
            year = kwargs['report_date'].year
            for i in range(len(self.historic_data.productions)):
                if self.historic_data.productions[-1 - i].year == year:
                    self.base_year_production = self.historic_data.productions[-1 - i].value
                    break
            if self.base_year_production is None:
                raise ValueError("invalid historic data for base_year_production")
        else:
            raise ValueError("missing historic data for base_year_production")
        if ghg_s1s2:
            self.ghg_s1s2=pint_ify(ghg_s1s2, self.emissions_metric.units)
        elif self.historic_data and self.historic_data.emissions:
            # TODO: This is a hack to get things going.
            year = kwargs['report_date'].year
            for i in range(len(self.historic_data.emissions.S1S2)):
                if self.historic_data.emissions.S1S2[-1  -i].year == year:
                    self.ghg_s1s2 = self.historic_data.emissions.S1S2[-1 - i].value
                    break
            if self.ghg_s1s2 is None:
                # TODO: cheap hack to treat S1 as S1S2, which we do for now for Consolidated Edison, Inc.
                for i in range(len(self.historic_data.emissions.S1)):
                    if self.historic_data.emissions.S1[-1 - i].year == year:
                        self.ghg_s1s2 = self.historic_data.emissions.S1[-1 - i].value
                        break
                if self.ghg_s1s2 is None:
                    print(self.company_name)
                    raise ValueError("invalid historic data for ghg_s1s2")
        else:
            raise ValueError("missing historic data for ghg_s1s2")
        if ghg_s3:
            self.ghg_s3 = pint_ify(ghg_s3, self.emissions_metric.units)
        # TODO: We don't need to worry about missing S3 scope data yet


class ICompanyAggregates(ICompanyData):
    cumulative_budget: Quantity['CO2']
    cumulative_trajectory: Quantity['CO2']
    cumulative_target: Quantity['CO2']
    benchmark_temperature: Quantity['delta_degC']
    benchmark_global_budget: Quantity['CO2']

    # projected_targets: Optional[ICompanyEIProjectionsScopes]
    # projected_intensities: Optional[ICompanyEIProjectionsScopes]

    def __init__(self, cumulative_budget, cumulative_trajectory, cumulative_target, benchmark_temperature,
                 benchmark_global_budget, *args, **kwargs):
        super().__init__(
            cumulative_budget=pint_ify(cumulative_budget, 't CO2'),
            cumulative_trajectory=pint_ify(cumulative_trajectory, 't CO2'),
            cumulative_target=pint_ify(cumulative_target, 't CO2'),
            benchmark_temperature=pint_ify(benchmark_temperature, 'delta_degC'),
            benchmark_global_budget=pint_ify(benchmark_global_budget, 'Gt CO2'),
            *args, **kwargs)


class TemperatureScoreControls(PintModel):
    base_year: int
    target_end_year: int
    projection_start_year: int
    projection_end_year: int
    tcre: Quantity['delta_degC']
    carbon_conversion: Quantity['CO2']
    scenario_target_temperature: Quantity['delta_degC']

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def tcre_multiplier(self) -> Quantity['delta_degC/CO2']:
        return self.tcre / self.carbon_conversion
