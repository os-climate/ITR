from enum import Enum
from typing import Optional, Dict, List, Literal, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError

from pint import Quantity
from ITR.data.osc_units import ureg, Q_
import numpy as np

class PintModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

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
    if type(x)==str:
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
    if not isinstance(u,dict):
        return u
    p = dict(u)
    p['value'] = pint_ify(p['value'], metric['units'])
    return p


def UScopes_to_IScopes(uscopes):
    if not isinstance(uscopes,dict):
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
    units: Literal['MWh']
class PowerGenerationJ(BaseModel):
    units: Literal['GJ']
PowerGeneration = Annotated[Union[PowerGenerationWh, PowerGenerationJ], Field(discriminator='units')]


class ManufactureSteel(BaseModel):
    units: Literal['Fe_ton']
Manufacturing = Annotated[Union[ManufactureSteel], Field(discriminator='units')]


ProductionMetric = Annotated[Union[PowerGeneration, ManufactureSteel], Field(discriminator='units')]


class EmissionIntensity(BaseModel):
    units: Union[Literal['t CO2/MWh'],Literal['t CO2/GJ'],Literal['t CO2/Fe_ton']]


class DimensionlessNumber(BaseModel):
    units: Literal['dimensionless']


OSC_Metric = Annotated[Union[ProductionMetric,EmissionIntensity,DimensionlessNumber], Field(discriminator='units')]

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

# I means we have quantified values
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


class IEmissionIntensityBenchmarkScopes(PintModel):
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

    def __init__(self, company_metric, projections, *args, **kwargs):
        super().__init__(company_metric=company_metric,
                         projections=UProjections_to_IProjections(projections, company_metric),
                         *args, **kwargs)

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjections(BaseModel):
    reports: List[ICompanyProjection]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjectionsScopes(BaseModel):
    S1S2: Optional[ICompanyProjections]
    S3: Optional[ICompanyProjections]
    S1S2S3: Optional[ICompanyProjections]

    def __getitem__(self, item):
        return getattr(self, item)

# ICompanyData does not itself use Quantity, but classes derived from it do
class ICompanyData(PintModel):
    company_name: str
    company_id: str

    region: str  # TODO: make SortableEnums
    sector: str  # TODO: make SortableEnums
    target_probability: float

    projected_ei_targets: Optional[ICompanyProjectionsScopes] = None
    projected_ei_trajectories: Optional[ICompanyProjectionsScopes] = None

    country: Optional[str]
    production_metric: ProductionMetric
    ghg_s1s2: Optional[Quantity]    # This seems to be the base year PRODUCTION number, nothing at all to do with any quantity of actual S1S2 emissions
    ghg_s3: Optional[Quantity]

    industry_level_1: Optional[str]
    industry_level_2: Optional[str]
    industry_level_3: Optional[str]
    industry_level_4: Optional[str]

    company_revenue: Optional[float]
    company_market_cap: Optional[float]
    company_enterprise_value: Optional[float]
    company_total_assets: Optional[float]
    company_cash_equivalents: Optional[float]

    def __init__(self, projected_ei_targets, projected_ei_trajectories,
                       production_metric, ghg_s1s2, ghg_s3, *args, **kwargs):
        super().__init__(projected_ei_targets=UScopes_to_IScopes(projected_ei_targets),
                         projected_ei_trajectories=UScopes_to_IScopes(projected_ei_trajectories),
                         production_metric=production_metric,
                         ghg_s1s2=pint_ify(ghg_s1s2, production_metric),
                         ghg_s3=pint_ify(ghg_s3, production_metric),
                         *args, **kwargs)


class ICompanyAggregates(ICompanyData):
    cumulative_budget: Quantity['CO2']
    cumulative_trajectory: Quantity['CO2']
    cumulative_target: Quantity['CO2']
    benchmark_temperature: Quantity['delta_degC']
    benchmark_global_budget: Quantity['CO2']

    def __init__(self, cumulative_budget, cumulative_trajectory, cumulative_target, benchmark_temperature, benchmark_global_budget, *args, **kwargs):
        super().__init__(
            cumulative_budget=pint_ify(cumulative_budget, 't CO2'),
            cumulative_trajectory=pint_ify(cumulative_trajectory, 't CO2'),
            cumulative_target=pint_ify(cumulative_target, 't CO2'),
            benchmark_temperature=pint_ify(benchmark_temperature, 'delta_degC'),
            benchmark_global_budget=pint_ify(benchmark_global_budget, 'Gt CO2'),
            *args, **kwargs)


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


class EScope(SortableEnum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S1S2 = "S1+S2"
    S1S2S3 = "S1+S2+S3"

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