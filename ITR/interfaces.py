from enum import Enum
from typing import Optional, Dict, List, Literal, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError

from pint import Quantity
from ITR.data.osc_units import ureg, Q_

class AggregationContribution(BaseModel):
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
    if x is None:
        return x
    if type(x)==str:
        return ureg(x)
    if isinstance(x, Quantity):
        return x
    return Q_(x, ureg(units))


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
    units: str


BenchmarkMetric = Annotated[Union[ProductionMetric,EmissionIntensity], Field(discriminator='units')]

class IBenchmarkProjection(BaseModel):
    year: int
    value: float
    units: str


class IBenchmark(BaseModel):
    sector: str
    region: str
    benchmark_metric: BenchmarkMetric
    projections: List[IBenchmarkProjection]

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


class IYOYBenchmarkProjection(BaseModel):
    year: int
    value: float


class IYOYBenchmark(BaseModel):
    sector: str
    region: str
    projections: List[IYOYBenchmarkProjection]

    def __getitem__(self, item):
        return getattr(self, item)


class IYOYBenchmarks(BaseModel):
    benchmarks: List[IYOYBenchmark]

    def __getitem__(self, item):
        return getattr(self, item)


class IYOYBenchmarkScopes(BaseModel):
    S1S2: Optional[IYOYBenchmarks]
    S3: Optional[IYOYBenchmarks]
    S1S2S3: Optional[IYOYBenchmarks]


class IEmissionIntensityBenchmarkScopes(PintModel):
    S1S2: Optional[IBenchmarks]
    S3: Optional[IBenchmarks]
    S1S2S3: Optional[IBenchmarks]
    benchmark_temperature: Quantity['delta_degC']
    benchmark_global_budget: Quantity['CO2']
    is_AFOLU_included: bool

    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, benchmark_temperature, benchmark_global_budget, *args, **kwargs):
        super().__init__(benchmark_temperature=pint_ify(benchmark_temperature, 'delta_degC'),
                         benchmark_global_budget=pint_ify(benchmark_global_budget, 'Gt CO2'),
                         *args, **kwargs)


class ICompanyProjection(BaseModel):
    year: int
    value: Optional[float]
    units: Optional[str]    # Annotated[Union[ProductionMetric, EmissionIntensity], Field(discriminator='units')]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjections(BaseModel):
    projections: List[ICompanyProjection]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjectionsScopes(BaseModel):
    S1S2: Optional[ICompanyProjections]
    S3: Optional[ICompanyProjections]
    S1S2S3: Optional[ICompanyProjections]

    def __getitem__(self, item):
        return getattr(self, item)

class ICompanyEIProjection(PintModel):
    year: int
    value: Optional[Quantity['CO2/Wh']]

    def __init__(self, year, value):
        super().__init__(year=year, value=pint_ify(value, 't CO2/MWh'))

    def __getitem__(self, item):
        return getattr(self, item)

class IEmissionRealization(PintModel):
    year: int
    value: Optional[Quantity['CO2/Wh']]


class IHistoricEmissionsScopes(PintModel):
    S1: List[IEmissionRealization]
    S2: List[IEmissionRealization]
    S1S2: List[IEmissionRealization]
    S3: List[IEmissionRealization]
    S1S2S3: List[IEmissionRealization]


class IEIRealization(PintModel):
    year: int
    value: Optional[Quantity['CO2/Wh']]


class IHistoricEIScopes(PintModel):
    S1: List[IEIRealization]
    S2: List[IEIRealization]
    S1S2: List[IEIRealization]
    S3: List[IEIRealization]
    S1S2S3: List[IEIRealization]


class IHistoricData(PintModel):
    productions: Optional[List[IProductionRealization]]
    emissions: Optional[IHistoricEmissionsScopes]
    emission_intensities: Optional[IHistoricEIScopes]


class ICompanyData(PintModel):
    company_name: str
    company_id: str

    region: str  # TODO: make SortableEnums
    sector: str  # TODO: make SortableEnums
    target_probability: float

    historic_data: Optional[IHistoricData] = None
    projected_targets: Optional[ICompanyProjectionsScopes] = None
    projected_intensities: Optional[ICompanyProjectionsScopes] = None

    country: Optional[str]
    production_metric: ProductionMetric
    ghg_s1s2: Optional[ICompanyProjection]    # This seems to be the base year PRODUCTION number, nothing at all to do with any quantity of actual S1S2 emissions
    ghg_s3: Optional[ICompanyProjection]

    industry_level_1: Optional[str]
    industry_level_2: Optional[str]
    industry_level_3: Optional[str]
    industry_level_4: Optional[str]

    company_revenue: Optional[float]
    company_market_cap: Optional[float]
    company_enterprise_value: Optional[float]
    company_total_assets: Optional[float]
    company_cash_equivalents: Optional[float]


class ICompanyAggregates(ICompanyData):
    cumulative_budget: Quantity['CO2']
    cumulative_trajectory: Quantity['CO2']
    cumulative_target: Quantity['CO2']
    benchmark_temperature: Quantity['delta_degC']
    benchmark_global_budget: Quantity['CO2']

    def __init__(self, cumulative_budget, cumulative_trajectory, cumulative_target, benchmark_temperature, benchmark_global_budget, *args, **kwargs):
        super().__init__(cumulative_budget=pint_ify(cumulative_budget, 't CO2'),
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
