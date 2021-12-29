from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel
from pint import Quantity

class PintModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

class AggregationContribution(PintModel):
    company_name: str
    company_id: str
    temperature_score: Quantity['degC']
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


class ScoreAggregation(PintModel):
    all: Aggregation
    influence_percentage: float
    grouped: Dict[str, Aggregation]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregationScopes(PintModel):
    S1S2: Optional[ScoreAggregation]
    S3: Optional[ScoreAggregation]
    S1S2S3: Optional[ScoreAggregation]

    def __getitem__(self, item):
        return getattr(self, item)


class ScoreAggregations(PintModel):
    short: Optional[ScoreAggregationScopes]
    mid: Optional[ScoreAggregationScopes]
    long: Optional[ScoreAggregationScopes]

    def __getitem__(self, item):
        return getattr(self, item)


class PortfolioCompany(PintModel):
    company_name: str
    company_id: str
    company_isin: Optional[str]
    investment_value: float
    user_fields: Optional[dict]


class IBenchmarkProjection(PintModel):
    year: int
    value: Quantity['CO2/Wh']


class IBenchmark(PintModel):
    sector: str
    region: str
    projections: List[IBenchmarkProjection]

    def __getitem__(self, item):
        return getattr(self, item)


class IBenchmarks(PintModel):
    benchmarks: List[IBenchmark]

    def __getitem__(self, item):
        return getattr(self, item)

class IProductionBenchmarkScopes(PintModel):
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

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjection(PintModel):
    year: int
    value: Optional[Quantity['CO2']]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjections(PintModel):
    projections: Optional[List[ICompanyProjection]]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjectionsScopes(PintModel):
    S1S2: Optional[ICompanyProjections]
    S3: Optional[ICompanyProjections]
    S1S2S3: Optional[ICompanyProjections]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjection(PintModel):
    year: int
    value: Optional[Quantity['CO2/Wh']]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjections(PintModel):
    projections: Optional[List[ICompanyEIProjection]]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjectionsScopes(PintModel):
    S1S2: Optional[ICompanyEIProjections]
    S3: Optional[ICompanyEIProjections]
    S1S2S3: Optional[ICompanyEIProjections]

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyData(PintModel):
    company_name: str
    company_id: str

    region: str  # TODO: make SortableEnums
    sector: str  # TODO: make SortableEnums
    target_probability: float

    projected_targets: Optional[ICompanyEIProjectionsScopes]
    projected_intensities: Optional[ICompanyEIProjectionsScopes]

    country: Optional[str]
    ghg_s1s2: Optional[Quantity['CO2']]
    ghg_s3: Optional[Quantity['CO2']]

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