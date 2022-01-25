from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel


class AggregationContribution(BaseModel):
    company_name: str
    company_id: str
    temperature_score: float
    contribution_relative: Optional[float]
    contribution: Optional[float]

    def __getitem__(self, item):
        return getattr(self, item)


class Aggregation(BaseModel):
    score: float
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


class IBenchmarkProjection(BaseModel):
    year: int
    value: float


class IBenchmark(BaseModel):
    sector: str
    region: str
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


class IEmissionIntensityBenchmarkScopes(BaseModel):
    S1S2: Optional[IBenchmarks]
    S3: Optional[IBenchmarks]
    S1S2S3: Optional[IBenchmarks]
    benchmark_temperature: float
    benchmark_global_budget: float
    is_AFOLU_included: bool

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjection(BaseModel):
    year: int
    value: Optional[float]

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


class IProductionRealization(BaseModel):
    year: int
    value: Optional[float]


class IEmissionRealization(BaseModel):
    year: int
    value: Optional[float]


class IHistoricEmissionsScopes(BaseModel):
    S1: List[IEmissionRealization]
    S2: List[IEmissionRealization]
    S1S2: List[IEmissionRealization]
    S3: List[IEmissionRealization]
    S1S2S3: List[IEmissionRealization]


class IEIRealization(BaseModel):
    year: int
    value: Optional[float]


class IHistoricEIScopes(BaseModel):
    S1: List[IEIRealization]
    S2: List[IEIRealization]
    S1S2: List[IEIRealization]
    S3: List[IEIRealization]
    S1S2S3: List[IEIRealization]


class IHistoricData(BaseModel):
    productions: Optional[List[IProductionRealization]]
    emissions: Optional[IHistoricEmissionsScopes]
    emission_intensities: Optional[IHistoricEIScopes]


class ICompanyData(BaseModel):
    company_name: str
    company_id: str

    region: str  # TODO: make SortableEnums
    sector: str  # TODO: make SortableEnums
    target_probability: float

    historic_data: Optional[IHistoricData]
    projected_targets: Optional[ICompanyProjectionsScopes]
    projected_intensities: Optional[ICompanyProjectionsScopes]

    country: Optional[str]
    ghg_s1s2: Optional[float]
    ghg_s3: Optional[float]

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
    cumulative_budget: float
    cumulative_trajectory: float
    cumulative_target: float
    benchmark_temperature: float
    benchmark_global_budget: float


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


class TemperatureScoreControls(BaseModel):
    base_year: int
    target_end_year: int
    projection_start_year: int
    projection_end_year: int
    tcre: float
    carbon_conversion: float
    scenario_target_temperature: float

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def tcre_multiplier(self) -> float:
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
