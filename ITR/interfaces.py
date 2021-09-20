from enum import Enum
from typing import Optional, Dict, List
import pandas as pd
from pydantic import BaseModel, validator


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


class ICompanyData(BaseModel):
    company_name: str
    company_id: str
    isic: str
    ghg_s1s2: float
    ghg_s3: float
    region: str  # TODO: make SortableEnums
    sector: str  # TODO: make SortableEnums
    target_probability: float

    country: Optional[str]

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
    global_budget : int  # TODO: is dependent on BASE_YEAR - fix
    current_temperature: float
    energy_unit_conversion_factor: float

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
