from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel
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


def pint_ify(x, units):
    if x is None:
        return x
    if type(x)==str:
        return ureg(x)
    return Q_(x, ureg(units))


class IBenchmarkProjection(PintModel):
    year: int
    value: Quantity['Wh']

    def __init__(self, year, value):
        super().__init__(year=year, value=pint_ify(value, 'MWh'))


class IEIBenchmarkProjection(PintModel):
    year: int
    value: Quantity['CO2/Wh']

    def __init__(self, year, value):
        super().__init__(year=year, value=pint_ify(value, 't CO2/MWh'))


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


class IYOYBenchmarkProjection(PintModel):
    year: int
    value: float


class IYOYBenchmark(PintModel):
    sector: str
    region: str
    projections: List[IYOYBenchmarkProjection]

    def __getitem__(self, item):
        return getattr(self, item)


class IYOYBenchmarks(PintModel):
    benchmarks: List[IYOYBenchmark]

    def __getitem__(self, item):
        return getattr(self, item)


class IYOYBenchmarkScopes(PintModel):
    S1S2: Optional[IYOYBenchmarks]
    S3: Optional[IYOYBenchmarks]
    S1S2S3: Optional[IYOYBenchmarks]


class IEIBenchmark(PintModel):
    sector: str
    region: str
    projections: List[IEIBenchmarkProjection]

    def __getitem__(self, item):
        return getattr(self, item)


class IEIBenchmarks(PintModel):
    benchmarks: List[IEIBenchmark]

    def __getitem__(self, item):
        return getattr(self, item)


class IEmissionIntensityBenchmarkScopes(PintModel):
    S1S2: Optional[IEIBenchmarks]
    S3: Optional[IEIBenchmarks]
    S1S2S3: Optional[IEIBenchmarks]
    benchmark_temperature: Quantity['delta_degC']
    benchmark_global_budget: Quantity['CO2']
    is_AFOLU_included: bool

    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, benchmark_temperature, benchmark_global_budget, *args, **kwargs):
        super().__init__(benchmark_temperature=pint_ify(benchmark_temperature, 'delta_degC'),
                         benchmark_global_budget=pint_ify(benchmark_global_budget, 'Gt CO2'),
                         *args, **kwargs)


class ICompanyProjection(PintModel):
    year: int
    value: Optional[Quantity['Wh']]

    def __init__(self, year, value):
        super().__init__(year=year, value=pint_ify(value, 'MWh'))

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyProjections(PintModel):
    projections: List[ICompanyProjection]

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

    historic_data: Optional[IHistoricData]
    projected_targets: Optional[ICompanyProjectionsScopes] = None
    projected_intensities: Optional[ICompanyProjectionsScopes] = None

    country: Optional[str]
    ghg_s1s2: Optional[Quantity['Wh']]    # This seems to be the base year PRODUCTION number, nothing at all to do with any quantity of actual S1S2 emissions
    ghg_s3: Optional[Quantity['Wh']]

    industry_level_1: Optional[str]
    industry_level_2: Optional[str]
    industry_level_3: Optional[str]
    industry_level_4: Optional[str]

    company_revenue: Optional[float]
    company_market_cap: Optional[float]
    company_enterprise_value: Optional[float]
    company_total_assets: Optional[float]
    company_cash_equivalents: Optional[float]

    def __init__(self, ghg_s1s2, ghg_s3, *args, **kwargs):
        super().__init__(ghg_s1s2=pint_ify(ghg_s1s2, 'MWh'), ghg_s3=pint_ify(ghg_s3, 'MWh'), *args, **kwargs)

class ICompanyAggregates(ICompanyData):
    cumulative_budget: Quantity['CO2/Wh']
    cumulative_trajectory: Quantity['CO2/Wh']
    cumulative_target: Quantity['CO2/Wh']
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
