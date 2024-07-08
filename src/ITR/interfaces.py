from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

import ITR

from .configs import LoggingConfig, ProjectionControls
from .data import PintType
from .data.osc_units import (
    PA_,
    Q_,
    BenchmarkMetric,
    BenchmarkQuantity,
    EI_Metric,
    EI_Quantity,
    EmissionsMetric,
    EmissionsQuantity,
    MonetaryQuantity,
    ProductionMetric,
    ProductionQuantity,
    Quantity,
    delta_degC_Quantity,
    percent_Quantity,
    ureg,
)

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


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
    AnyScope = "AnyScope"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @classmethod
    def get_scopes(cls) -> List[str]:
        """Get a list of all scopes.
        :return: A list of EScope string values
        """
        return ["S1", "S2", "S3", "S1S2", "S1S2S3"]

    @classmethod
    def get_result_scopes(cls) -> List["EScope"]:
        """Get a list of scopes that should be calculated if the user leaves it open.

        :return: A list of EScope objects
        """
        # FIXME: Should this also contain cls.S2 or no?
        return [cls.S1, cls.S1S2, cls.S3, cls.S1S2S3]


class ETimeFrames(SortableEnum):
    """TODO: add support for multiple timeframes. Long currently corresponds to 2050."""

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
    TARGET_ONLY = "Target only"
    COMPLETE = "Complete"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @classmethod
    def get_result_types(cls) -> List[EScoreResultType]:
        """Get a list of all result types, ordered by priority (first << last priority).
        :return: A list of the EScoreResultType values
        """
        return [
            EScoreResultType.DEFAULT,
            EScoreResultType.TRAJECTORY_ONLY,
            EScoreResultType.TARGET_ONLY,
            EScoreResultType.COMPLETE,
        ]


class AggregationContribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    company_name: str
    company_id: str
    temperature_score: delta_degC_Quantity
    contribution_relative: Optional[percent_Quantity] = Q_(np.nan, "percent")
    contribution: Optional[delta_degC_Quantity] = Q_(np.nan, "delta_degC")

    def __getitem__(self, item):
        return getattr(self, item)


class Aggregation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: delta_degC_Quantity = Q_(np.nan, "delta_degC")
    # proportion is a number from 0..1
    proportion: float = np.nan
    contributions: List[AggregationContribution] = []

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def empty(self):
        return len(self.contributions) == 0


emptyAggregation = Aggregation()


class ScoreAggregation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    all: Aggregation = emptyAggregation
    influence_percentage: percent_Quantity = Q_(np.nan, "percent")
    grouped: Dict[str, Aggregation] = {}

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def empty(self):
        return self.all.empty


emptyScoreAggregation = ScoreAggregation()


class ScoreAggregationScopes(BaseModel):
    S1: ScoreAggregation = emptyScoreAggregation
    S2: ScoreAggregation = emptyScoreAggregation
    S1S2: ScoreAggregation = emptyScoreAggregation
    S3: ScoreAggregation = emptyScoreAggregation
    S1S2S3: ScoreAggregation = emptyScoreAggregation

    def __getitem__(self, item):
        return getattr(self, item)


emptyScoreAggregationScopes = ScoreAggregationScopes()


class ScoreAggregations(BaseModel):
    short: ScoreAggregationScopes = emptyScoreAggregationScopes
    mid: ScoreAggregationScopes = emptyScoreAggregationScopes
    long: ScoreAggregationScopes = emptyScoreAggregationScopes

    def __getitem__(self, item):
        return getattr(self, item)


class PortfolioCompany(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    company_name: str
    company_id: str
    company_isin: Optional[str] = ""
    investment_value: MonetaryQuantity
    user_fields: Optional[Dict[str, str]] = {}


# U is Unquantified, which is presently how our benchmarks come in (production_metric comes in elsewhere)
class UProjection(BaseModel):
    year: int
    value: float


class IProjection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    year: int
    value: BenchmarkQuantity


class IBenchmark(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sector: str
    region: str
    benchmark_metric: BenchmarkMetric
    projections_nounits: List[UProjection]
    projections: List[IProjection]
    base_year_production: Optional[ProductionQuantity] = (
        None  # FIXME: applies only to production benchmarks
    )

    def __init__(
        self,
        benchmark_metric,
        projections_nounits=[],
        projections=[],
        base_year_production=None,
        *args,
        **kwargs,
    ):
        # FIXME: Probably want to define `target_end_year` to be 2051, not 2050...
        super().__init__(
            benchmark_metric=benchmark_metric,
            projections_nounits=projections_nounits,
            projections=projections,
            base_year_production=base_year_production,
            *args,
            **kwargs,
        )
        # Sadly we need to build the full projection range before cutting it down to size...
        # ...until Tiemann learns the bi-valence of dict and Model parameters
        if self.projections_nounits:
            if self.projections:
                # Check if we've already seen/processed these exact projections
                changed_projections = [
                    p
                    for p in self.projections
                    if not any(
                        [
                            n
                            for n in self.projections_nounits
                            if n.year == p.year and n.value == p.value.m
                        ]
                    )
                ]
                if changed_projections:
                    raise ValueError
                return
            self.projections = [
                IProjection(
                    year=p.year, value=BenchmarkQuantity(Q_(p.value, benchmark_metric))
                )
                for p in self.projections_nounits
                if p.year
                in range(
                    ProjectionControls.BASE_YEAR, ProjectionControls.TARGET_YEAR + 1
                )
            ]
        elif not self.projections:
            logger.warning(
                f"Empty Benchmark for sector {self.sector}, region {self.region}"
            )

    def __getitem__(self, item):
        return getattr(self, item)


class IBenchmarks(BaseModel):
    benchmarks: List[IBenchmark]
    production_centric: bool = False

    def __getitem__(self, item):
        return getattr(self, item)


empty_IBenchmarks = IBenchmarks(benchmarks=[], production_centric=False)

# These IProductionBenchmarkScopes and IEIBenchmarkScopes are vessels for holding initialization data
# The CompanyDataProvider methods create their own dataframes that are then used throughout


class IProductionBenchmarkScopes(BaseModel):
    AnyScope: IBenchmarks

    def __getitem__(self, item):
        return getattr(self, item)


class IEIBenchmarkScopes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    S1: Optional[IBenchmarks] = empty_IBenchmarks
    S2: Optional[IBenchmarks] = empty_IBenchmarks
    S1S2: Optional[IBenchmarks] = empty_IBenchmarks
    S3: Optional[IBenchmarks] = empty_IBenchmarks
    S1S2S3: Optional[IBenchmarks] = empty_IBenchmarks
    benchmark_temperature: delta_degC_Quantity
    benchmark_global_budget: EmissionsQuantity
    is_AFOLU_included: bool

    def __getitem__(self, item):
        return getattr(self, item)


class ICompanyEIProjection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    year: int
    value: EI_Quantity

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, o):
        if self.year != o.year:
            raise ValueError(
                f"EI Projection years not aligned for __eq__(): {self.year} vs. {o.year}"
            )
        if ITR.isna(self.value.m) and ITR.isna(o.value.m):
            return True
        return self.value == o.value

    def add(self, o):  # noqa: F811
        if self.year != o.year:
            # breakpoint()
            raise ValueError(
                f"EI Projection years not aligned for add(): {self.year} vs. {o.year}"
            )
        return ICompanyEIProjection(
            year=self.year,
            value=self.value
            if ITR.isna(o.value.m)
            else self.value + o.value.to(self.value.units),
        )

    def min(self, o):
        if self.year != o.year:
            raise ValueError(
                f"EI Projection years not aligned for min(): {self.year} vs. {o.year}"
            )
        return ICompanyEIProjection(year=self.year, value=min(self.value, o.value))


class ICompanyEIProjections(BaseModel):
    ei_metric: EI_Metric
    projections: List[ICompanyEIProjection]

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        # Work-around for https://github.com/hgrecco/pint/issues/1687
        ei_metric = ureg.parse_units(self.ei_metric)
        series = (
            lambda z: (
                idx := z[0],
                values := z[1],
                pd.Series(PA_(np.asarray(values), dtype=str(ei_metric)), index=idx),
            )[-1]
        )(
            list(
                zip(
                    *[
                        (x.year, round(ITR.Q_m_as(x.value, ei_metric), 4))
                        for x in self.projections
                    ]
                )
            )
        )
        return str(series)


class DF_ICompanyEIProjections(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ei_metric: EI_Metric
    projections: pd.Series

    @field_validator("projections")
    def val_projections(cls, v: pd.Series):
        from pint_pandas.pint_array import PintSeriesAccessor

        if isinstance(v.pint, PintSeriesAccessor):
            return v
        raise ValidationError(f"{v} is not composed of a PintArray")

    def __init__(self, icompany_ei_projections=None, *args, **kwargs):
        projections = None
        projections_gen = None
        if icompany_ei_projections is not None:
            ei_metric = icompany_ei_projections.ei_metric
            projections_gen = icompany_ei_projections.projections
        else:
            ei_metric = kwargs["ei_metric"]
            projections = kwargs["projections"]
            if not isinstance(projections, pd.Series):
                projections_gen = projections
                projections = None
        if projections_gen is not None:
            # Work-around for https://github.com/hgrecco/pint/issues/1687
            ei_metric = ureg.parse_units(ei_metric)
            years, values = list(
                map(
                    list,
                    zip(
                        *[
                            (
                                x["year"],
                                np.nan
                                if x["value"] is None
                                else ITR.Q_m_as(x["value"], ei_metric, inplace=True),
                            )
                            for x in projections_gen
                        ]
                    ),
                )
            )
            projections = pd.Series(
                PA_(np.asarray(values), dtype=str(ei_metric)),
                index=pd.Index(years, name="year"),
                name="value",
            )
        else:
            projections = projections.astype(f"pint[{str(ei_metric)}]")
        super().__init__(ei_metric=str(ei_metric), projections=projections)


class ICompanyEIProjectionsScopes(BaseModel):
    S1: Optional[DF_ICompanyEIProjections] = None
    S2: Optional[DF_ICompanyEIProjections] = None
    S1S2: Optional[DF_ICompanyEIProjections] = None
    S3: Optional[DF_ICompanyEIProjections] = None
    S1S2S3: Optional[DF_ICompanyEIProjections] = None

    def __init__(self, *args, **kwargs):
        # We don't validate anything in the first step because incoming parameters are the wild west
        # (dict, ICompanyEIProjections, pd.Series)
        super().__init__()

        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(
                    self,
                    k,
                    DF_ICompanyEIProjections(
                        ei_metric=EI_Metric(v["ei_metric"]),
                        projections=v["projections"],
                    ),
                )
            elif isinstance(v, ICompanyEIProjections):
                setattr(self, k, DF_ICompanyEIProjections(icompany_ei_projections=v))
            elif isinstance(v, pd.Series):
                ei_metric = EI_Metric(str(v.dtype))
                if ei_metric.startswith("pint["):
                    ei_metric = ei_metric[5:-1]
                setattr(
                    self,
                    k,
                    DF_ICompanyEIProjections(ei_metric=ei_metric, projections=v),
                )
            elif isinstance(v, DF_ICompanyEIProjections) or v is None:
                setattr(self, k, v)
            else:
                # breakpoint()
                assert False
            # We could do a post-hoc validation here...

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        dict_items = {
            scope: getattr(self, scope).projections
            for scope in ["S1", "S2", "S1S2", "S3", "S1S2S3"]
            if getattr(self, scope) is not None
        }
        return str(pd.DataFrame.from_dict(dict_items))

    def _adjust_trajectories(self, primary_scope_attr: str):
        if not getattr(self, primary_scope_attr):
            setattr(self, primary_scope_attr, self.S3)
        else:
            scope = getattr(self, primary_scope_attr)
            assert self.S3 is not None
            if isinstance(self.S3.projections, pd.Series):
                scope.projections = scope.projections.add(self.S3.projections)
            else:
                # Should not be reached as we are using DF_ICompanyEIProjections consistently now
                # breakpoint()
                assert False
                scope.projections = list(
                    map(
                        ICompanyEIProjection.add,
                        scope.projections,
                        self.S3.projections,
                    )
                )

    def _align_and_sum_projected_targets(self, primary_scope_attr):
        scope = getattr(self, primary_scope_attr)
        if scope is None:
            raise AttributeError
        primary_projections = scope.projections
        s3_projections = self.S3.projections
        if isinstance(s3_projections, pd.Series):
            scope.projections = (
                # We should convert S3 data from benchmark-type to disclosed-type earlier in the chain
                primary_projections + s3_projections.astype(primary_projections.dtype)
            )
        else:
            # Should not be reached as we are using DF_ICompanyEIProjections consistently now
            # breakpoint()
            assert False
            if primary_projections[0].year < s3_projections[0].year:
                while primary_projections[0].year < s3_projections[0].year:
                    primary_projections = primary_projections[1:]
            elif primary_projections[0].year > s3_projections[0].year:
                while primary_projections[0].year > s3_projections[0].year:
                    s3_projections = s3_projections[1:]
                    scope.projections = list(
                        map(
                            ICompanyEIProjection.add,
                            primary_projections,
                            s3_projections,
                        )
                    )

    @property
    def empty(self):
        return self == empty_ICompanyEIProjectionsScopes


empty_ICompanyEIProjectionsScopes = ICompanyEIProjectionsScopes()


class IProductionRealization(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    year: int
    # Need to keep this if we want to be able to read JSON files with null for value
    value: Optional[ProductionQuantity] = None


class IEmissionRealization(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    year: int
    # Need to keep this if we want to be able to read JSON files with null for value
    value: Optional[EmissionsQuantity] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, o):
        assert self.year == o.year
        if ITR.isna(self.value.m) and ITR.isna(o.value.m):
            return True
        return self.value == o.value

    def add(self, o):  # noqa: F811
        assert self.year == o.year
        return IEmissionRealization(
            year=self.year,
            value=self.value
            if ITR.isna(o.value.m)
            else self.value + o.value.to(self.value.units),
        )


class IHistoricEmissionsScopes(BaseModel):
    S1: Optional[List[IEmissionRealization]] = []
    S2: Optional[List[IEmissionRealization]] = []
    S1S2: Optional[List[IEmissionRealization]] = []
    S3: Optional[List[IEmissionRealization]] = []
    S1S2S3: Optional[List[IEmissionRealization]] = []

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        dict_items = {
            scope: (
                lambda z: (
                    idx := z[0],
                    values := z[1],
                    pd.Series(PA_(np.asarray(values), dtype="Mt CO2e"), index=idx),
                )[-1]
            )(
                list(
                    zip(
                        *[
                            (x.year, round(x.value.to("Mt CO2e").m, 4))
                            for x in getattr(self, scope)
                        ]
                    )
                )
            )
            for scope in ["S1", "S2", "S1S2", "S3", "S1S2S3"]
            if getattr(self, scope) is not None
        }
        return str(pd.DataFrame.from_dict(dict_items))

    @property
    def empty(self):
        return self == empty_IHistoricEmissionsScopes


empty_IHistoricEmissionsScopes = IHistoricEmissionsScopes()


class IEIRealization(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    year: int
    # Need to keep this if we want to be able to read JSON files with null for value
    value: Optional[EI_Quantity] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, o):
        assert self.year == o.year
        if ITR.isna(self.value.m) and ITR.isna(o.value.m):
            return True
        return self.value == o.value

    def add(self, o):  # noqa: F811
        assert self.year == o.year
        return IEIRealization(
            year=self.year,
            value=self.value
            if ITR.isna(o.value.m)
            else self.value + o.value.to(self.value.units),
        )


class IHistoricEIScopes(BaseModel):
    S1: Optional[List[IEIRealization]] = []
    S2: Optional[List[IEIRealization]] = []
    S1S2: Optional[List[IEIRealization]] = []
    S3: Optional[List[IEIRealization]] = []
    S1S2S3: Optional[List[IEIRealization]] = []

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        dict_items = {
            scope: (
                lambda z: (
                    idx := z[0],
                    values := z[1],
                    pd.Series(PA_(np.asarray(values), dtype=ei_metric), index=idx),
                )[-1]
            )(
                list(
                    zip(
                        *[
                            (x.year, round(x.value.to(ei_metric).m, 4))
                            for x in getattr(self, scope)
                        ]
                    )
                )
            )
            for scope in ["S1", "S2", "S1S2", "S3", "S1S2S3"]
            # Work-around for https://github.com/hgrecco/pint/issues/1687
            for ei_metric in [str(ureg.parse_units(getattr(self, scope).ei_metric))]
            if getattr(self, scope) is not None
        }
        return str(pd.DataFrame.from_dict(dict_items))

    @property
    def empty(self):
        return self == empty_IHistoricEIScopes


empty_IHistoricEIScopes = IHistoricEIScopes()


class IHistoricData(BaseModel):
    productions: List[IProductionRealization]
    emissions: IHistoricEmissionsScopes
    emissions_intensities: IHistoricEIScopes

    def __init__(
        self,
        productions=[],
        emissions=empty_IHistoricEmissionsScopes,
        emissions_intensities=empty_IHistoricEIScopes,
        *args,
        **kwargs,
    ):
        # Tolerate `null` values in JSON files that we don't tolerate within our programs
        super().__init__(
            productions=productions,
            emissions=emissions or empty_IHistoricEmissionsScopes,
            emissions_intensities=emissions_intensities or empty_IHistoricEIScopes,
            *args,
            **kwargs,
        )

    def _normalize(
        self, production_metric: ProductionMetric, emissions_metric: EmissionsMetric
    ) -> None:
        def _normalize_qty(value, metric) -> Quantity:
            if value is None or ITR.isna(value):
                return PintType(metric).na_value
            if value.u == metric:
                return value
            # We've pre-conditioned metric so don't need to work around https://github.com/hgrecco/pint/issues/1687
            return value.to(metric)

        production_metric = ureg.parse_units(production_metric)  # Catch things like '$'
        self.productions = [
            IProductionRealization(
                year=p.year, value=_normalize_qty(p.value, production_metric)
            )
            for p in self.productions
        ]

        # Work-around for https://github.com/hgrecco/pint/issues/1687
        # emissions_metric = ureg(emissions_metric).u
        ei_metric = ureg.parse_units(f"{emissions_metric} / ({production_metric})")
        for scope_name in EScope.get_scopes():
            setattr(
                self.emissions,
                scope_name,
                [
                    IEmissionRealization(
                        year=p.year, value=_normalize_qty(p.value, emissions_metric)
                    )
                    for p in self.emissions[scope_name]
                ],
            )
            setattr(
                self.emissions_intensities,
                scope_name,
                [
                    IEIRealization(
                        year=p.year, value=_normalize_qty(p.value, ei_metric)
                    )
                    for p in self.emissions_intensities[scope_name]
                ],
            )

    @property
    def empty(self) -> bool:
        if self.productions:
            return False
        for scope_name in EScope.get_scopes():
            if getattr(self.emissions, scope_name, None):
                return False
            if getattr(self.emissions_intensities, scope_name, None):
                return False
        return True


class ITargetData(BaseModel):
    netzero_year: Optional[int]
    target_type: Union[
        Literal["intensity"],
        Literal["absolute"],
        Literal["Intensity"],
        Literal["Absolute"],
    ]
    target_scope: EScope
    target_start_year: Optional[int] = None
    target_base_year: int
    target_end_year: int

    target_base_year_qty: float
    target_base_year_err: Optional[float] = None
    target_base_year_unit: str
    target_reduction_pct: float  # This is actually a fraction, not a percentage.  1.0 = complete reduction to zero.

    @model_validator(mode="before")
    def start_end_base_order(cls, v):
        if v["target_start_year"] < v["target_base_year"]:
            raise ValueError(
                f"Scope {v['target_scope']}: Target start year ({v['target_start_year']}) must be equal or greater than base year {v['target_base_year']}"
            )
        if v["target_end_year"] <= v["target_base_year"]:
            raise ValueError(
                f"Scope {v['target_scope']}: Target end year ({v['target_end_year']}) must be greater than base year {v['target_base_year']}"
            )
        if v["target_end_year"] <= v["target_start_year"]:
            raise ValueError(
                f"Scope {v['target_scope']}: Target end year ({v['target_end_year']}) must be greater than start year {v['target_start_year']}"
            )
        return v


class ICompanyData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    company_name: str
    company_id: str

    sector: str  # TODO: make SortableEnums
    region: str  # TODO: make SortableEnums
    # TemperatureScoreConfig.CONTROLS_CONFIG.target_probability is not company-specific,
    # while target_probability in ICompanyData is company-specific
    target_probability: float = np.nan

    target_data: Optional[List[ITargetData]] = []
    # IHistoric data can contain None values; need to convert to Quantified NaNs
    historic_data: IHistoricData

    country: Optional[str] = None

    # Emissions typically use t CO2 for MWh/GJ and Mt CO2 for TWh/PJ
    emissions_metric: Optional[EmissionsMetric] = None
    production_metric: Optional[ProductionMetric] = None

    # These three instance variables match against financial data below, but are incomplete as historic_data and target_data
    base_year_production: Optional[ProductionQuantity] = None
    ghg_s1s2: Optional[EmissionsQuantity] = None
    ghg_s3: Optional[EmissionsQuantity] = None

    industry_level_1: Optional[str] = None
    industry_level_2: Optional[str] = None
    industry_level_3: Optional[str] = None
    industry_level_4: Optional[str] = None

    company_revenue: Optional[MonetaryQuantity] = None
    company_market_cap: Optional[MonetaryQuantity] = None
    company_enterprise_value: Optional[MonetaryQuantity] = None
    company_ev_plus_cash: Optional[MonetaryQuantity] = None
    company_total_assets: Optional[MonetaryQuantity] = None
    company_cash_equivalents: Optional[MonetaryQuantity] = None

    # Initialized later when we have benchmark information.  It is OK to initialize as None and fix later.
    # They will show up as {'S1S2': { 'projections': [ ... ] }}
    projected_targets: ICompanyEIProjectionsScopes
    projected_intensities: ICompanyEIProjectionsScopes

    # TODO: Do we want to do some sector inferencing here?

    def _sector_to_production_units(self, sector, region="Global"):
        sector_unit_dict = {
            "Electricity Utilities": {"North America": "MWh", "Global": "GJ"},
            "Gas Utilities": {"Global": "PJ"},
            "Utilities": {"Global": "PJ"},
            "Steel": {"Global": "t Steel"},
            "Aluminum": {"Global": "t Aluminum"},
            "Energy": {"Global": "PJ"},
            "Coal": {"Global": "t Coal"},
            "Oil": {"Global": "bbl/d"},
            "Gas": {"Global": "bcm"},
            "Oil & Gas": {"Global": "PJ"},
            "Autos": {"Global": "pkm"},
            "Trucking": {"Global": "tkm"},
            "Cement": {"Global": "t Cement"},
            "Construction Buildings": {"Global": "billion USD"},
            "Residential Buildings": {
                "Global": "billion m**2"
            },  # Should it be 'built m**2' ?
            "Commercial Buildings": {
                "Global": "billion m**2"
            },  # Should it be 'built m**2' ?
            "Textiles": {"Global": "billion USD"},
            "Chemicals": {"Global": "billion USD"},
            "Pharmaceuticals": {"Global": "billion USD"},
            "Ag Chem": {"Global": "billion USD"},
            "Consumer Products": {"Global": "billion USD"},
            "Fiber & Rubber": {"Global": "billion USD"},
            "Petrochem & Plastics": {"Global": "billion USD"},
        }
        units = None
        if sector_unit_dict.get(sector):
            region_unit_dict = sector_unit_dict[sector]
            if region_unit_dict.get(region):
                units = region_unit_dict[region]
            else:
                units = region_unit_dict["Global"]
        else:
            raise ValueError(f"No source of production metrics for {self.company_name}")
        return units

    def _get_base_realization_from_historic(
        self, realized_values: List[BaseModel], metric, base_year=None
    ):
        valid_realizations = [rv for rv in realized_values if not ITR.isna(rv.value)]
        if not valid_realizations:
            retval = realized_values[0].model_copy()
            retval.year = None
            return retval
        valid_realizations.sort(key=lambda x: x.year, reverse=True)
        if base_year and valid_realizations[0].year != base_year:
            retval = realized_values[0].copy()
            retval.year = base_year
            # FIXME: Unless and until we accept uncertainties as input, rather than computed data, we don't need to make this a UFloat here
            retval.value = PintType(metric).na_value
            return retval
        return valid_realizations[0]

    def __init__(
        self,
        emissions_metric=None,
        production_metric=None,
        base_year_production=None,
        ghg_s1s2=None,
        ghg_s3=None,
        target_data=None,
        historic_data=None,
        projected_targets=empty_ICompanyEIProjectionsScopes,
        projected_intensities=empty_ICompanyEIProjectionsScopes,
        *args,
        **kwargs,
    ):
        super().__init__(
            emissions_metric=emissions_metric,
            production_metric=production_metric,
            base_year_production=base_year_production,
            ghg_s1s2=ghg_s1s2,
            ghg_s3=ghg_s3,
            target_data=target_data,
            historic_data=historic_data or IHistoricData(),
            projected_targets=projected_targets or empty_ICompanyEIProjectionsScopes,
            projected_intensities=projected_intensities
            or empty_ICompanyEIProjectionsScopes,
            *args,
            **kwargs,
        )
        # In-bound parameters are JSON (str, int, float, dict), which are converted to models by __super__ and stored as instance variables
        if production_metric is None:
            units = self._sector_to_production_units(self.sector, self.region)
            self.production_metric = ProductionMetric(units)
            if emissions_metric is None:
                self.emissions_metric = EmissionsMetric("t CO2")
        elif emissions_metric is None:
            if str(self.production_metric) in [
                "TWh",
                "PJ",
                "Mt Steel",
                "megaFe_ton",
                "mmboe",
            ]:
                self.emissions_metric = EmissionsMetric("Mt CO2")
            else:
                self.emissions_metric = EmissionsMetric("t CO2")
            # TODO: Should raise a warning here

        if self.historic_data.empty:
            # We are only partly initialized.  Remaining will be done later
            return
        self.historic_data._normalize(self.production_metric, self.emissions_metric)
        base_year = None
        if self.base_year_production:
            pass
        # Right now historic_data comes in via template.py ESG data
        elif self.historic_data.productions:
            # TODO: This is a hack to get things going.
            base_realization = self._get_base_realization_from_historic(
                self.historic_data.productions, self.production_metric, base_year
            )
            base_year = base_realization.year
            self.base_year_production = base_realization.value
        else:
            logger.warning(
                f"missing historic data for base_year_production for {self.company_name}"
            )
            self.base_year_production = PintType(self.production_metric).na_value
        if self.ghg_s1s2 is None and (
            self.historic_data.emissions.S1S2 or self.historic_data.emissions.S1
        ):
            if self.historic_data.emissions.S1S2:
                base_realization = self._get_base_realization_from_historic(
                    self.historic_data.emissions.S1S2, self.emissions_metric, base_year
                )
                base_year = base_year or base_realization.year
                self.ghg_s1s2 = base_realization.value
            elif self.historic_data.emissions.S1 and self.historic_data.emissions.S2:
                base_realization_s1 = self._get_base_realization_from_historic(
                    self.historic_data.emissions.S1, self.emissions_metric, base_year
                )
                base_realization_s2 = self._get_base_realization_from_historic(
                    self.historic_data.emissions.S2, self.emissions_metric, base_year
                )
                base_year = base_year or base_realization_s1.year
                if (
                    base_realization_s1.value is not None
                    and base_realization_s2.value is not None
                ):
                    self.ghg_s1s2 = (
                        base_realization_s1.value + base_realization_s2.value
                    )
        if self.ghg_s1s2 is None and (
            self.historic_data.emissions_intensities.S1S2
            or self.historic_data.emissions_intensities.S1
        ):
            intensity_metric = ureg.parse_units(
                f"({self.emissions_metric}) / ({self.production_metric})"
            )
            if self.historic_data.emissions_intensities.S1S2:
                base_realization = self._get_base_realization_from_historic(
                    self.historic_data.emissions_intensities.S1S2,
                    intensity_metric,
                    base_year,
                )
                base_year = base_year or base_realization.year
                if base_realization.value is not None:
                    self.ghg_s1s2 = base_realization.value * self.base_year_production
            elif (
                self.historic_data.emissions_intensities.S1
                and self.historic_data.emissions_intensities.S2
            ):
                base_realization_s1 = self._get_base_realization_from_historic(
                    self.historic_data.emissions_intensities.S1,
                    intensity_metric,
                    base_year,
                )
                base_realization_s2 = self._get_base_realization_from_historic(
                    self.historic_data.emissions_intensities.S2,
                    intensity_metric,
                    base_year,
                )
                base_year = base_year or base_realization_s1.year
                if (
                    base_realization_s1.value is not None
                    and base_realization_s2.value is not None
                ):
                    self.ghg_s1s2 = (
                        base_realization_s1.value + base_realization_s2.value
                    ) * self.base_year_production
            else:
                raise ValueError(
                    f"missing S1S2 historic intensity data for {self.company_name}"
                )
        if self.ghg_s1s2 is None:
            raise ValueError(
                f"missing historic emissions or intensity data to calculate ghg_s1s2 for {self.company_name}"
            )
        if self.ghg_s3 is None and self.historic_data.emissions.S3:
            base_realization_s3 = self._get_base_realization_from_historic(
                self.historic_data.emissions.S3, self.emissions_metric, base_year
            )
            self.ghg_s3 = base_realization_s3.value
        if self.ghg_s3 is None and self.historic_data.emissions_intensities:
            if self.historic_data.emissions_intensities.S3:
                intensity_metric = ureg.parse_units(
                    f"({self.emissions_metric}) / ({self.production_metric})"
                )
                base_realization_s3 = self._get_base_realization_from_historic(
                    self.historic_data.emissions_intensities.S3,
                    intensity_metric,
                    base_year,
                )
                if base_realization_s3.value is not None:
                    self.ghg_s3 = base_realization_s3.value * self.base_year_production


# These aggregate terms are all derived from the benchmark being used
class ICompanyAggregates(ICompanyData):
    cumulative_budget: Optional[EmissionsQuantity] = None
    cumulative_scaled_budget: Optional[EmissionsQuantity] = None
    cumulative_trajectory: Optional[EmissionsQuantity] = None
    cumulative_target: Optional[EmissionsQuantity] = None
    benchmark_temperature: Optional[delta_degC_Quantity] = None
    benchmark_global_budget: Optional[EmissionsQuantity] = None
    scope: Optional[EScope] = None

    # The first year that cumulative_projections exceeds the 2050 cumulative_budget
    trajectory_exceedance_year: Optional[int] = None
    target_exceedance_year: Optional[int] = None

    # projected_production is computed but never saved, so computed at least 2x: initialiation/projection and cumulative budget
    # projected_targets: ICompanyEIProjectionsScopes
    # projected_intensities: ICompanyEIProjectionsScopes

    # Custom validator here
    @field_validator("trajectory_exceedance_year", "target_exceedance_year")
    def val_exceedance_year(cls, v):
        if isinstance(v, int):
            return v
        if pd.isna(v):
            return None
        raise ValueError(f"{v} is not compatible with Int64 dtype")

    @classmethod
    def from_ICompanyData(cls, super_instance, scope_company_data):
        """Fast way to add instance variables to a pre-validated SUPER_INSTANCE
        SCOPE_COMPANY_DATA is the dictionary of the new values we want to add...for this one company
        """
        # FIXME: Would love to know how to run these automatically...
        EmissionsQuantity(scope_company_data["cumulative_budget"])
        EmissionsQuantity(scope_company_data["cumulative_scaled_budget"])
        if not ITR.isna(scope_company_data["cumulative_trajectory"]):
            EmissionsQuantity(scope_company_data["cumulative_trajectory"])
        if not ITR.isna(scope_company_data["cumulative_target"]):
            EmissionsQuantity(scope_company_data["cumulative_target"])
        if not Q_(scope_company_data["benchmark_temperature"]).is_compatible_with(
            ureg("delta_degC")
        ):
            raise ValueError(
                f"benchmark temperature {scope_company_data['benchmark_temperature']} is not compatible with delta_degC"
            )
        else:
            scope_company_data["benchmark_temperature"] = Q_(
                scope_company_data["benchmark_temperature"]
            )
        EmissionsQuantity(scope_company_data["benchmark_global_budget"])
        if not isinstance(scope_company_data["scope"], EScope):
            raise ValueError(
                f"scope {scope_company_data['scope']} is not a valid scope"
            )
        if not ITR.isna(
            scope_company_data["trajectory_exceedance_year"]
        ) and not isinstance(scope_company_data["trajectory_exceedance_year"], int):
            raise ValueError(
                f"scope {scope_company_data['trajectory_exceedance_year']} is not a valid trajectory exceedance year value"
            )
        if not ITR.isna(
            scope_company_data["target_exceedance_year"]
        ) and not isinstance(scope_company_data["target_exceedance_year"], int):
            raise ValueError(
                f"scope {scope_company_data['target_exceedance_year']} is not a valid target exceedance year value"
            )
        # ...while not re-running any validation on super_instnace
        return cls.model_construct(**scope_company_data, **super_instance.__dict__)
