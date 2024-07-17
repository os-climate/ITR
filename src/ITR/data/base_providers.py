import logging
import warnings  # needed until quantile behaves better with Pint quantities in arrays
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Optional, Type, cast

import numpy as np
import pandas as pd
from pint import DimensionalityError

import ITR

from ..configs import ColumnsConfig, LoggingConfig, ProjectionControls, VariablesConfig
from ..data import PA_, Q_, PintType, ureg
from ..data.data_providers import (
    CompanyDataProvider,
    IntensityBenchmarkDataProvider,
    ProductionBenchmarkDataProvider,
)
from ..data.osc_units import (
    Quantity,
    align_production_to_bm,
    asPintDataFrame,
    asPintSeries,
)
from ..interfaces import (
    DF_ICompanyEIProjections,
    EI_Quantity,
    EScope,
    IBenchmark,
    ICompanyData,
    ICompanyEIProjection,
    ICompanyEIProjections,
    ICompanyEIProjectionsScopes,
    IEIBenchmarkScopes,
    IEIRealization,
    IEmissionRealization,
    IHistoricData,
    IHistoricEIScopes,
    IHistoricEmissionsScopes,
    IProductionBenchmarkScopes,
    IProductionRealization,
    ITargetData,
)

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


# TODO handling of scopes in benchmarks


# The benchmark projected production format is based on year-over-year growth and starts out like this:

#                                                2019     2020            2049       2050
# region                 sector        scope                    ...
# Steel                  Global        AnyScope   0.0   0.00306  ...     0.0155     0.0155
#                        Europe        AnyScope   0.0   0.00841  ...     0.0155     0.0155
#                        North America AnyScope   0.0   0.00748  ...     0.0155     0.0155
# Electricity Utilities  Global        AnyScope   0.0    0.0203  ...     0.0139     0.0139
#                        Europe        AnyScope   0.0    0.0306  ...   -0.00113   -0.00113
#                        North America AnyScope   0.0    0.0269  ...   0.000426   0.000426
# etc.

# To compute the projected production for a company in given sector/region, we need to start with the
# base_year_production for that company and apply the year-over-year changes projected by the benchmark
# until all years are computed.  We need to know production of each year, not only the final year
# because the cumumulative emissions of the company will be the sum of the emissions of each year,
# which depends on both the production projection (computed here) and the emissions intensity projections
# (computed elsewhere).

# Let Y2019 be the production of a company in 2019.
# Y2020 = Y2019 + (Y2019 * df_pp[2020]) = Y2019 + Y2019 * (1.0 + df_pp[2020])
# Y2021 = Y2020 + (Y2020 * df_pp[2020]) = Y2020 + Y2020 * (1.0 + df_pp[2021])
# etc.

# The Pandas `cumprod` function calculates precisely the cumulative product we need
# As the math shows above, the terms we need to accumulate are 1.0 + growth.

# df.add(1).cumprod(axis=1).astype('pint[dimensionless]') results in a project that looks like this:
#
#                                                2019     2020  ...      2049      2050
# region                 sector        scope                    ...
# Steel                  Global        AnyScope   1.0  1.00306  ...  1.419076  1.441071
#                        Europe        AnyScope   1.0  1.00841  ...  1.465099  1.487808
#                        North America AnyScope   1.0  1.00748  ...  1.457011  1.479594
# Electricity Utilities  Global        AnyScope   1.0  1.02030  ...  2.907425  2.947838
#                        Europe        AnyScope   1.0  1.03060  ...  1.751802  1.749822
#                        North America AnyScope   1.0  1.02690  ...  2.155041  2.155959
# etc.


class BaseProviderProductionBenchmark(ProductionBenchmarkDataProvider):
    def __init__(
        self,
        production_benchmarks: IProductionBenchmarkScopes,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
    ):
        """Base provider that relies on pydantic interfaces. Default for FastAPI usage
        :param production_benchmarks: List of IProductionBenchmarkScopes
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        super().__init__()
        self.column_config = column_config
        self._productions_benchmarks = production_benchmarks
        self._own_data = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _prod_delta_df_t = pd.concat(
                    [
                        self._convert_benchmark_to_series(bm, EScope.AnyScope).pint.m
                        for bm in self._productions_benchmarks[
                            EScope.AnyScope.name
                        ].benchmarks
                    ],
                    axis=1,
                )
        except AttributeError:
            assert False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
            # See comment above to understand use of `cumprod` function
            self._prod_df = (
                _prod_delta_df_t.add(1.0)
                .cumprod(axis=0)
                .astype("pint[dimensionless]")
                .T
            )
        self._prod_df.columns.name = "year"
        self._prod_df.index.names = [
            self.column_config.SECTOR,
            self.column_config.REGION,
            self.column_config.SCOPE,
        ]

    def benchmark_changed(
        self, new_projected_production: ProductionBenchmarkDataProvider
    ) -> bool:
        assert isinstance(new_projected_production, BaseProviderProductionBenchmark)
        return (
            self._productions_benchmarks
            != new_projected_production._productions_benchmarks
        )

    # Note that benchmark production series are dimensionless.
    # FIXME: They also don't need a scope.  Remove scope when we change IBenchmark format...
    def _convert_benchmark_to_series(
        self, benchmark: IBenchmark, scope: EScope
    ) -> pd.Series:
        """Extracts the company projected intensity or production targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        # Benchmarks don't need work-around for https://github.com/hgrecco/pint/issues/1687, but if they did:
        # units = ureg.parse_units(benchmark.benchmark_metric)
        years, values = list(
            map(list, zip(*{r.year: r.value.m for r in benchmark.projections}.items()))
        )
        return pd.Series(
            PA_(np.array(values), dtype="pint[dimensionless]"),
            index=years,
            name=(benchmark.sector, benchmark.region, scope),
        )

    # Production benchmarks are dimensionless, relevant for AnyScope
    def _get_projected_production(
        self, scope: EScope = EScope.AnyScope
    ) -> pd.DataFrame:
        """Converts IProductionBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: a pint[dimensionless] pd.DataFrame
        """
        return self._prod_df

        # The call to this function generates a 42-row (and counting...) DataFrame for the one row we're going to end up needing...
        df_bm_t = pd.concat(
            [
                self._convert_benchmark_to_series(bm, scope).pint.m
                for bm in self._productions_benchmarks[scope.name].benchmarks
            ],
            axis=1,
        )

        df_partial_pp = df_bm_t.add(1.0).cumprod(axis=0).astype("pint[dimensionless]").T
        df_partial_pp.index.names = [
            self.column_config.SECTOR,
            self.column_config.REGION,
            self.column_config.SCOPE,
        ]

        return df_partial_pp

    def get_company_projected_production(
        self, company_sector_region_scope: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the projected productions for list of companies
        :param company_sector_region_scope: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE
        :return: DataFrame of projected productions for [base_year through 2050]
        """
        from ..utils import get_benchmark_projections

        company_benchmark_projections = get_benchmark_projections(
            self._prod_df, company_sector_region_scope
        )
        company_production = company_sector_region_scope.set_index(
            self.column_config.SCOPE, append=True
        )[self.column_config.BASE_YEAR_PRODUCTION]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We have to use lambda function here because company_production is heterogeneous, not a PintArray
            nan_production = company_production.map(lambda x: ITR.isna(x))
            if nan_production.any():
                # If we don't have valid production data for base year, we get back a nan result that's a pain to debug, so nag here
                logger.error(
                    f"these companies are missing production data: {nan_production[nan_production].index.get_level_values(0).to_list()}"
                )
            # We transpose the operation so that Pandas is happy to preserve the dtype integrity of the column
            company_projected_productions_t = company_benchmark_projections.T.mul(
                company_production, axis=1
            )
            return company_projected_productions_t.T


class BaseProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(
        self,
        EI_benchmarks: IEIBenchmarkScopes,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
    ):
        super().__init__(
            EI_benchmarks.benchmark_temperature,
            EI_benchmarks.benchmark_global_budget,
            EI_benchmarks.is_AFOLU_included,
        )
        self._own_data = True
        self._EI_benchmarks = EI_benchmarks
        self.column_config = column_config
        self.projection_controls = projection_controls
        benchmarks_as_series = []
        for scope_name in EScope.get_scopes():
            try:
                for bm in EI_benchmarks[scope_name].benchmarks:
                    benchmarks_as_series.append(
                        self._convert_benchmark_to_series(bm, EScope[scope_name])
                    )
            except AttributeError:
                pass

        self._EI_df_t = pd.concat(benchmarks_as_series, axis=1)
        self._EI_df_t.index.name = "year"
        self._EI_df_t.columns.set_names(["sector", "region", "scope"], inplace=True)
        # https://stackoverflow.com/a/56528071/1291237
        self._EI_df_t.sort_index(axis=1, inplace=True)

    def get_scopes(self) -> List[EScope]:
        scopes = [
            scope
            for scope in EScope.get_result_scopes()
            if getattr(self._EI_benchmarks, scope.name)
            != ITR.interfaces.empty_IBenchmarks
        ]
        return scopes

    def benchmarks_changed(
        self, new_projected_ei: IntensityBenchmarkDataProvider
    ) -> bool:
        assert isinstance(new_projected_ei, BaseProviderIntensityBenchmark)
        return self._EI_benchmarks != new_projected_ei._EI_benchmarks

    def prod_centric_changed(
        self, new_projected_ei: IntensityBenchmarkDataProvider
    ) -> bool:
        prev_prod_centric = next_prod_centric = False
        if getattr(self._EI_benchmarks, "S1S2", None):
            prev_prod_centric = self._EI_benchmarks["S1S2"].production_centric
        assert isinstance(new_projected_ei, BaseProviderIntensityBenchmark)
        if getattr(new_projected_ei._EI_benchmarks, "S1S2", None):
            next_prod_centric = new_projected_ei._EI_benchmarks[
                "S1S2"
            ].production_centric
        return prev_prod_centric != next_prod_centric

    def is_production_centric(self) -> bool:
        """Returns True if benchmark is "production_centric" (as defined by OECM)"""
        if getattr(self._EI_benchmarks, "S1S2", None):
            return self._EI_benchmarks["S1S2"].production_centric
        return False

    # SDA stands for Sectoral Decarbonization Approach; see https://sciencebasedtargets.org/resources/files/SBTi-Power-Sector-15C-guide-FINAL.pdf
    def get_SDA_intensity_benchmarks(
        self,
        company_info_at_base_year: pd.DataFrame,
        scope_to_calc: Optional[EScope] = None,
    ) -> pd.DataFrame:
        """Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_info_at_base_year: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        # To make pint happier, we do our math in columns that can be represented by PintArrays
        intensity_benchmarks_t = self._get_intensity_benchmarks(
            company_info_at_base_year, scope_to_calc
        )
        decarbonization_paths_t = self._get_decarbonizations_paths(
            intensity_benchmarks_t
        )
        last_ei = intensity_benchmarks_t.loc[self.projection_controls.TARGET_YEAR]
        ei_base = intensity_benchmarks_t.loc[self.projection_controls.BASE_YEAR]
        df_t = decarbonization_paths_t.mul((ei_base - last_ei), axis=1)
        df_t = df_t.add(last_ei, axis=1)
        df_t.index.name = "year"
        idx = pd.Index.intersection(
            df_t.columns,
            pd.MultiIndex.from_arrays(
                [company_info_at_base_year.index, company_info_at_base_year.scope]
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pint units don't like being twisted from columns to rows, but it's ok
            df = df_t[idx].T
        return df

    def _get_decarbonizations_paths(
        self, intensity_benchmarks_t: pd.DataFrame
    ) -> pd.DataFrame:
        """Overrides subclass method
        Returns a DataFrame with the projected decarbonization paths for the supplied companies in intensity_benchmarks.
        :param: A DataFrame with company and intensity benchmarks per calendar year per row
        :return: A pd.DataFrame with company and decarbonisation path s per calendar year per row
        """
        return intensity_benchmarks_t.apply(lambda col: self._get_decarbonization(col))

    def _get_decarbonization(self, intensity_benchmark_ser: pd.Series) -> pd.Series:
        """Overrides subclass method
        returns a Series with the decarbonization path for a benchmark.
        :param: A Series with a company's intensity benchmarks per calendar year per row
        :return: A pd.Series with a company's decarbonisation paths per calendar year per row
        """
        ei_units = intensity_benchmark_ser.dtype.units
        last_ei = intensity_benchmark_ser[self.projection_controls.TARGET_YEAR].to(
            ei_units
        )
        ei_diff = (
            intensity_benchmark_ser[self.projection_controls.BASE_YEAR] - last_ei
        ).to(ei_units)
        # We treat zero divided by zero as zero, not NaN.
        # Because our starting units are homogeneous and our target units are dimensionless, we do our math with magnitudes only.
        numerator_m = intensity_benchmark_ser.pint.m - last_ei.m
        mask = numerator_m == 0.0
        decarb_m = numerator_m.where(mask, numerator_m / ei_diff.m)
        return decarb_m.astype("pint[dimensionless]")

    def _convert_benchmark_to_series(
        self, benchmark: IBenchmark, scope: EScope
    ) -> pd.Series:
        """Extracts the company projected intensities or targets for a given scope
        :param scope: a scope
        :return: pd.Series
        """
        s = pd.Series(
            {
                p.year: p.value
                for p in benchmark.projections
                if p.year
                in range(
                    self.projection_controls.BASE_YEAR,
                    self.projection_controls.TARGET_YEAR + 1,
                )
            },
            name=(benchmark.sector, benchmark.region, scope),
            dtype=f"pint[{str(benchmark.benchmark_metric)}]",
        )
        return s

    def _get_intensity_benchmarks(
        self,
        company_sector_region_scope: Optional[pd.DataFrame] = None,
        scope_to_calc: Optional[EScope] = None,
    ) -> pd.DataFrame:
        """Overrides subclass method
        returns dataframe of all EI benchmarks if COMPANY_SECTOR_REGION_SCOPE is None.  Otherwise
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_scope: DataFrame indexed by ColumnsConfig.COMPANY_ID
        with at least the following columns: ColumnsConfig.SECTOR, ColumnsConfig.REGION, and ColumnsConfig.SCOPE
        :return: A DataFrame with company and intensity benchmarks; rows are calendar years, columns are company data
        """
        if company_sector_region_scope is None:
            return self._EI_df_t
        sec_reg_scopes = company_sector_region_scope[["sector", "region", "scope"]]
        if scope_to_calc is not None:
            sec_reg_scopes = sec_reg_scopes[sec_reg_scopes.scope.eq(scope_to_calc)]
        sec_reg_scopes_mi = pd.MultiIndex.from_frame(sec_reg_scopes).unique()
        bm_proj_t = self._EI_df_t.loc[
            range(
                self.projection_controls.BASE_YEAR,
                self.projection_controls.TARGET_YEAR + 1,
            ),
            # Here we gather all requested combos as well as ensuring we have 'Global' regional coverage
            # for sector/scope combinations that arrive with unknown region values
            [
                col
                for col in sec_reg_scopes_mi.append(
                    pd.MultiIndex.from_frame(sec_reg_scopes.assign(region="Global"))
                ).unique()
                if col in self._EI_df_t.columns
            ],
        ]
        # This piece of work essentially does a column-based join (to avoid extra transpositions)
        result = pd.concat(
            [
                (
                    bm_proj_t[tuple(ser)].rename((idx, ser.iloc[2]))
                    if tuple(ser) in bm_proj_t
                    else (
                        bm_proj_t[ser_global].rename((idx, ser.iloc[2]))
                        if (
                            ser_global := (
                                ser.iloc[0],
                                "Global",
                                ser.iloc[2],
                            )
                        )
                        in bm_proj_t
                        else pd.Series()
                    )
                )
                for idx, ser in sec_reg_scopes.iterrows()
            ],
            axis=1,
        ).dropna(axis=1, how="all")
        result.columns = pd.MultiIndex.from_tuples(
            result.columns, names=["company_id", "scope"]
        )
        return result


class BaseCompanyDataProvider(CompanyDataProvider):
    """Data provider skeleton for JSON files parsed by the fastAPI json encoder. This class serves primarily for connecting
    to the ITR tool via API.

    :param companies: A list of ICompanyData objects that each contain fundamental company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param projection_controls: An optional ProjectionControls object containing projection settings
    """

    def __init__(
        self,
        companies: List[ICompanyData],
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
    ):
        super().__init__()
        self._own_data = True
        self._column_config = column_config
        self.projection_controls = projection_controls
        # In the initialization phase, `companies` has minimal fundamental values (company_id, company_name, sector, region,
        # but not projected_intensities, projected_targets, etc)
        self._companies = companies
        # Initially we don't have to do any allocation of emissions across multiple sectors, but if we do, we'll update the index here.
        self._bm_allocation_index = pd.DataFrame().index

    @property
    def column_config(self) -> Type[ColumnsConfig]:
        """:return: ColumnsConfig values for this Data Provider"""
        return self._column_config

    @property
    def own_data(self) -> bool:
        """Return True if this object contains its own data; false if data housed elsewhere"""
        return self._own_data

    def get_projection_controls(self) -> ProjectionControls:
        return self.projection_controls

    def get_company_ids(self) -> List[str]:
        company_ids = [c.company_id for c in self._companies]
        return company_ids

    def _validate_projected_trajectories(
        self,
        companies: List[ICompanyData],
        ei_benchmarks: IntensityBenchmarkDataProvider,
    ):
        """Called when benchmark data is first known, or when projection control parameters or benchmark data changes.
        COMPANIES are a list of companies with historic data that need to be projected.
        EI_BENCHMARKS are the benchmarks for all sectors, regions, and scopes
        In previous incarnations of this function, no benchmark data was needed for any reason.
        """
        if hasattr(ei_benchmarks, "_EI_df_t"):
            ei_df_t: pd.DataFrame = ei_benchmarks._EI_df_t
        else:
            raise AttributeError(
                f"object {ei_benchmarks} does not have _EI_df_t attribute"
            )
        company_ids_without_data = [
            c.company_id
            for c in companies
            if c.historic_data.empty and c.projected_intensities.empty
        ]
        if company_ids_without_data:
            error_message = (
                "Provide either historic emission data or projections for companies with "
                f"IDs {company_ids_without_data}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        companies_without_historic_data = [
            c for c in companies if c.historic_data.empty
        ]
        if companies_without_historic_data:
            # Can arise from degenerate test cases
            pass
        base_year = self.projection_controls.BASE_YEAR
        for company in companies_without_historic_data:
            scope_em = {}
            scope_ei = {}
            if not company.projected_intensities.empty:
                for scope_name in EScope.get_scopes():
                    if isinstance(
                        company.projected_intensities[scope_name],
                        DF_ICompanyEIProjections,
                    ):
                        scope_ei[scope_name] = [
                            IEIRealization(
                                year=base_year,
                                value=company.projected_intensities[
                                    scope_name
                                ].projections[base_year],
                            )
                        ]
                    elif company.projected_intensities[scope_name] is None:
                        scope_ei[scope_name] = []
                    else:
                        # Should not be reached, but this gives right answer if it is.
                        scope_ei[scope_name] = [
                            eir.value
                            for eir in company.projected_intensities[
                                scope_name
                            ].projections
                            if eir.year == base_year
                        ]
                scope_em = {
                    scope: (
                        [
                            IEmissionRealization(
                                year=base_year,
                                value=ei[0].value * company.base_year_production,  # type: ignore
                            )
                        ]
                        if ei
                        else []
                    )
                    for scope, ei in scope_ei.items()
                }
            else:
                scope_em["S1"] = scope_em["S2"] = []
                scope_em["S3"] = (
                    [IEmissionRealization(year=base_year, value=company.ghg_s3)]
                    if company.ghg_s3
                    else []
                )
                scope_em["S1S2"] = [
                    IEmissionRealization(year=base_year, value=company.ghg_s1s2)
                ]
                scope_em["S1S2S3"] = (
                    [
                        IEmissionRealization(
                            year=base_year, value=company.ghg_s1s2 + company.ghg_s3
                        )
                    ]
                    if company.ghg_s1s2 and company.ghg_s3
                    else []
                )
                scope_ei = {
                    scope: (
                        [
                            IEIRealization(
                                year=base_year,
                                value=em[0].value / company.base_year_production,  # type: ignore
                            )
                        ]
                        if em
                        else []
                    )
                    for scope, em in scope_em.items()
                }
            company.historic_data = IHistoricData(
                productions=[
                    IProductionRealization(
                        year=base_year, value=company.base_year_production
                    )
                ],
                emissions=IHistoricEmissionsScopes(**scope_em),
                emissions_intensities=IHistoricEIScopes(**scope_ei),
            )
        companies_with_base_year_production = []
        companies_with_projections = []
        companies_without_base_year_production = []
        companies_without_projections = []
        for c in companies:
            if c.projected_intensities.empty:
                companies_without_projections.append(c)
            else:
                companies_with_projections.append(c)
            if c.base_year_production and not ITR.isna(c.base_year_production):
                companies_with_base_year_production.append(c)
            elif base_year_production_list := [
                p
                for p in c.historic_data.productions
                if p.year == base_year and not ITR.isna(p.value)
            ]:
                c.base_year_production = base_year_production_list[0].value
                companies_with_base_year_production.append(c)
            else:
                companies_without_base_year_production.append(c)
        if companies_without_projections:
            new_company_projections = EITrajectoryProjector(
                self.projection_controls, ei_df_t
            ).project_ei_trajectories(companies_without_projections)
            for c in new_company_projections:
                assert c.base_year_production is not None
                production_units = c.base_year_production.units
                if c.projected_intensities.S1S2 is None:
                    # When Gas Utilities split out S3, they often don't drag along S1S2 (and S3 are the biggies anyway)
                    assert c.projected_intensities.S3 is not None
                    production_value = (
                        c.ghg_s3 / c.projected_intensities.S3.projections[base_year]
                    )
                else:
                    production_value = (
                        c.ghg_s1s2 / c.projected_intensities.S1S2.projections[base_year]
                    )
                c.base_year_production = production_value.to(production_units)
                if not ITR.isna(c.base_year_production):
                    for i, p in enumerate(c.historic_data.productions):
                        if p.year == base_year:
                            c.historic_data.productions[i] = IProductionRealization(
                                year=base_year, value=c.base_year_production
                            )
                            break
                    for i, c2 in enumerate(companies_without_base_year_production):
                        if c.company_id == c2.company_id:
                            del companies_without_base_year_production[i]
                            break
            companies = companies_with_projections + new_company_projections
        if companies_without_base_year_production:
            logger.error(
                f"Companies without base year production: {[c.company_id for c in companies_without_base_year_production]}"
            )
        # Normalize all intensity metrics to match benchmark intensity metrics (as much as we can)
        logger.info("Normalizing intensity metrics")
        for company in companies:
            sector = company.sector
            region = company.region
            if (sector, region) in ei_df_t.columns:
                ei_dtype = ei_df_t[(sector, region)].dtypes.iloc[0]
            elif (sector, "Global") in ei_df_t.columns:
                ei_dtype = ei_df_t[(sector, "Global")].dtypes.iloc[0]
            else:
                continue
            for scope in EScope.get_scopes():
                if company.projected_intensities[scope]:
                    setattr(
                        company.projected_intensities,
                        scope,
                        DF_ICompanyEIProjections(
                            ei_metric=str(ei_dtype.units),
                            projections=company.projected_intensities[
                                scope
                            ].projections,
                        ),
                    )
        logger.info("Done normalizing intensity metrics")
        self._companies = companies

    # Because this presently defaults to S1S2 always, targets spec'd for S1 only, S2 only, or S1+S2+S3 are not well-handled.
    def _convert_projections_to_series(
        self, company: ICompanyData, feature: str, scope: EScope = EScope.S1S2
    ) -> pd.Series:
        """Extracts the company projected intensities or targets for a given scope
        :param feature: PROJECTED_TRAJECTORIES or PROJECTED_TARGETS (both are intensities)
        :param scope: a scope
        :return: pd.Series
        """
        company_dict = company.model_dump()
        production_units = str(company_dict[self.column_config.PRODUCTION_METRIC])
        emissions_units = str(company_dict[self.column_config.EMISSIONS_METRIC])

        if company_dict[feature][scope.name]:
            # Simple case: just one scope
            projections = company_dict[feature][scope.name]["projections"]
            if isinstance(projections, pd.Series):
                # FIXME: should do this upstream somehow
                projections.name = (company.company_id, scope)
                return projections.loc[
                    pd.Index(
                        range(
                            self.projection_controls.BASE_YEAR,
                            self.projection_controls.TARGET_YEAR + 1,
                        )
                    )
                ]
            return pd.Series(
                {
                    p["year"]: p["value"]
                    for p in projections
                    if p["year"]
                    in range(
                        self.projection_controls.BASE_YEAR,
                        self.projection_controls.TARGET_YEAR + 1,
                    )
                },
                name=(company.company_id, scope),
                dtype=f"pint[{emissions_units}/({production_units})]",
            )
        else:
            assert False
            # Complex case: S1+S2 or S1+S2+S3...we really don't handle yet
            scopes = [EScope[s] for s in scope.value.split("+")]
            projection_scopes = {
                s: company_dict[feature][s]["projections"]
                for s in scopes
                if company_dict[feature][s.name]
            }
            if len(projection_scopes) > 1:
                projection_series = {}
                for s in scopes:
                    projection_series[s] = pd.Series(
                        {
                            p["year"]: p["value"]
                            for p in company_dict[feature][s.name]["projections"]
                            if p["year"]
                            in range(
                                self.projection_controls.BASE_YEAR,
                                self.projection_controls.TARGET_YEAR + 1,
                            )
                        },
                        name=(company.company_id, s),
                        dtype=f"pint[{emissions_units}/({production_units})]",
                    )
                series_adder = partial(pd.Series.add, fill_value=0)
                res = reduce(series_adder, projection_series.values())
                return res
            elif len(projection_scopes) == 0:
                return pd.Series(
                    {
                        year: np.nan
                        for year in range(
                            self.historic_years[-1] + 1,
                            self.projection_controls.TARGET_YEAR + 1,
                        )
                    },
                    name=company.company_id,
                    dtype=f"pint[{emissions_units}/({production_units})]",
                )
            else:
                projections = company_dict[feature][list(projection_scopes.keys())[0]][
                    "projections"
                ]

    def _calculate_target_projections(
        self,
        production_bm: ProductionBenchmarkDataProvider,
        ei_bm: IntensityBenchmarkDataProvider,
    ):
        """We cannot calculate target projections until after we have loaded benchmark data.
        We do so when companies are associated with benchmarks, in the DataWarehouse construction

        :param production_bm: A Production Benchmark (multi-sector, single-scope, 2020-2050)
        :param ei_bm: Intensity Benchmarks for all sectors and scopes defined by the benchmark, 2020-2050
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # FIXME: Note that we don't need to call with a scope, because production is independent of scope.
            # We use the arbitrary EScope.AnyScope just to be explicit about that.
            df_partial_pp = production_bm._get_projected_production(EScope.AnyScope)

        ei_df_t = ei_bm._get_intensity_benchmarks()

        for c in self._companies:
            if not c.projected_targets.empty:
                continue
            if c.target_data is None:
                logger.warning(f"No target data for {c.company_name}")
            else:
                base_year_production = next(
                    (
                        p.value
                        for p in c.historic_data.productions
                        if p.year == self.projection_controls.BASE_YEAR
                    ),
                    None,
                )
                try:
                    co_cumprod = (
                        df_partial_pp.loc[c.sector, c.region, EScope.AnyScope]
                        * base_year_production
                    )
                except KeyError:
                    # FIXME: Should we fix region info upstream when setting up comopany data?
                    co_cumprod = (
                        df_partial_pp.loc[c.sector, "Global", EScope.AnyScope]
                        * base_year_production
                    )
                try:
                    if ei_bm:
                        if (c.sector, c.region) in ei_df_t.columns:
                            df = ei_df_t.loc[:, (c.sector, c.region)]
                        elif (c.sector, "Global") in ei_df_t.columns:
                            df = ei_df_t.loc[:, (c.sector, "Global")]
                        else:
                            logger.error(
                                f"company {c.company_name} with ID {c.company_id} sector={c.sector} region={c.region} not in EI benchmark"
                            )
                            df = None
                    else:
                        df = None
                    c.projected_targets = EITargetProjector(
                        self.projection_controls
                    ).project_ei_targets(
                        c,
                        align_production_to_bm(co_cumprod, df.iloc[:, 0]),
                        df,
                    )
                except IndexError as err:
                    import traceback

                    logger.error(
                        f"While calculating target projections for {c.company_id}, raised IndexError({err})"
                    )
                    traceback.print_exc()
                    logger.info("Continuing from _calculate_target_projections...")
                    c.projected_targets = (
                        ITR.interfaces.empty_ICompanyEIProjectionsScopes
                    )
                except Exception as err:
                    import traceback

                    logger.error(
                        f"While calculating target projections for {c.company_id}, raised {err} (possible intensity vs. absolute unit mis-match?)"
                    )
                    traceback.print_exc()
                    logger.info("Continuing from _calculate_target_projections...")
                    c.projected_targets = (
                        ITR.interfaces.empty_ICompanyEIProjectionsScopes
                    )

    # ??? Why prefer TRAJECTORY over TARGET?
    def _get_company_intensity_at_year(
        self, year: int, company_ids: List[str]
    ) -> pd.Series:
        """Returns projected intensities for a given set of companies and year
        :param year: calendar year
        :param company_ids: List of company ids
        :return: pd.Series with intensities for given company ids
        """
        return self.get_company_projected_trajectories(company_ids, year=year)

    def get_company_data(
        self, company_ids: Optional[List[str]] = None
    ) -> List[ICompanyData]:
        """Get all relevant data for a list of company ids (ISIN), or all company data if `company_ids` is None.
        This method should return a list of ICompanyData instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        if company_ids is None:
            return self._companies

        company_data = [
            company for company in self._companies if company.company_id in company_ids
        ]

        if len(company_data) is not len(company_ids):
            missing_ids = set(company_ids) - set(self.get_company_ids())
            logger.warning(
                f"Companies not found in fundamental data and excluded from further computations: "
                f"{missing_ids}"
            )

        return company_data

    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """Gets the value of a variable for a list of companies ids
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        # FIXME: this is an expensive operation as it converts all fields in the model just to get a single VARIABLE_NAME
        return self.get_company_fundamentals(company_ids)[variable_name]

    def get_company_intensity_and_production_at_base_year(
        self, company_ids: List[str]
    ) -> pd.DataFrame:
        """Overrides subclass method
        :param: company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.PRODUCTION_METRIC, ColumnsConfig.BASE_EI,
        ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE,
        ColumnsConfig.GHG_SCOPE12, ColumnsConfig.GHG_SCOPE3

        The BASE_EI column is for the scope in the SCOPE column.
        """
        # FIXME: this creates an untidy data mess.  GHG_SCOPE12 and GHG_SCOPE3 are anachronisms.
        # company_data = self.get_company_data(company_ids)
        df_fundamentals = self.get_company_fundamentals(company_ids)
        base_year = self.projection_controls.BASE_YEAR
        company_info = df_fundamentals.loc[
            company_ids,
            [
                self.column_config.SECTOR,
                self.column_config.REGION,
                self.column_config.BASE_YEAR_PRODUCTION,
                self.column_config.GHG_SCOPE12,
                self.column_config.GHG_SCOPE3,
            ],
        ]
        # Do rely on getting info from projections; Don't grovel through historic data instead
        ei_at_base = self._get_company_intensity_at_year(base_year, company_ids).rename(
            self.column_config.BASE_EI
        )
        # historic_ei = { (company.company_id, scope): { self.column_config.BASE_EI: eir.value }
        #                 for scope in EScope.get_result_scopes()
        #                 for company in company_data
        #                 for eir in getattr(company.historic_data.emissions_intensities, scope.name)
        #                 if eir.year==base_year }
        #
        # ei_at_base = pd.DataFrame.from_dict(historic_ei, orient='index')
        # ei_at_base.index.names=['company_id', 'scope']
        df = company_info.merge(ei_at_base, left_index=True, right_index=True)
        df.reset_index("scope", inplace=True)
        cols = df.columns.tolist()
        df = df[cols[1:3] + [cols[0]] + cols[3:]]
        return df

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """:param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company (company_id is a column)
        """
        excluded_cols = [
            "projected_targets",
            "projected_intensities",
            "historic_data",
            "target_data",
        ]
        df = pd.DataFrame.from_records(
            [
                dict(
                    ICompanyData.model_validate(
                        {k: v for k, v in dict(c).items() if k not in excluded_cols}
                    )
                )
                for c in self.get_company_data(company_ids)
            ]
        ).set_index(self.column_config.COMPANY_ID)
        return df

    def get_company_projected_trajectories(
        self, company_ids: List[str], year=None
    ) -> pd.DataFrame:
        """:param company_ids: A list of company IDs
        :param year: values for a specific year, or all years if None
        :return: A pandas DataFrame with projected intensity trajectories per company, indexed by company_id and scope
        """
        c_ids: List[str] = []
        scopes: List[EScope] = []
        projections: List[DF_ICompanyEIProjections] = []

        for c in self._companies:
            if c.company_id in company_ids:
                for scope_name in EScope.get_scopes():
                    if c.projected_intensities[scope_name]:
                        c_ids.append(c.company_id)
                        scopes.append(EScope[scope_name])
                        projections.append(
                            c.projected_intensities[scope_name].projections
                        )

        if len(projections) == 0:
            return pd.DataFrame()
        index = pd.MultiIndex.from_tuples(
            zip(c_ids, scopes), names=["company_id", "scope"]
        )
        if year is not None:
            values = list(
                map(
                    cast(Callable[[pd.Series], Any], lambda x: x[year].squeeze()),
                    projections,
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # pint units don't like columns of heterogeneous data...tough!
                return pd.Series(data=values, index=index, name=year)
        else:
            values = projections
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return pd.DataFrame(data=values, index=index)

    def get_company_projected_targets(
        self, company_ids: List[str], year=None
    ) -> pd.DataFrame:
        """:param company_ids: A list of company IDs
        :param year: values for a specific year, or all years if None
        :return: A pandas DataFrame with projected intensity targets per company, indexed by company_id
        """
        # Tempting as it is to follow the pattern of constructing the same way we create `projected_trajectories`
        # targets are trickier because they have ragged left edges that want to fill with NaNs when put into DataFrames.
        # _convert_projections_to_series has the nice side effect that PintArrays produce NaNs with units.
        # So if we just need a year from this dataframe, we compute the whole dataframe and return one column.
        # Feel free to write a better implementation if you have time!
        target_list = [
            self._convert_projections_to_series(
                c, self.column_config.PROJECTED_TARGETS, EScope[scope_name]
            )
            for c in self.get_company_data(company_ids)
            for scope_name in EScope.get_scopes()
            if c.projected_targets[scope_name]
        ]
        if target_list:
            with warnings.catch_warnings():
                # pd.DataFrame.__init__ (in pandas/core/frame.py) ignores
                # the beautiful dtype information adorning the pd.Series list
                # elements we are providing.  Sad!
                warnings.simplefilter("ignore")
                # If target_list produces a ragged left edge,
                # resort columns so that earliest year is leftmost
                df = pd.DataFrame(target_list).sort_index(axis=1)
                df.index.set_names(["company_id", "scope"], inplace=True)
                if year is not None:
                    return df[year]
                return df
        return pd.DataFrame()

    def _allocate_emissions(
        self,
        new_companies: List[ICompanyData],
        benchmarks_projected_ei: IntensityBenchmarkDataProvider,
        projection_controls: ProjectionControls,
    ):
        """Use benchmark data from `ei_benchmarks` to allocate sector-level emissions from aggregated emissions.
        For example, a Utility may supply both Electricity and Gas to customers, reported separately.
        When we split the company into Electricity and Gas lines of business, we can allocate Scope emissions
        to the respective lines of business using benchmark averages to guide the allocation.
        """
        logger.info("Allocating emissions to align with benchmark data")
        bm_ei_df_t = benchmarks_projected_ei._get_intensity_benchmarks()
        bm_sectors = bm_ei_df_t.columns.get_level_values("sector").unique().to_list()
        base_year = self.get_projection_controls().BASE_YEAR

        from collections import defaultdict

        sectors_dict = defaultdict(list)
        region_dict = {}
        historic_dict = {}

        for c in new_companies:
            orig_id, sector = c.company_id.split("+")
            if sector in bm_sectors:
                sectors_dict[orig_id].append(sector)
            else:
                logger.error(
                    f"No benchmark sector data for {orig_id}: sector = {sector}"
                )
                continue
            if (sector, c.region) in bm_ei_df_t.columns:
                region_dict[orig_id] = c.region
            elif (sector, "Global") in bm_ei_df_t.columns:
                region_dict[orig_id] = "Global"
            else:
                logger.error(
                    f"No benchmark region data for {orig_id}: sector = {sector}; region = {c.region}"
                )
                continue

            # Though we mutate below, it's our own unique copy of c.historic_data we are mutating, so OK
            historic_dict[c.company_id] = c.historic_data

        for orig_id, sectors in sectors_dict.items():
            region = region_dict[orig_id]
            sector_ei = [
                (
                    sector,
                    scope,
                    bm_ei_df_t.loc[:, (sector, region, scope)][base_year],
                )
                for scope in EScope.get_result_scopes()
                for sector in sectors
                if (sector, region, scope) in bm_ei_df_t.columns
                and ((new_id := "+".join([orig_id, sector])), scope.name)
                in self._bm_allocation_index
                and historic_dict[new_id].emissions[scope.name]
            ]
            if sector_ei == []:
                continue
            sector_ei_df = pd.DataFrame(
                sector_ei, columns=["sector", "scope", "ei"]
            ).set_index(["sector"])
            sector_prod_df = pd.DataFrame(
                [
                    (sector, prod.value)
                    for sector in sectors
                    for prod in historic_dict["+".join([orig_id, sector])].productions
                    # FIXME: if we don't have proudction values for BASE_YEAR, this fails!  See 'US2333311072+Gas Utilities'
                    if prod.year == base_year
                ],
                columns=["sector", "prod"],
            ).set_index("sector")
            sector_em_df = (
                sector_ei_df.join(sector_prod_df)
                .assign(em=lambda x: x["ei"] * x["prod"])
                .set_index("scope", append=True)
                .drop(columns=["ei", "prod"])
            )
            # Only now can we drop whatever is not in self.company_data._bm_allocation_index from sector_em_df
            if self._bm_allocation_index.empty:
                # No allocations to make for any companies
                continue
            to_allocate_idx = self._bm_allocation_index[
                self._bm_allocation_index.map(lambda x: x[0].startswith(orig_id))
            ].map(lambda x: (x[0].split("+")[1], EScope[x[1]]))
            if to_allocate_idx.empty:
                logger.info(
                    f"Already allocated emissions for {orig_id} across {sectors}"
                )
                continue
            # FIXME: to_allocate_idx is missing S1S2S3 for US2091151041
            to_allocate_idx.names = ["sector", "scope"]
            try:
                sector_em_df = sector_em_df.loc[
                    sector_em_df.index.intersection(to_allocate_idx)
                ].astype("pint[Mt CO2e]")
            except DimensionalityError:
                # breakpoint()
                assert False
            em_tot = sector_em_df.groupby("scope")["em"].sum()
            # The alignment calculation: Company Scope-Sector emissions = Total Company Scope emissions * (BM Scope Sector / SUM(All Scope Sectors of Company))
            aligned_em = [
                (
                    sector,
                    [
                        (
                            scope,
                            list(
                                map(
                                    lambda em: (
                                        em[0],
                                        em[1]
                                        * sector_em_df.loc[(sector, scope)].squeeze()
                                        / em_tot.loc[scope].squeeze(),
                                    ),
                                    [
                                        (em.year, em.value)
                                        for em in historic_dict[
                                            "+".join([orig_id, sector])
                                        ].emissions[scope.name]
                                    ],
                                )
                            ),
                        )
                        for scope in em_tot.index
                        if em_tot.loc[scope].squeeze().m != 0.0
                    ],
                )
                for sector in sectors
            ]

            # Having done all scopes and sectors for this company above,
            # replace historic Em and EI data below
            for sector_aligned in aligned_em:
                sector, scopes = sector_aligned
                historic_sector = historic_dict["+".join([orig_id, sector])]
                assert historic_sector is not None
                # print(f"Historic {sector} initially\n{historic_sector.emissions}")
                for scope_tuple in scopes:
                    scope, em_list = scope_tuple
                    setattr(
                        historic_sector.emissions,
                        scope.name,
                        list(
                            map(
                                lambda em: IEmissionRealization(
                                    year=em[0], value=em[1].to("Mt CO2e")
                                ),
                                em_list,
                            )
                        ),
                    )
                    prod_list = historic_sector.productions
                    ei_list = list(
                        map(
                            lambda em_p: IEIRealization(
                                year=em_p[0].year,
                                value=(
                                    Q_(
                                        np.nan,
                                        f"({em_p[0].value.u}) / ({em_p[1].value.u})",  # type: ignore
                                    )
                                    if em_p[1].value.m == 0.0  # type: ignore
                                    else em_p[0].value / em_p[1].value
                                ),
                            ),
                            zip(historic_sector.emissions[scope.name], prod_list),
                        )
                    )
                    setattr(historic_sector.emissions_intensities, scope.name, ei_list)
                # print(f"Historic {sector} adjusted\n{historic_dict['+'.join([orig_id, sector])].emissions}")
        logger.info("Sector alignment complete")


class EIProjector(object):
    """This class implements generic projection functions used for both trajectory and target projection."""

    def __init__(self, projection_controls: ProjectionControls = ProjectionControls()):
        self.projection_controls = projection_controls

    def _get_bounded_projections(self, results) -> List[ICompanyEIProjection]:
        if isinstance(results, list):
            projections = [
                projection
                for projection in results
                if projection.year
                in range(
                    self.projection_controls.BASE_YEAR,
                    self.projection_controls.TARGET_YEAR + 1,
                )
            ]
        else:
            projections = [
                ICompanyEIProjection(year=year, value=value)
                for year, value in results.items()
                if year
                in range(
                    self.projection_controls.BASE_YEAR,
                    self.projection_controls.TARGET_YEAR + 1,
                )
            ]
        return projections


class EITrajectoryProjector(EIProjector):
    """This class projects emissions intensities on company level based on historic data on:
    - A company's emission history (in t CO2)
    - A company's production history (units depend on industry, e.g. TWh for electricity)

    It returns the full set of both historic emissions intensities and projected emissions intensities.
    """

    # EI benchmark data indexed by SECTOR, REGION, and SCOPE

    def __init__(
        self,
        projection_controls: ProjectionControls = ProjectionControls(),
        ei_df_t=None,
        *args,
        **kwargs,
    ):
        super().__init__(projection_controls=projection_controls)
        self._EI_df_t = pd.DataFrame() if ei_df_t is None else ei_df_t

    def project_ei_trajectories(
        self, companies: List[ICompanyData], backfill_needed=True
    ) -> List[ICompanyData]:
        historic_df = self._extract_historic_df(companies)
        # This modifies historic_df in place...which feeds the intensity extrapolations below
        self._align_and_compute_missing_historic_ei(companies, historic_df)
        historic_years = [
            column for column in historic_df.columns if isinstance(column, int)
        ]
        projection_years = range(
            max(historic_years), self.projection_controls.TARGET_YEAR + 1
        )
        with warnings.catch_warnings():
            # Don't worry about warning that we are intentionally dropping units as we transpose
            warnings.simplefilter("ignore")
            historic_ei_t = asPintDataFrame(
                historic_df[historic_years]
                .query(f"variable=='{VariablesConfig.EMISSIONS_INTENSITIES}'")
                .T
            ).pint.dequantify()
            historic_ei_t.index.name = "year"
        if backfill_needed:
            # Fill in gaps between BASE_YEAR and the first data we have
            if ITR.HAS_UNCERTAINTIES:
                historic_ei_t = historic_ei_t.map(
                    lambda x: np.nan if ITR.isna(x) else x
                )
            backfilled_t = historic_ei_t.bfill(axis=0)
            # FIXME: this hack causes backfilling only on dates on or after the
            # first year of the benchmark, which keeps it from disrupting current test cases
            # while also working on real-world use cases.  But we need to formalize this decision.
            backfilled_t = backfilled_t.reset_index()
            backfilled_t = backfilled_t.where(
                backfilled_t.year >= self.projection_controls.BASE_YEAR,
                historic_ei_t.reset_index(),
            )
            backfilled_t.set_index("year", inplace=True)
            if not historic_ei_t.compare(backfilled_t).empty:
                logger.warning(
                    f"some data backfilled to {self.projection_controls.BASE_YEAR} for company_ids in list \
                    {historic_ei_t.compare(backfilled_t).columns.get_level_values('company_id').unique().tolist()}"
                )
                historic_ei_t = backfilled_t.sort_index(axis=1)
                for company in companies:
                    if ITR.isna(company.base_year_production):
                        # If we have no valid production data, we cannot use EI data to compute emissions
                        continue
                    for ghg_attr, ghg_scope in [
                        (ColumnsConfig.GHG_SCOPE3, EScope.S3),
                        (ColumnsConfig.GHG_SCOPE12, EScope.S1S2),
                    ]:
                        if ITR.isna(getattr(company, ghg_attr)):
                            try:
                                idx = (
                                    company.company_id,
                                    "Emissions Intensities",
                                    ghg_scope,
                                )
                                setattr(
                                    company,
                                    ghg_attr,
                                    Q_(
                                        historic_ei_t[idx]
                                        .loc[self.projection_controls.BASE_YEAR]
                                        .squeeze(),
                                        historic_ei_t[idx].columns[0],
                                    )
                                    * company.base_year_production,
                                )
                            except KeyError:
                                # If it's not there, we'll complain later
                                pass
        standardized_ei_t = self._standardize(historic_ei_t)
        intensity_trends_t = self._get_trends(standardized_ei_t)
        extrapolated_t = self._extrapolate(
            intensity_trends_t, projection_years, historic_ei_t
        )
        # Restrict projection to benchmark years
        extrapolated_t = extrapolated_t[
            extrapolated_t.index >= self.projection_controls.BASE_YEAR
        ]
        # Restore row-wise shape of DataFrame
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pint units don't like being twisted from columns to rows, but it's ok
            self._add_projections_to_companies(
                companies, extrapolated_t.pint.quantify()
            )
        return companies

    def _extract_historic_df(self, companies: List[ICompanyData]) -> pd.DataFrame:
        data = []
        for company in companies:
            if company.historic_data.empty:
                continue
            c_hd = company.historic_data
            if len(c_hd.productions):
                data.append(
                    self._historic_productions_to_dict(
                        company.company_id, c_hd.productions
                    )
                )
            if not c_hd.emissions.empty:
                data.extend(
                    self._historic_emissions_to_dicts(
                        company.company_id, c_hd.emissions
                    )
                )
            if not c_hd.emissions_intensities.empty:
                data.extend(
                    self._historic_ei_to_dicts(
                        company.company_id, c_hd.emissions_intensities
                    )
                )
        if not data:
            logger.error(
                f"No historic data for companies: {[c.company_id for c in companies]}"
            )
            raise ValueError("No historic data anywhere")
        df = pd.DataFrame.from_records(data).set_index(
            [ColumnsConfig.COMPANY_ID, ColumnsConfig.VARIABLE, ColumnsConfig.SCOPE]
        )

        # Note that the first valid index may well be Quantity with a NaN value--that's fine
        # We just need to fill in the pure NaNs that arise from very ragged data
        with warnings.catch_warnings():
            # TODO: need to investigate whether there is a more sane way to avoid unit warnings
            warnings.simplefilter("ignore")
            df_first_valid = df.apply(lambda x: x[x.first_valid_index()], axis=1)
        df_filled = df.fillna(
            df.apply(lambda x: df_first_valid.map(lambda y: Q_(np.nan, y.u)))
        )
        return df_filled

    def _historic_productions_to_dict(
        self, id: str, productions: List[IProductionRealization]
    ) -> Dict[Any, Any]:
        """Construct a dictionary that will later turned into a DataFrame indexed by COMAPNY_ID, VARIABLE, and SCOPE.
        In this case (Production), scope is 'Production'.
        Columns are YEARs and values are Quantiities.
        """
        prods = {prod.year: prod.value for prod in productions}
        return {
            ColumnsConfig.COMPANY_ID: id,  # type: ignore
            ColumnsConfig.VARIABLE: VariablesConfig.PRODUCTIONS,  # type: ignore
            ColumnsConfig.SCOPE: "Production",  # type: ignore
            **prods,
        }

    def _historic_emissions_to_dicts(
        self, id: str, emissions_scopes: IHistoricEmissionsScopes
    ) -> List[Dict[Any, Any]]:
        """Construct a dictionary that will later turned into a DataFrame indexed by COMAPNY_ID, VARIABLE, and SCOPE.
        In this case (Emissions), scopes are 'S1', 'S2', 'S3', 'S1S2', and 'S1S2S3'.
        Columns are YEARs and values are Quantiities.
        """
        data = []
        for scope, emissions in dict(emissions_scopes).items():
            if emissions:
                ems = {em["year"]: em["value"] for em in emissions}
                data.append(
                    {
                        ColumnsConfig.COMPANY_ID: id,
                        ColumnsConfig.VARIABLE: VariablesConfig.EMISSIONS,
                        ColumnsConfig.SCOPE: EScope[scope],
                        **ems,
                    }
                )
        return data

    def _historic_ei_to_dicts(
        self, id: str, intensities_scopes: IHistoricEIScopes
    ) -> List[Dict[Any, Any]]:
        data = []
        for scope, intensities in dict(intensities_scopes).items():
            if intensities:
                intsties = {intsty["year"]: intsty["value"] for intsty in intensities}
                data.append(
                    {
                        ColumnsConfig.COMPANY_ID: id,
                        ColumnsConfig.VARIABLE: VariablesConfig.EMISSIONS_INTENSITIES,
                        ColumnsConfig.SCOPE: EScope[scope],
                        **intsties,
                    }
                )
        return data

    # Each benchmark defines its own scope requirements on a per-sector/per-region basis.
    # The benchmark EI metrics (t CO2e/GJ) may not align with disclosed EI (t CO2/ CH4 / bcm)
    # So we both align the disclosed EI data to the benchmark metrics, and we fill data
    # gaps where EI can be immediately computed from emissions and production metrics.
    # No fancy estimations or allocations here.
    def _align_and_compute_missing_historic_ei(
        self, companies: List[ICompanyData], historic_df: pd.DataFrame
    ):
        scopes = [EScope[scope_name] for scope_name in EScope.get_scopes()]
        missing_data = []
        misaligned_data = []
        # https://github.com/pandas-dev/pandas/issues/53053
        index_names = historic_df.index.names
        for company in companies:
            # Create keys to index historic_df DataFrame for readability
            production_key = (
                company.company_id,
                VariablesConfig.PRODUCTIONS,
                "Production",
            )
            emissions_keys = {
                scope: (company.company_id, VariablesConfig.EMISSIONS, scope)
                for scope in scopes
            }
            ei_keys = {
                scope: (
                    company.company_id,
                    VariablesConfig.EMISSIONS_INTENSITIES,
                    scope,
                )
                for scope in scopes
            }
            this_missing_data = []
            this_misaligned_data: List[str] = []
            append_this_missing_data = True
            try:
                aligned_production = asPintSeries(historic_df.loc[production_key])
            except KeyError:
                this_missing_data.append(f"{company.company_id} - Production")
                continue
            if self._EI_df_t.empty:
                pass
            else:
                if (company.sector, company.region) in self._EI_df_t.columns:
                    ei_df_t = self._EI_df_t.loc[:, (company.sector, company.region)]
                else:
                    ei_df_t = self._EI_df_t.loc[:, (company.sector, "Global")]
                # This assumes all benchmark scopes have the same units, so we can just choose the first
                # and lift this computation out of a loop
                try:
                    aligned_production = align_production_to_bm(
                        aligned_production, ei_df_t.iloc[0]
                    )
                except DimensionalityError:
                    if not this_misaligned_data:
                        # We only need one such for our error report
                        this_misaligned_data.append(
                            f"{company.company_id} - {aligned_production.iloc[0].units} vs {ei_df_t.iloc[0].dtype.units}"
                        )
                    continue
            for scope in scopes:
                if ei_keys[scope] in historic_df.index:
                    assert company.historic_data.emissions_intensities[scope.name]
                    # We could check that our EI calculations agree with our disclosed EI data, but bad reference data in test cases would cause tests to fail
                    # assert all([historic_df.loc[emissions_keys[scope]].loc[item.year] / aligned_production.loc[item.year] == item.value
                    #            for item in company.historic_data.emissions_intensities[scope.name] if not pd.isna(item.value)])
                    append_this_missing_data = False
                    continue
                # Emissions intensities not yet computed for this scope
                try:  # All we will try is computing EI from Emissions / Production
                    historic_df.loc[ei_keys[scope]] = (
                        historic_df.loc[emissions_keys[scope]] / aligned_production
                    )
                    append_this_missing_data = False
                except KeyError:
                    this_missing_data.append(f"{company.company_id} - {scope.name}")
                    continue
                # Note that we don't actually add new-found data to company.historic data
                # ...only as the starting point for projections (in historic_df)
            if this_misaligned_data:
                misaligned_data.extend(this_misaligned_data)
            # This only happens if ALL scope data is missing.  If ANY scope data is present, we'll work with what we get.
            if this_missing_data and append_this_missing_data:
                missing_data.extend(this_missing_data)
        # https://github.com/pandas-dev/pandas/issues/53053
        historic_df.index.names = index_names
        if misaligned_data:
            warning_message = f"Ignoring unalignable production metrics with benchmark intensity metrics for these companies: {misaligned_data}"
            logger.warning(warning_message)
        if missing_data:
            error_message = (
                "Provide either historic emissions intensity data, or historic emission and "
                f"production data for these company - scope combinations: {missing_data}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

    def _add_projections_to_companies(
        self, companies: List[ICompanyData], extrapolations_t: pd.DataFrame
    ):
        projection_range = range(
            self.projection_controls.BASE_YEAR, self.projection_controls.TARGET_YEAR + 1
        )
        for company in companies:
            scope_projections: Dict[str, pd.Series | None] = {}
            scope_dfs = {}
            scope_names = EScope.get_scopes()
            for scope_name in scope_names:
                if not company.historic_data.emissions_intensities[scope_name]:
                    scope_projections[scope_name] = None
                    continue
                results = extrapolations_t[
                    (
                        company.company_id,
                        VariablesConfig.EMISSIONS_INTENSITIES,
                        EScope[scope_name],
                    )
                ]
                if not isinstance(results.dtype, PintType):
                    if results.isna().all():
                        # Pure NaN results (not Quantity(nan)) likely constructed from degenerate test case
                        scope_projections[scope_name] = None
                        continue
                    assert False
                # FIXME: is it OK to discard purely NAN results (and change the testsuite accordingly)?
                # if results.isna().all():
                #     scope_projections[scope_name] = None
                #     continue
                scope_dfs[scope_name] = results
                scope_projections[scope_name] = results[
                    results.index.isin(projection_range)
                ]
            if (
                scope_projections["S1S2"] is None
                and scope_projections["S1"] is not None
                and scope_projections["S2"] is not None
            ):
                results = scope_dfs["S1"] + scope_dfs["S2"]
                scope_dfs["S1S2"] = results
                scope_projections["S1S2"] = results[
                    results.index.isin(projection_range)
                ]
            if (
                scope_projections["S1S2S3"] is None
                and scope_projections["S1S2"] is not None
                and scope_projections["S3"] is not None
            ):
                results = scope_dfs["S1S2"] + scope_dfs["S3"]
                # We don't need to compute scope_dfs['S1S2S3'] because nothing further depends on accessing it here
                scope_projections["S1S2S3"] = results[
                    results.index.isin(projection_range)
                ]
            company.projected_intensities = ICompanyEIProjectionsScopes(
                **scope_projections
            )

    def _standardize(self, intensities_t: pd.DataFrame) -> pd.DataFrame:
        # At the starting point, we expect that if we have S1, S2, and S1S2 intensities, that S1+S2 = S1S2
        # After winsorization, this is no longer true, because S1 and S2 will be clipped differently than S1S2.

        # It is very convenient to integrate interpolation (which only works on numeric datatypes, not
        # quantities and not uncertainties) with the winsorization process.  So there's no separate
        # _interpolate method.
        winsorized_ei_t: pd.DataFrame = self._winsorize(intensities_t)
        return winsorized_ei_t

    def _winsorize(self, historic_ei_t: pd.DataFrame) -> pd.DataFrame:
        # quantile doesn't handle pd.NA inside Quantity; FIXME: we can use np.nan because not expecting UFloat in input data

        # Turns out we have to dequantify here: https://github.com/pandas-dev/pandas/issues/45968
        # Can try again when ExtensionArrays are supported by `quantile`, `clip`, and friends
        if ITR.HAS_UNCERTAINTIES:
            try:
                nominal_ei_t = historic_ei_t.apply(
                    lambda col: (
                        pd.Series(
                            ITR.nominal_values(col.values),
                            index=col.index,
                            name=col.name,
                        )
                        if col.dtype.kind == "O"
                        else col.astype("float64")
                    )
                )
                err_ei_t = historic_ei_t.apply(
                    lambda col: pd.Series(
                        ITR.std_devs(col.values), index=col.index, name=col.name
                    )
                )
            except ValueError:
                logger.error("ValueError in _winsorize")
                raise
        else:
            # pint.dequantify did all the hard work for us
            nominal_ei_t = historic_ei_t
        # See https://github.com/hgrecco/pint-pandas/issues/114
        lower = nominal_ei_t.quantile(
            q=self.projection_controls.LOWER_PERCENTILE,
            axis="index",
            numeric_only=False,
        )
        upper = nominal_ei_t.quantile(
            q=self.projection_controls.UPPER_PERCENTILE,
            axis="index",
            numeric_only=False,
        )
        # FIXME: the clipping process can properly introduce uncertainties.  The low and high values that are clipped could be
        # replaced by the clipped values +/- the lower and upper percentile values respectively.
        winsorized_t: pd.DataFrame = nominal_ei_t.clip(
            lower=lower, upper=upper, axis="columns"
        )
        wnom_t = winsorized_t.astype("float64").apply(
            lambda col: col.interpolate(
                method="linear",
                inplace=False,
                limit_direction="forward",
                limit_area="inside",
            )
        )
        if ITR.HAS_UNCERTAINTIES:
            werr_t = err_ei_t.apply(
                lambda col: col.where(
                    winsorized_t[col.name].notna(),
                    abs(
                        (wnom_col := wnom_t[col.name]).shift(
                            1, fill_value=wnom_col.iloc[0]
                        )
                        - wnom_col.shift(-1, fill_value=wnom_col.iloc[-1])
                    ),
                )
            )
            uwinsorized_t = wnom_t.combine(werr_t, ITR.recombine_nom_and_std)
            return uwinsorized_t

        # FIXME: If we have S1, S2, and S1S2 intensities, should we treat winsorized(S1)+winsorized(S2) as winsorized(S1S2)?
        # FIXME: If we have S1S2 (or S1 and S2) and S3 and S1S23 intensities, should we treat winsorized(S1S2)+winsorized(S3) as winsorized(S1S2S3)?
        return wnom_t

    def _interpolate(self, historic_ei_t: pd.DataFrame) -> pd.DataFrame:
        # Interpolate NaNs surrounded by values, but don't extrapolate NaNs with last known value
        raise NotImplementedError

    def _get_trends(self, intensities_t: pd.DataFrame):
        # FIXME: rolling windows require conversion to float64.  Don't want to be a nuisance...
        intensities_t = intensities_t.apply(
            lambda col: (
                col
                if col.dtype == np.float64
                # Float64 NA needs to be converted to np.nan before we can apply nominal_values
                else ITR.nominal_values(col.fillna(np.nan)).astype(np.float64)
            )
        )
        # FIXME: Pandas 2.1
        # Treat NaN ratios as "unchnaged year on year"
        # FIXME Could we ever have UFloat NaNs here?  np.nan is valid UFloat.
        ratios_t: pd.DataFrame = intensities_t.rolling(window=2, closed="right").apply(
            func=self._year_on_year_ratio, raw=True
        )
        ratios_t = ratios_t.apply(
            lambda col: col.fillna(0) if all(col.map(lambda x: ITR.isna(x))) else col
        )

        # # Add weight to trend movements across multiple years (normalized to year-over-year, not over two years...)
        # # FIXME: we only want to do this for median, not mean.
        # if self.projection_controls.TREND_CALC_METHOD==pd.DataFrame.median:
        #     ratios_2 = ratios
        #     ratios_3: pd.DataFrame = intensities.rolling(window=3, axis='index', closed='right') \
        #         .apply(func=self._year_on_year_ratio, raw=True).div(2.0)
        #     ratios = pd.concat([ratios_2, ratios_3])
        # elif self.projection_controls.TREND_CALC_METHOD==pd.DataFrame.mean:
        #     pass
        # else:
        #     raise ValueError("Unhanlded TREND_CALC_METHOD")

        trends_t: pd.DataFrame = self.projection_controls.TREND_CALC_METHOD(
            ratios_t, axis="index", skipna=True
        ).clip(  # type: ignore
            lower=self.projection_controls.LOWER_DELTA,
            upper=self.projection_controls.UPPER_DELTA,
        )
        return trends_t

    def _extrapolate(
        self,
        trends_t: pd.Series,
        projection_years: range,
        historic_ei_t: pd.DataFrame,
    ) -> pd.DataFrame:
        historic_ei_t = historic_ei_t[
            historic_ei_t.columns.intersection(trends_t.index)
        ]
        # We need to do a mini-extrapolation if we don't have complete historic data

        def _extrapolate_mini(col, trend):
            from pandas.api.types import is_numeric_dtype

            # Inside these functions, columns are numeric PintArrays, not quantity-based Series
            col_na = col.isna()
            col_na_idx = col_na[col_na].index
            last_valid = col[~col_na].tail(1).squeeze()
            if np.isnan(trend):
                # If there's no visible trend, just assume it doesn't decrease and copy results forward
                col.loc[col_na_idx] = last_valid
            else:
                mini_trend = pd.Series(
                    [trend + 1] * len(col_na[col_na]), index=col_na_idx, dtype="float64"
                ).cumprod()
                col.loc[col_na_idx] = last_valid * mini_trend
            if col.dtype != np.float64 and is_numeric_dtype(col.dtype):
                # FIXME: Pandas 2.1 we prefer NaNs to NA in Arrays for now
                return col.astype(np.float64)
            return col

        historic_ei_t = historic_ei_t.apply(
            lambda col: _extrapolate_mini(col, trends_t[col.name])
        )

        # Now the big extrapolation
        projected_ei_t = (
            pd.concat([trends_t.add(1.0)] * len(projection_years[1:]), axis=1)
            .T.cumprod()
            .rename(
                index=dict(
                    zip(range(0, len(projection_years[1:])), projection_years[1:])
                )
            )
            .mul(historic_ei_t.iloc[-1], axis=1)
        )

        # Clean up rows by converting NaN/None into Quantity(np.nan, unit_type)
        columnwise_ei_t = pd.concat([historic_ei_t, projected_ei_t])
        columnwise_ei_t.index.name = "year"
        return columnwise_ei_t

    # Might return a float, might return a ufloat
    def _year_on_year_ratio(self, arr: np.ndarray):
        # Subsequent zeroes represent no year-on-year change
        if arr[0] == 0.0 and arr[-1] == 0.0:
            return 0.0
        # Due to rounding, we might overshoot the zero target and go negative
        # So round the negative number to zero and treat it as a 100% year-on-year decline
        if arr[0] >= 0.0 and arr[-1] <= 0.0:
            return -1.0
        return (arr[-1] / arr[0]) - 1.0


class EITargetProjector(EIProjector):
    """This class projects emissions intensities from a company's targets and historic data. Targets are specified per
    scope in terms of either emissions or emission intensity reduction. Interpolation between last known historic data
    and (a) target(s) is CAGR-based, but not entirely CAGR (beacuse zero can only be approached asymptotically
    and any CAGR that approaches zero in finite time must have extraordinarily steep initial drop, which is unrealistic).

    Remember that pd.Series are always well-behaved with pint[] quantities.  pd.DataFrame columns are well-behaved,
    but data across columns is not always well-behaved.  We therefore make this function assume we are projecting targets
    for a specific company, in a specific sector.  If we want to project targets for multiple sectors, we have to call it multiple times.
    This function doesn't need to know what sector it's computing for...only tha there is only one such, for however many scopes.
    """

    def __init__(self, projection_controls: ProjectionControls = ProjectionControls()):
        self.projection_controls = projection_controls

    def _order_scope_targets(self, scope_targets):
        if not scope_targets:
            # Nothing to do
            return scope_targets
        # If there are multiple targets that land on the same year for the same scope, choose the most recently set target
        unique_target_years = [
            (target.target_end_year, target.target_start_year)
            for target in scope_targets
        ]
        # This sorts targets into ascending target years and descending start years
        unique_target_years.sort(key=lambda t: (t[0], -t[1]))
        # Pick the first target year most recently articulated, preserving ascending order of target yeares
        unique_target_years = [
            (uk, next(v for k, v in unique_target_years if k == uk))
            for uk in dict(unique_target_years).keys()
        ]
        # Now use those pairs to select just the targets we want
        unique_scope_targets = [
            unique_targets[0]
            for unique_targets in [
                [
                    target
                    for target in scope_targets
                    if (target.target_end_year, target.target_start_year) == u
                ]
                for u in unique_target_years
            ]
        ]
        unique_scope_targets.sort(key=lambda target: (target.target_end_year))

        # We only trust the most recently communicated netzero target, but prioritize the most recently communicated, most aggressive target
        netzero_scope_targets = [
            target for target in unique_scope_targets if target.netzero_year
        ]
        netzero_scope_targets.sort(key=lambda t: (-t.target_start_year, t.netzero_year))
        if netzero_scope_targets:
            netzero_year = netzero_scope_targets[0].netzero_year
            for target in unique_scope_targets:
                target.netzero_year = netzero_year
        return unique_scope_targets

    def calculate_nz_target_years(self, targets: List[ITargetData]) -> dict:
        """Input:
        @target: A list of stated carbon reduction targets
        @returns: A dict of SCOPE_NAME: NETZERO_YEAR pairs
        """
        # We first try to find the earliest netzero year target for each scope
        nz_target_years = {
            "S1": 9999,
            "S2": 9999,
            "S1S2": 9999,
            "S3": 9999,
            "S1S2S3": 9999,
        }
        for target in targets:
            scope_name = target.target_scope.name
            if (
                target.netzero_year is not None
                and target.netzero_year < nz_target_years[scope_name]
            ):
                nz_target_years[scope_name] = target.netzero_year
            if (
                target.target_reduction_pct == 1.0
                and target.target_end_year < nz_target_years[scope_name]
            ):
                nz_target_years[scope_name] = target.target_end_year

        # We then infer netzero year targets for constituents of compound scopes from compound scopes
        # and infer netzero year taregts for compound scopes as the last of all constituents
        if nz_target_years["S1S2S3"] < nz_target_years["S1S2"]:
            logger.warning("target S1S2S3 date <= S1S2 date")
            nz_target_years["S1S2"] = nz_target_years["S1S2S3"]
        nz_target_years["S1"] = min(nz_target_years["S1S2"], nz_target_years["S1"])
        nz_target_years["S2"] = min(nz_target_years["S1S2"], nz_target_years["S2"])
        nz_target_years["S1S2"] = min(
            nz_target_years["S1S2"], max(nz_target_years["S1"], nz_target_years["S2"])
        )
        nz_target_years["S3"] = min(nz_target_years["S1S2S3"], nz_target_years["S3"])
        # nz_target_years['S1S2'] and nz_target_years['S3'] must both be <= nz_target_years['S1S2S3'] at this point
        nz_target_years["S1S2S3"] = max(nz_target_years["S1S2"], nz_target_years["S3"])
        return {
            scope_name: nz_year if nz_year < 9999 else None
            for scope_name, nz_year in nz_target_years.items()
        }

    def _get_ei_projections_from_ei_realizations(self, ei_realizations, i):
        for j in range(0, i + 1):
            if ei_realizations[
                j
            ].year >= self.projection_controls.BASE_YEAR and not ITR.isna(
                ei_realizations[j].value
            ):
                break
        model_ei_projections = [
            ICompanyEIProjection(
                year=ei_realizations[k].year, value=ei_realizations[k].value
            )
            # NaNs in the middle may still be a problem!
            for k in range(j, i + 1)
        ]
        while model_ei_projections[0].year > self.projection_controls.BASE_YEAR:
            model_ei_projections = [
                ICompanyEIProjection(
                    year=model_ei_projections[0].year - 1,
                    value=model_ei_projections[0].value,
                )
            ] + model_ei_projections
        return model_ei_projections

    def project_ei_targets(
        self,
        company: ICompanyData,
        production_proj: pd.Series,
        ei_df_t: pd.DataFrame = None,
    ) -> ICompanyEIProjectionsScopes:
        """Input:
        @company: Company-specific data: target_data and base_year_production
        @production_proj: company's production projection computed from region-sector benchmark growth rates

        If the company has no target or the target can't be processed, then the output the emission database, unprocessed
        If successful, it returns the full set of historic emissions intensities and projections based on targets
        """
        if company.target_data is None:
            targets = []
        else:
            targets = company.target_data
        target_scopes = {t.target_scope for t in targets}
        ei_projection_scopes: Dict[str, ICompanyEIProjections | None] = {
            "S1": None,
            "S2": None,
            "S1S2": None,
            "S3": None,
            "S1S2S3": None,
        }
        if (
            not ei_df_t.empty
            and EScope.S1 in target_scopes
            and EScope.S2 not in target_scopes
            and EScope.S1S2 not in target_scopes
        ):
            # We could create an S1S2 target based on S1 and S2 targets, but we don't yet
            # Syntehsize an S2 target using benchmark-aligned data
            s2_ei = ei_df_t.loc[:, (EScope.S2)]
            s2_netzero_year = s2_ei.idxmin()
            for target in targets:
                if target.target_scope == EScope.S1:
                    s2_target_base_year = max(target.target_base_year, s2_ei.index[0])
                    s2_target_base_m = s2_ei[s2_target_base_year].m
                    if ITR.HAS_UNCERTAINTIES:
                        s2_target_base_err = s2_ei[s2_target_base_year].m
                    else:
                        s2_target_base_err = None
                    s2_target_start_year = max(
                        target.target_start_year, s2_target_base_year
                    )
                    s2_target = ITargetData(
                        netzero_year=s2_netzero_year,
                        target_type="intensity",
                        target_scope=EScope.S2,
                        target_start_year=s2_target_start_year,
                        target_base_year=s2_target_base_year,
                        target_end_year=target.target_end_year,
                        target_base_year_qty=s2_target_base_m,
                        target_base_year_err=s2_target_base_err,
                        target_base_year_unit=str(s2_ei[s2_target_base_year].u),
                        target_reduction_pct=1.0
                        - (s2_ei[target.target_end_year] / s2_ei[s2_target_base_year]),
                    )
                    targets.append(s2_target)

        nz_target_years = self.calculate_nz_target_years(targets)

        for scope_name in ei_projection_scopes:
            netzero_year = nz_target_years[scope_name]
            # If there are no other targets specified (which can happen when we are dealing with inferred netzero targets)
            # target_year and target_ei_value pick up the year and value of the last EI realized
            # Otherwise, they are specified by the targets (intensity or absolute)
            target_year = 9999
            target_ei_value = Q_(np.nan, "dimensionless")

            scope_targets = [
                target for target in targets if target.target_scope.name == scope_name
            ]
            no_scope_targets = scope_targets == []
            # If we don't have an explicit scope target but we do have an implicit netzero target that applies to this scope,
            # prime the pump for projecting that netzero target, in case we ever need such a projection.  For example,
            # a netzero target for S1+S2 implies netzero targets for both S1 and S2.  The TPI benchmark needs an S1 target
            # for some sectors, and projecting a netzero target for S1 from S1+S2 makes that benchmark useable.
            # Note that we can only infer separate S1 and S2 targets from S1+S2 targets when S1+S2 = 0, because S1=0 + S2=0 is S1+S2=0
            if no_scope_targets:
                if company.historic_data.empty:
                    # This just defends against poorly constructed test cases
                    nz_target_years[scope_name] = None
                    continue
                if nz_target_years[scope_name]:
                    if (
                        company.projected_intensities[scope_name]
                        and not company.historic_data.emissions_intensities[scope_name]
                    ):
                        ei_projection_scopes[scope_name] = (
                            company.projected_intensities[scope_name]
                        )
                        continue
                    ei_realizations = company.historic_data.emissions_intensities[
                        scope_name
                    ]
                    # We can infer a netzero target.  Use our last year historic year of data as the target_year (i.e., target_base_year) value
                    # Due to ragged right edge, we have to hunt.  But we know there's at least one such value.
                    # If there's a proper target for this scope, historic values will be replaced by target values
                    for i in range(len(ei_realizations) - 1, -1, -1):
                        target_ei_value = ei_realizations[i].value
                        if ITR.isna(target_ei_value):
                            continue
                        model_ei_projections = (
                            self._get_ei_projections_from_ei_realizations(
                                ei_realizations, i
                            )
                        )
                        ei_projection_scopes[scope_name] = ICompanyEIProjections(
                            ei_metric=f"{target_ei_value.u:~P}",
                            projections=self._get_bounded_projections(
                                model_ei_projections
                            ),
                        )
                        if not ITR.isna(target_ei_value):
                            target_year = ei_realizations[i].year
                            break
                    if target_year == 9999:
                        # Either no realizations or they are all NaN
                        continue
                    # FIXME: if we have aggressive targets for source of this inference, the inferred
                    # netzero targets may be very slack (because non-netzero targets are not part of the inference)
            scope_targets_intensity = self._order_scope_targets(
                [
                    target
                    for target in scope_targets
                    if target.target_type == "intensity"
                ]
            )
            scope_targets_absolute = self._order_scope_targets(
                [target for target in scope_targets if target.target_type == "absolute"]
            )
            while scope_targets_intensity or scope_targets_absolute:
                if scope_targets_intensity and scope_targets_absolute:
                    target_i = scope_targets_intensity[0]
                    target_a = scope_targets_absolute[0]
                    if target_i.target_end_year == target_a.target_end_year:
                        if target_i.target_start_year >= target_a.target_start_year:
                            if target_i.target_start_year == target_a.target_start_year:
                                warnings.warn(
                                    f"intensity target overrides absolute target for \
                                    target_start_year={target_i.target_start_year} and \
                                    target_end_year={target_i.target_end_year}"
                                )
                            scope_targets_absolute.pop(0)
                            scope_targets = scope_targets_intensity
                        else:
                            scope_targets_intensity.pop(0)
                            scope_targets = scope_targets_absolute
                    elif target_i.target_end_year < target_a.target_end_year:
                        scope_targets = scope_targets_intensity
                    else:
                        scope_targets = scope_targets_absolute
                elif not scope_targets_intensity:
                    scope_targets = scope_targets_absolute
                else:  # not scope_targets_absolute
                    scope_targets = scope_targets_intensity

                target = scope_targets.pop(0)
                # Work-around for https://github.com/hgrecco/pint/issues/1687
                target_base_year_unit = ureg.parse_units(target.target_base_year_unit)

                # Solve for intensity and absolute
                model_ei_projections = None
                if target.target_type == "intensity":
                    # Simple case: the target is in intensity
                    # If target is not the first one for this scope, we continue from last year of the previous target
                    if ei_projection_scopes[scope_name]:
                        (_, last_ei_year), (_, last_ei_value) = ei_projection_scopes[
                            scope_name
                        ].projections[-1]  # type: ignore
                        last_ei_value = last_ei_value.to(target_base_year_unit)
                        skip_first_year = 1
                    else:
                        # When starting from scratch, use recent historic data if available.
                        if company.historic_data.empty:
                            ei_realizations = []
                        else:
                            ei_realizations = (
                                company.historic_data.emissions_intensities[scope_name]
                            )
                        skip_first_year = 0
                        if ei_realizations == []:
                            # Alas, we have no data to align with constituent or containing scope
                            last_ei_year = target.target_base_year
                            target_base_year_m = target.target_base_year_qty
                            if (
                                ITR.HAS_UNCERTAINTIES
                                and target.target_base_year_err is not None
                            ):
                                target_base_year_m = ITR.ufloat(
                                    target_base_year_m, target.target_base_year_err
                                )
                            last_ei_value = Q_(
                                target_base_year_m, target_base_year_unit
                            )
                        else:
                            for i in range(len(ei_realizations) - 1, -1, -1):
                                last_ei_year, last_ei_value = (
                                    ei_realizations[i].year,
                                    ei_realizations[i].value,
                                )
                                if ITR.isna(last_ei_value):
                                    continue
                                model_ei_projections = (
                                    self._get_ei_projections_from_ei_realizations(
                                        ei_realizations, i
                                    )
                                )
                                ei_projection_scopes[scope_name] = (
                                    ICompanyEIProjections(
                                        ei_metric=f"{last_ei_value.u:~P}",
                                        projections=self._get_bounded_projections(
                                            model_ei_projections
                                        ),
                                    )
                                )
                                skip_first_year = 1
                                break
                            if last_ei_year < target.target_base_year:
                                logger.error(
                                    f"Target data for {company.company_id} more up-to-date than disclosed data; please fix and re-run"
                                )
                                # breakpoint()
                                raise ValueError
                    target_year = target.target_end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_base_year_m = target.target_base_year_qty
                    if (
                        ITR.HAS_UNCERTAINTIES
                        and target.target_base_year_err is not None
                    ):
                        target_base_year_m = ITR.ufloat(
                            target_base_year_m, target.target_base_year_err
                        )
                    target_ei_value = Q_(
                        target_base_year_m * (1 - target.target_reduction_pct),
                        target_base_year_unit,
                    )
                    last_ei_value = last_ei_value.to(target_base_year_unit)
                    if target_ei_value >= last_ei_value:
                        # We've already achieved target, so aim for the next one
                        target_year = last_ei_year
                        target_ei_value = last_ei_value
                        continue
                    CAGR = self._compute_CAGR(
                        last_ei_year, last_ei_value, target_year, target_ei_value
                    )
                    model_ei_projections = [
                        ICompanyEIProjection(year=year, value=CAGR[year])
                        for year in range(
                            last_ei_year + skip_first_year, 1 + target_year
                        )
                        if year >= self.projection_controls.BASE_YEAR
                    ]

                elif target.target_type == "absolute":
                    # Complicated case, the target must be switched from absolute value to intensity.
                    # We use benchmark production data

                    # If target is not the first one for this scope, we continue from last year of the previous target
                    if ei_projection_scopes[scope_name]:
                        (_, last_ei_year), (_, last_ei_value) = ei_projection_scopes[
                            scope_name
                        ].projections[-1]  # type: ignore
                        last_prod_value = production_proj.loc[last_ei_year]
                        last_em_value = last_ei_value * last_prod_value
                        last_em_value = last_em_value.to(target_base_year_unit)
                        skip_first_year = 1
                    else:
                        if company.historic_data.empty:
                            em_realizations = []
                        else:
                            em_realizations = company.historic_data.emissions[
                                scope_name
                            ]
                        skip_first_year = 0
                        # Put these variables into scope with initial conditions that will be overwritten if we have em_realizations
                        last_ei_year = target.target_base_year
                        target_base_year_m = target.target_base_year_qty
                        if (
                            ITR.HAS_UNCERTAINTIES
                            and target.target_base_year_err is not None
                        ):
                            target_base_year_m = ITR.ufloat(
                                target_base_year_m, target.target_base_year_err
                            )
                        last_em_value = Q_(target_base_year_m, target_base_year_unit)
                        # FIXME: assert company.base_year_production == target.base_year_production
                        last_prod_value = company.base_year_production
                        if em_realizations:
                            for i in range(len(em_realizations) - 1, -1, -1):
                                last_ei_year, last_em_value = (
                                    em_realizations[i].year,
                                    em_realizations[i].value,
                                )
                                if ITR.isna(last_em_value):
                                    continue
                                # Just like _get_ei_projections_from_ei_realizations, except these are based on em_realizations, not ei_realizations
                                for j in range(0, i + 1):
                                    if (
                                        em_realizations[j].year
                                        >= self.projection_controls.BASE_YEAR
                                        and not ITR.isna(em_realizations[j].value)
                                    ):
                                        break
                                model_ei_projections = [
                                    ICompanyEIProjection(
                                        year=em_realizations[k].year,
                                        value=em_realizations[k].value
                                        / production_proj.loc[em_realizations[k].year],
                                    )
                                    # NaNs in the middle may still be a problem!
                                    for k in range(j, i + 1)
                                    if em_realizations[k].year
                                ]
                                while (
                                    model_ei_projections[0].year
                                    > self.projection_controls.BASE_YEAR
                                ):
                                    model_ei_projections = [
                                        ICompanyEIProjection(
                                            year=model_ei_projections[0].year - 1,
                                            value=model_ei_projections[0].value,
                                        )
                                    ] + model_ei_projections
                                last_prod_value = production_proj.loc[last_ei_year]
                                ei_projection_scopes[scope_name] = (
                                    ICompanyEIProjections(
                                        ei_metric=f"{(last_em_value/last_prod_value).u:~P}",
                                        projections=self._get_bounded_projections(
                                            model_ei_projections
                                        ),
                                    )
                                )
                                skip_first_year = 1
                                break
                            assert last_ei_year >= target.target_base_year
                        last_ei_value = last_em_value / last_prod_value

                    target_year = target.target_end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_base_year_m = target.target_base_year_qty
                    if (
                        ITR.HAS_UNCERTAINTIES
                        and target.target_base_year_err is not None
                    ):
                        target_base_year_m = ITR.ufloat(
                            target_base_year_m, target.target_base_year_err
                        )
                    target_em_value = Q_(
                        target_base_year_m * (1 - target.target_reduction_pct),
                        target_base_year_unit,
                    )
                    if target_em_value >= last_em_value:
                        # We've already achieved target, so aim for the next one
                        target_year = last_ei_year
                        target_ei_value = last_ei_value
                        continue
                    CAGR = self._compute_CAGR(
                        last_ei_year, last_em_value, target_year, target_em_value
                    )

                    model_emissions_projections = CAGR.loc[
                        (last_ei_year + skip_first_year) : target_year
                    ]  # noqa: E203
                    emissions_projections = model_emissions_projections.astype(
                        f"pint[{target_base_year_unit}]"
                    )
                    idx = production_proj.index.intersection(
                        emissions_projections.index
                    )
                    ei_projections = (
                        emissions_projections.loc[idx] / production_proj.loc[idx]
                    )

                    model_ei_projections = [
                        ICompanyEIProjection(year=year, value=ei_projections[year])
                        for year in range(
                            last_ei_year + skip_first_year, 1 + target_year
                        )
                        if year >= self.projection_controls.BASE_YEAR
                    ]
                    if ei_projection_scopes[scope_name] is None:
                        while (
                            model_ei_projections[0].year
                            > self.projection_controls.BASE_YEAR
                        ):
                            model_ei_projections = [
                                ICompanyEIProjection(
                                    year=model_ei_projections[0].year - 1,
                                    value=model_ei_projections[0].value,
                                )
                            ] + model_ei_projections
                else:
                    # No target (type) specified
                    ei_projection_scopes[scope_name] = None
                    continue

                target_ei_value = model_ei_projections[-1].value
                if ei_projection_scopes[scope_name] is not None:
                    ic_eiproj = ei_projection_scopes[scope_name]
                    assert ic_eiproj is not None
                    assert ic_eiproj.projections is not None
                    ic_eiproj.projections.extend(model_ei_projections)  # type: ignore
                else:
                    while (
                        model_ei_projections[0].year
                        > self.projection_controls.BASE_YEAR
                    ):
                        model_ei_projections = [
                            ICompanyEIProjection(
                                year=model_ei_projections[0].year - 1,
                                value=model_ei_projections[0].value,
                            )
                        ] + model_ei_projections
                    ei_projection_scopes[scope_name] = ICompanyEIProjections(
                        ei_metric=f"{target_ei_value.u:~P}",
                        projections=self._get_bounded_projections(model_ei_projections),
                    )

                if scope_targets_intensity and scope_targets_intensity[0].netzero_year:
                    # Let a later target set the netzero year
                    continue
                if scope_targets_absolute and scope_targets_absolute[0].netzero_year:
                    # Let a later target set the netzero year
                    continue

            # Handle final netzero targets.  Note that any absolute zero target is also zero intensity target (so use target_ei_value)
            # TODO What if target is a 100% reduction.  Does it work whether or not netzero_year is set?
            if (
                netzero_year and netzero_year > target_year
            ):  # add in netzero target at the end
                netzero_qty = Q_(0.0, target_ei_value.u)
                if (
                    no_scope_targets
                    and scope_name in ["S1S2S3"]
                    and nz_target_years["S1S2"] <= netzero_year
                    and nz_target_years["S3"] <= netzero_year
                ):
                    if ei_projection_scopes["S1S2"] is None:
                        raise ValueError(
                            f"{company.company_id} is missing S1+S2 historic data for S1+S2 target"
                        )
                    if ei_projection_scopes["S3"] is None:
                        raise ValueError(
                            f"{company.company_id} is missing S3 historic data for S3 target"
                        )
                    ei_projections = [
                        ei_sum
                        for ei_sum in list(
                            map(
                                ICompanyEIProjection.add,
                                ei_projection_scopes["S1S2"].projections,
                                ei_projection_scopes["S3"].projections,
                            )
                        )
                        if ei_sum.year in range(1 + target_year, 1 + netzero_year)
                    ]
                elif (
                    no_scope_targets
                    and scope_name in ["S1S2"]
                    and nz_target_years["S1"] <= netzero_year
                    and nz_target_years["S2"] <= netzero_year
                ):
                    if ei_projection_scopes["S1"] is None:
                        raise ValueError(
                            f"{company.company_id} is missing S1 historic data for S1 target"
                        )
                    if ei_projection_scopes["S2"] is None:
                        raise ValueError(
                            f"{company.company_id} is missing S2 historic data for S2 target"
                        )
                    ei_projections = [
                        ei_sum
                        for ei_sum in list(
                            map(
                                ICompanyEIProjection.add,
                                ei_projection_scopes["S1"].projections,
                                ei_projection_scopes["S2"].projections,
                            )
                        )
                        if ei_sum.year in range(1 + target_year, 1 + netzero_year)
                    ]
                else:
                    CAGR = self._compute_CAGR(
                        target_year, target_ei_value, netzero_year, netzero_qty
                    )
                    ei_projections = [
                        ICompanyEIProjection(year=year, value=CAGR[year])
                        for year in range(1 + target_year, 1 + netzero_year)
                    ]
                if ei_projection_scopes[scope_name]:
                    ei_projection_scopes[scope_name].projections.extend(ei_projections)  # type: ignore
                else:
                    ei_projection_scopes[scope_name] = ICompanyEIProjections(
                        ei_metric=EI_Quantity(f"{target_ei_value.u:~P}"),
                        projections=self._get_bounded_projections(ei_projections),
                    )
                target_year = netzero_year
                target_ei_value = netzero_qty
            if (
                ei_projection_scopes[scope_name]
                and target_year < ProjectionControls.TARGET_YEAR
            ):
                # Assume everything stays flat until 2050
                ei_projection_scopes[scope_name].projections.extend(  # type: ignore
                    [
                        ICompanyEIProjection(year=year, value=target_ei_value)
                        for y, year in enumerate(
                            range(1 + target_year, 1 + ProjectionControls.TARGET_YEAR)
                        )
                    ]
                )

        # If we are production-centric, S3 and S1S2S3 targets will make their way into S1 and S1S2
        return ICompanyEIProjectionsScopes(**ei_projection_scopes)

    def _compute_CAGR(
        self,
        first_year: int,
        first_value: Quantity,
        last_year: int,
        last_value: Quantity,
    ) -> pd.Series:
        """Compute CAGR, returning pd.Series of the growth (or reduction) applied to first to converge with last
        :param first_year: the year of the first datapoint in the Calculation (most recent actual datapoint)
        :param first_value: the value of the first datapoint in the Calculation (most recent actual datapoint)
        :param last_year: the year of the final target
        :param last_value: the value of the final target

        :return: pd.Series index by the years from first_year:last_year, with units based on last_value (the target value)
        """
        period = last_year - first_year
        if period <= 0:
            return pd.Series(PA_([], dtype=f"pint[{first_value.u:~P}]"))
        if last_value >= first_value or first_value.m == 0:
            # If we have a slack target, i.e., target goal is actually above current data, clamp so CAGR computes as zero
            return pd.Series(
                PA_(np.ones(period + 1) * first_value.m, dtype=f"{first_value.u:~P}"),
                index=range(first_year, last_year + 1),
                name="CAGR",
            )

        # CAGR doesn't work well with large reductions, so solve with cases:
        CAGR_limit = 1 / 11.11
        # PintArrays make it easy to convert arrays of magnitudes to types, so ensure magnitude consistency
        first_value = first_value.to(last_value.u)
        if last_value < first_value * CAGR_limit:
            # - If CAGR target > 90% reduction, blend a linear reduction with CAGR to get CAGR-like shape that actually hits the target
            cagr_factor = CAGR_limit ** (1 / period)
            linear_factor = CAGR_limit * first_value.m - last_value.m
            cagr_data = [
                cagr_factor**y * first_value.m - linear_factor * (y / period)
                for y, year in enumerate(range(first_year, last_year + 1))
            ]
        else:
            if ITR.HAS_UNCERTAINTIES and (
                isinstance(first_value.m, ITR.UFloat)
                or isinstance(last_value.m, ITR.UFloat)
            ):
                if isinstance(first_value.m, ITR.UFloat):
                    first_nom = first_value.m.n
                    first_err = first_value.m.s
                else:
                    first_nom = first_value.m
                    first_err = 0.0
                if isinstance(last_value.m, ITR.UFloat):
                    last_nom = last_value.m.n
                    last_err = last_value.m.s
                else:
                    last_nom = last_value.m
                    last_err = 0.0
                cagr_factor_nom = (last_nom / first_nom) ** (1 / period)
                cagr_data = [
                    ITR.ufloat(
                        cagr_factor_nom**y * first_nom,
                        first_err * (period - y) / period + last_err * (y / period),
                    )
                    for y, year in enumerate(range(first_year, last_year + 1))
                ]
            else:
                # - If CAGR target <= 90% reduction, use CAGR model directly
                cagr_factor = (last_value / first_value).m ** (1 / period)
                cagr_data = [
                    cagr_factor**y * first_value.m
                    for y, year in enumerate(range(first_year, last_year + 1))
                ]
        cagr_result = pd.Series(
            PA_(np.array(cagr_data), dtype=f"{last_value.u:~P}"),
            index=range(first_year, last_year + 1),
            name="CAGR",
        )
        return cagr_result
