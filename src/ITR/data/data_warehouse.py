import logging
import warnings  # needed until apply behaves better with Pint quantities in arrays
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pint import DimensionalityError
from pydantic import ValidationError

import ITR

from ..configs import ColumnsConfig, LoggingConfig
from ..data import PA_, Q_
from ..data.data_providers import (
    CompanyDataProvider,
    IntensityBenchmarkDataProvider,
    ProductionBenchmarkDataProvider,
)
from ..data.osc_units import EmissionsQuantity, asPintDataFrame, delta_degC_Quantity
from ..interfaces import (
    DF_ICompanyEIProjections,
    EScope,
    ICompanyAggregates,
    ICompanyData,
    ICompanyEIProjection,
    ICompanyEIProjections,
    IEIRealization,
    IEmissionRealization,
    IHistoricData,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
LoggingConfig.add_config_to_logger(logger)


class DataWarehouse(ABC):
    """General data provider super class."""

    def __init__(
        self,
        company_data: CompanyDataProvider,
        benchmark_projected_production: Optional[ProductionBenchmarkDataProvider],
        benchmarks_projected_ei: Optional[IntensityBenchmarkDataProvider],
        estimate_missing_data: Optional[
            Callable[["DataWarehouse", ICompanyData], None]
        ] = None,
    ):
        """Create a new data warehouse instance.

        :param company_data: CompanyDataProvider
        :param benchmark_projected_production: ProductionBenchmarkDataProvider
        :param benchmarks_projected_ei: IntensityBenchmarkDataProvider
        """
        self.benchmark_projected_production = None
        self.benchmarks_projected_ei = None
        # benchmarks_projected_ei._EI_df_t is the (transposed) EI dataframe for the benchmark
        # benchmark_projected_production.get_company_projected_production(company_sector_region_scope) gives production data per company (per year)
        # multiplying these two gives aligned emissions data for the company, in case we want to add missing data based on sector averages
        self.company_data = company_data
        self.estimate_missing_data = estimate_missing_data
        # Place to stash historic data before doing PC-conversion so it can be retreived when switching to non-PC benchmarks
        self.orig_historic_data: Dict[str, Any] = {}
        self.company_scope: Dict[str, EScope] = {}

        # Production benchmark data is needed to project trajectories
        # Trajectories + Emissions Intensities benchmark data are needed to estimate missing S3 data
        # Target projections rely both on Production benchmark data and S3 estimated data
        # Production-centric manipulations must happen after targets have been projected
        if (
            benchmark_projected_production is not None
            and benchmark_projected_production.own_data
        ) or (benchmarks_projected_ei is not None and benchmarks_projected_ei.own_data):
            self.update_benchmarks(
                benchmark_projected_production, benchmarks_projected_ei
            )
            self._own_data = True
        else:
            self._own_data = False

    @property
    def own_data(self) -> bool:
        return self._own_data

    def _preserve_historic_data(self):
        for c in self.company_data._companies:
            self.orig_historic_data[c.company_id] = {
                "ghg_s1s2": c.ghg_s1s2,
                "ghg_s3": c.ghg_s3,
                "historic_data": IHistoricData(
                    productions=c.historic_data.productions,
                    emissions=c.historic_data.emissions.model_copy(),
                    emissions_intensities=c.historic_data.emissions_intensities.model_copy(),
                ),
                "projected_intensities": c.projected_intensities.model_copy(),
                "projected_targets": c.projected_targets.model_copy(),
            }
            orig_historic_data = self.orig_historic_data[c.company_id]
            for scope_name in EScope.get_scopes():
                intensity_scope = getattr(
                    orig_historic_data["projected_intensities"], scope_name, None
                )
                if intensity_scope is not None:
                    setattr(
                        orig_historic_data["projected_intensities"],
                        scope_name,
                        (
                            type(intensity_scope)(
                                ei_metric=intensity_scope.ei_metric,
                                projections=intensity_scope.projections.copy(),
                            )
                        ),
                    )
                target_scope = getattr(
                    orig_historic_data["projected_targets"], scope_name, None
                )
                if target_scope is not None:
                    setattr(
                        orig_historic_data["projected_targets"],
                        scope_name,
                        (
                            type(target_scope)(
                                ei_metric=target_scope.ei_metric,
                                projections=target_scope.projections.copy(),
                            )
                        ),
                    )

    def _restore_historic_data(self):
        for c in self.company_data._companies:
            orig_data = self.orig_historic_data[c.company_id]
            c.ghg_s1s2 = orig_data["ghg_s1s2"]
            c.ghg_s3 = orig_data["ghg_s3"]
            c.historic_data = orig_data["historic_data"]
            c.projected_intensities = orig_data["projected_intensities"]
            c.projected_targets = orig_data["projected_targets"]

    def update_benchmarks(
        self,
        benchmark_projected_production: Optional[ProductionBenchmarkDataProvider],
        benchmarks_projected_ei: Optional[IntensityBenchmarkDataProvider],
    ):
        """Update the benchmark data used in this instance of the DataWarehouse.  If there is no change, do nothing."""
        new_production_bm = new_ei_bm = new_prod_centric = False
        if benchmark_projected_production is None:
            pass
        if self.benchmark_projected_production is None or (
            self.benchmark_projected_production.own_data
            and self.benchmark_projected_production.benchmark_changed(
                benchmark_projected_production
            )
        ):
            self.benchmark_projected_production = benchmark_projected_production  # type: ignore
            new_production_bm = True

        if benchmarks_projected_ei is None:
            pass
        elif self.benchmarks_projected_ei is None:
            self.benchmarks_projected_ei = benchmarks_projected_ei  # type: ignore
            if benchmarks_projected_ei.own_data:
                new_ei_bm = True
        elif self.benchmarks_projected_ei.own_data:
            if benchmarks_projected_ei.own_data:
                if self.benchmarks_projected_ei.benchmarks_changed(
                    benchmarks_projected_ei
                ):
                    new_prod_centric = (
                        self.benchmarks_projected_ei.prod_centric_changed(
                            benchmarks_projected_ei
                        )
                    )
                    new_ei_bm = True
            self.benchmarks_projected_ei = benchmarks_projected_ei

        if not new_production_bm and not new_ei_bm:
            return

        assert self.benchmarks_projected_ei is not None

        # Production benchmark data is needed to project trajectories
        # Trajectories + Emissions Intensities benchmark data are needed to estimate missing S3 data
        # Target projections rely both on Production benchmark data and S3 estimated data
        # Production-centric benchmarks shift S3 data after trajectory and targets have been projected
        if new_production_bm:
            cd_pc = self.company_data.get_projection_controls()
            logger.info(
                f"new_production_bm calculating trajectories for {len(self.company_data.get_company_data())}"
                f"companies (times {len(EScope.get_scopes())} scopes times "
                f"{cd_pc.TARGET_YEAR-cd_pc.BASE_YEAR} years)"
            )
            self.company_data._validate_projected_trajectories(
                self.company_data.get_company_data(), self.benchmarks_projected_ei
            )

        if new_ei_bm and (
            new_companies := [
                c for c in self.company_data.get_company_data() if "+" in c.company_id
            ]
        ):
            self.company_data._allocate_emissions(
                new_companies,
                self.benchmarks_projected_ei,
                self.company_data.get_projection_controls(),
            )

        # If we are missing S3 (or other) data, fill in before projecting targets
        if new_ei_bm and self.estimate_missing_data is not None:
            logger.info("estimating missing data")
            for c in self.company_data.get_company_data():
                self.estimate_missing_data(self, c)

        # Changes to production benchmark requires re-calculating targets (which are production-dependent)
        if new_production_bm:
            cd_pc = self.company_data.get_projection_controls()
            logger.info(
                f"projecting targets for {len(self.company_data.get_company_data())} companies "
                f"(times {len(EScope.get_scopes())} scopes times {cd_pc.TARGET_YEAR-cd_pc.BASE_YEAR} years)"
            )
            self.company_data._calculate_target_projections(
                benchmark_projected_production, benchmarks_projected_ei
            )

        # If our benchmark is production-centric, migrate S3 data (including estimated S3 data) into S1S2
        # If we shift before we project, then S3 targets will not be projected correctly.
        if new_ei_bm and benchmarks_projected_ei.is_production_centric():
            logger.info(
                "Shifting S3 emissions data into S1 according to Production-Centric benchmark rules"
            )
            if self.orig_historic_data != {}:
                self._restore_historic_data()
            else:
                self._preserve_historic_data()
            for c in self.company_data.get_company_data():
                if c.ghg_s3:
                    # For Production-centric and energy-only data (except for Cement), convert all S3 numbers to S1 numbers
                    if not ITR.isna(c.ghg_s3):
                        c.ghg_s1s2 = c.ghg_s1s2 + c.ghg_s3
                    c.ghg_s3 = None  # Q_(0.0, c.ghg_s3.u)
                if not c.historic_data.empty:

                    def _adjust_historic_data(data, primary_scope_attr, data_adder):
                        if data[primary_scope_attr]:
                            pre_s3_data = [
                                p
                                for p in data[primary_scope_attr]
                                if p.year <= data.S3[0].year
                            ]
                            if len(pre_s3_data) == 0:
                                # Could not adjust
                                # breakpoint()
                                assert False
                                return
                            if len(pre_s3_data) > 1:
                                pre_s3_data = list(
                                    map(
                                        lambda x: type(x)(
                                            year=x.year,
                                            value=data.S3[0].value
                                            * x.value
                                            / pre_s3_data[-1].value,
                                        ),
                                        pre_s3_data[:-1],
                                    )
                                )
                                s3_data = pre_s3_data + data.S3
                            else:
                                s3_data = data.S3
                            setattr(
                                data,
                                primary_scope_attr,
                                list(
                                    map(data_adder, data[primary_scope_attr], s3_data)
                                ),
                            )
                        else:
                            setattr(data, primary_scope_attr, data.S3)

                    if c.historic_data.emissions.S3:
                        _adjust_historic_data(
                            c.historic_data.emissions, "S1", IEmissionRealization.add
                        )
                        _adjust_historic_data(
                            c.historic_data.emissions, "S1S2", IEmissionRealization.add
                        )
                        c.historic_data.emissions.S3 = []
                    if c.historic_data.emissions.S1S2S3:
                        # assert c.historic_data.emissions.S1S2 == c.historic_data.emissions.S1S2S3
                        c.historic_data.emissions.S1S2S3 = []
                    if c.historic_data.emissions_intensities.S3:
                        _adjust_historic_data(
                            c.historic_data.emissions_intensities,
                            "S1",
                            IEIRealization.add,
                        )
                        _adjust_historic_data(
                            c.historic_data.emissions_intensities,
                            "S1S2",
                            IEIRealization.add,
                        )
                        c.historic_data.emissions_intensities.S3 = []
                    if c.historic_data.emissions_intensities.S1S2S3:
                        # assert c.historic_data.emissions_intensities.S1S2 == c.historic_data.emissions_intensities.S1S2S3
                        c.historic_data.emissions_intensities.S1S2S3 = []
                if c.projected_intensities.S3:
                    c.projected_intensities._adjust_trajectories("S1")
                    c.projected_intensities._adjust_trajectories("S1S2")
                    c.projected_intensities.S3 = None
                    if c.projected_intensities.S1S2S3:
                        # assert c.projected_intensities.S1S2.projections == c.projected_intensities.S1S2S3.projections

                        c.projected_intensities.S1S2S3 = None
                if c.projected_targets.S3:
                    # For production-centric benchmarks, S3 emissions are counted against S1 (and/or the S1 in S1+S2)

                    if c.projected_targets.S1:
                        c.projected_targets._align_and_sum_projected_targets("S1")
                    try:
                        # S3 projected targets may have been synthesized from a netzero S1S2S3 target and might need to be date-aligned with S1S2
                        c.projected_targets._align_and_sum_projected_targets("S1S2")
                    except AttributeError:
                        if c.projected_targets.S2 is not None:
                            logger.warning(
                                f"Scope 1+2 target projections should have been created for {c.company_id}; repairing"
                            )
                            assert c.projected_targets.S1 is not None
                            if isinstance(
                                c.projected_targets.S1.projections, pd.Series
                            ) and isinstance(
                                c.projected_targets.S2.projections, pd.Series
                            ):
                                c.projected_targets.S1S2 = DF_ICompanyEIProjections(
                                    ei_metric=c.projected_targets.S1.ei_metric,
                                    projections=c.projected_targets.S1.projections
                                    + c.projected_targets.S2.projections,
                                )
                            else:
                                c.projected_targets.S1S2 = ICompanyEIProjections(
                                    ei_metric=c.projected_targets.S1.ei_metric,
                                    projections=list(
                                        map(
                                            ICompanyEIProjection.add,
                                            c.projected_targets.S1.projections,
                                            c.projected_targets.S2.projections,
                                        )
                                    ),
                                )
                        else:
                            logger.warning(
                                f"Scope 2 target projections missing from company with ID {c.company_id}; treating as zero"
                            )
                            assert c.projected_targets.S1 is not None
                            if isinstance(
                                c.projected_targets.S1.projections, pd.Series
                            ):
                                c.projected_targets.S1S2 = DF_ICompanyEIProjections(
                                    ei_metric=c.projected_targets.S1.ei_metric,
                                    projections=c.projected_targets.S1.projections,
                                )
                            else:
                                c.projected_targets.S1S2 = ICompanyEIProjections(
                                    ei_metric=c.projected_targets.S1.ei_metric,
                                    projections=c.projected_targets.S1.projections,
                                )
                        if c.projected_targets.S3:
                            c.projected_targets._align_and_sum_projected_targets("S1S2")
                        else:
                            logger.warning(
                                f"Scope 3 target projections missing from company with ID {c.company_id}; treating as zero"
                            )
                    except ValueError:
                        logger.error(
                            f"S1+S2 targets not aligned with S3 targets for company with ID {c.company_id}; ignoring S3 data"
                        )
                    c.projected_targets.S3 = None
                if c.projected_targets.S1S2S3:
                    # assert c.projected_targets.S1S2 == c.projected_targets.S1S2S3
                    c.projected_targets.S1S2S3 = None
        elif new_prod_centric and self.orig_historic_data != {}:
            # Switch to non-product-centric benchmark of this version.
            self._restore_historic_data()

        # Set scope information based on what company reports and what benchmark requres
        # benchmarks_projected_ei._EI_df_t makes life a bit easier...
        missing_company_scopes = []
        ei_df_t = benchmarks_projected_ei._get_intensity_benchmarks()
        for c in self.company_data.get_company_data():
            region = c.region
            try:
                _ = ei_df_t[(c.sector, region)]
            except KeyError:
                try:
                    region = "Global"
                    _ = ei_df_t[(c.sector, region)]
                except KeyError:
                    missing_company_scopes.append(c.company_id)
                    continue
            scopes = ei_df_t[(c.sector, region)].columns.tolist()
            if len(scopes) == 1:
                self.company_scope[c.company_id] = scopes[0]
                continue
            for scope in EScope.get_result_scopes():
                if scope in scopes:
                    self.company_scope[c.company_id] = scope
                    break
            if c.company_id not in self.company_scope:
                missing_company_scopes.append(c.company_id)

        if missing_company_scopes:
            logger.warning(
                f"The following companies do not disclose scope data required by benchmark and will be not be analyzed: {missing_company_scopes}"
            )

    def update_trajectories(self):
        """Update the trajectory calculations after changing global ProjectionControls.  Production and EI benchmarks remain the same."""
        # We cannot only update trajectories without regard for all that depend on those trajectories
        # For example, different benchmarks may have different scopes defined, units for benchmarks, etc.
        logger.info(
            f"re-calculating trajectories for {len(self.company_data._companies)} companies"
            f"\n    (times {len(EScope.get_scopes())} scopes times "
            f"{self.company_data.projection_controls.TARGET_YEAR-self.company_data.projection_controls.BASE_YEAR} years)"
        )
        for company in self.company_data._companies:
            company.projected_intensities = None
        self.company_data._validate_projected_trajectories(
            self.company_data._companies, self.benchmarks_projected_ei
        )

    def estimate_missing_s3_data(self, company: ICompanyData):
        # We need benchmark data to estimate S3 from projected_intensities (which go back in time to BASE_YEAR).
        # We don't need to estimate further back than that, as we don't rewrite values stored in historic_data.
        # Benchmarks that don't have S3 emissions (such as TPI for Electricity Utilities) take this early return
        assert company.projected_intensities is not None
        if (
            company.projected_intensities.S3
            or not self.benchmarks_projected_ei
            or EScope.S3 not in self.benchmarks_projected_ei.get_scopes()
        ):
            return

        # This is an estimation function, not a mathematical processing function.
        # It won't solve S3 = S1S2S3 - S1S2 (which should be done elsewhere)
        sector = company.sector
        region = company.region
        ei_df_t = self.benchmarks_projected_ei._get_intensity_benchmarks()
        if (sector, region) in ei_df_t.columns:
            pass
        elif (sector, "Global") in ei_df_t.columns:
            region = "Global"
        else:
            return
        if (
            sector,
            region,
            EScope.S3,
        ) not in ei_df_t.columns:
            if sector == "Construction Buildings":
                # Construction Buildings don't have an S3 scope defined
                return
            else:
                # Some benchmarks don't include S3 for all sectors; so nothing to estimate
                return
        else:
            bm_ei_s3 = ei_df_t[(sector, region, EScope.S3)]
        if (
            sector,
            region,
            EScope.S1S2,
        ) in ei_df_t.columns:
            # If we have only S1S2S3 emissions, we can "allocate" them according to the benchmark's allocation
            bm_ei_s1s2 = ei_df_t[(sector, region, EScope.S1S2)]
        else:
            bm_ei_s1s2 = None

        if company.projected_intensities.S1S2S3:
            assert company.projected_intensities.S1S2 is None
            assert company.projected_intensities.S1 is None
            ei_metric = company.projected_intensities.S1S2S3.ei_metric
            s1s2_projections = (
                company.projected_intensities.S1S2S3.projections
                * bm_ei_s1s2
                / (bm_ei_s1s2 + bm_ei_s3)
            )
            s3_projections = (
                company.projected_intensities.S1S2S3.projections
                * bm_ei_s3
                / (bm_ei_s1s2 + bm_ei_s3)
            )
            if ITR.HAS_UNCERTAINTIES:
                nominal_s3 = ITR.nominal_values(s3_projections.pint.quantity.m)
                std_dev_s3 = ITR.uarray(np.zeros(len(nominal_s3)), nominal_s3)
                s1s2_projections = pd.Series(
                    data=PA_(
                        s1s2_projections.pint.quantity.m + std_dev_s3,
                        dtype=s1s2_projections.dtype,
                    ),
                    index=s1s2_projections.index,
                )
                s3_projections = pd.Series(
                    data=PA_(
                        s3_projections.pint.quantity.m + std_dev_s3,
                        dtype=s3_projections.dtype,
                    ),
                    index=s3_projections.index,
                )
            company.projected_intensities.S1S2 = DF_ICompanyEIProjections(
                ei_metric=ei_metric, projections=s1s2_projections
            )
            company.projected_intensities.S3 = DF_ICompanyEIProjections(
                ei_metric=ei_metric, projections=s3_projections
            )
            company.ghg_s1s2 = (
                s1s2_projections[self.company_data.get_projection_controls().BASE_YEAR]
                * company.base_year_production
            ).to("t CO2e")
        else:
            # Penalize non-disclosure by assuming 2x aligned S3 emissions.  That's still likely undercounting, because
            # most non-disclosing companies are nowhere near the reduction rates of the benchmarks.
            # It would be lovely to use S1S2 or S1 data to inform S3, but that likely adds error on top of error
            assert company.projected_intensities.S1S2S3 is None
            ei_metric = str(bm_ei_s3.dtype.units)
            # If we don't have uncertainties, ITR.ufloat just returns the nom value 2.0
            s3_projections = bm_ei_s3 * ITR.ufloat(2.0, 1.0)
            company.projected_intensities.S3 = DF_ICompanyEIProjections(
                ei_metric=ei_metric, projections=s3_projections
            )
            if company.projected_intensities.S1S2 is not None:
                try:
                    company.projected_intensities.S1S2S3 = DF_ICompanyEIProjections(
                        ei_metric=ei_metric,
                        # Mismatched dimensionalities are not automatically converted
                        projections=s3_projections
                        + company.projected_intensities.S1S2.projections.astype(
                            f"pint[{ei_metric}]"
                        ),
                    )
                except DimensionalityError:
                    logger.error(
                        f"Company {company.company_id}'s S1+S2 intensity units "
                        f"({company.projected_intensities.S1S2.projections.dtype})"
                        f"are not compatible with benchmark units ({ei_metric})"
                    )

        company.ghg_s3 = (
            s3_projections[self.company_data.get_projection_controls().BASE_YEAR]
            * company.base_year_production
        ).to("t CO2e")
        logger.info(
            f"Added S3 estimates for {company.company_id} (sector = {sector}, region = {region})"
        )

    @classmethod
    def _process_company_data(
        cls,
        df_company_data: pd.DataFrame,
        projected_production: pd.DataFrame,
        projected_trajectories: pd.DataFrame,
        projected_targets: pd.DataFrame,
        base_year: int,
        target_year: int,
        budgeted_ei: pd.DataFrame,
        benchmark_temperature: delta_degC_Quantity,
        global_budget: EmissionsQuantity,
    ) -> pd.DataFrame:
        # trajectories are projected from historic data and we are careful to fill all gaps between historic and projections
        # FIXME: we just computed ALL company data above into a dataframe.  Why not use that?  Answer: because the following is inscrutible
        # (lambda xx: pd.DataFrame(data=list(xx.map(lambda x: { 'scope': x.name } | x.to_dict())), index=xx.index)) \
        #     (df_company_data.projected_intensities
        #      .map(lambda x: [pd.Series(x[scope]['projections'], name=getattr(EScope, scope))
        #                      for scope in ['S1', 'S1S2', 'S3', 'S1S2S3']
        #                      if x[scope] is not None]).explode()).set_index('scope', append=True)

        # If we have excess projections (compared to projected_production), _get_cumulative_emissions will drop them
        df_trajectory = DataWarehouse._get_cumulative_emissions(
            projected_ei=projected_trajectories,
            projected_production=projected_production,
        )

        # Ensure we haven't set any targets for scopes we are not prepared to deal with
        projected_targets = projected_targets.loc[
            projected_production.index.intersection(projected_targets.index)
        ]
        # Fill in ragged left edge of projected_targets with historic data, interpolating where we need to
        for year_col, company_ei_data in projected_targets.items():
            # company_ei_data is an unruly collection of unit types, so need to check NaN values row by row
            mask = company_ei_data.apply(lambda x: ITR.isna(x))
            if mask.all():
                # No sense trying to do anything with left-side all-NaN columns
                projected_targets = projected_targets.drop(columns=year_col)
                continue
            if mask.any():
                projected_targets.loc[mask[mask].index, year_col] = (
                    projected_trajectories.loc[mask[mask].index, year_col]
                )
            else:
                break

        df_target = DataWarehouse._get_cumulative_emissions(
            projected_ei=projected_targets, projected_production=projected_production
        )
        df_budget = DataWarehouse._get_cumulative_emissions(
            projected_ei=budgeted_ei, projected_production=projected_production
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
            base_year_scale = df_trajectory.loc[df_budget.index][base_year].mul(
                df_budget[base_year].map(
                    lambda x: Q_(0.0, f"1/({x.u})") if x.m == 0.0 else 1 / x
                )
            )
        df_scaled_budget = df_budget.mul(base_year_scale, axis=0)
        # FIXME: we calculate exceedance only against df_budget, not also df_scaled_budget
        # df_trajectory_exceedance = self._get_exceedance_year(df_trajectory, df_budget, self.company_data.projection_controls.TARGET_YEAR, None)
        # df_target_exceedance = self._get_exceedance_year(df_target, df_budget, self.company_data.projection_controls.TARGET_YEAR, None)
        df_trajectory_exceedance = DataWarehouse._get_exceedance_year(
            df_trajectory, df_budget, base_year, target_year, target_year
        )
        df_target_exceedance = DataWarehouse._get_exceedance_year(
            df_target, df_budget, base_year, target_year, target_year
        )
        df_scope_data = pd.concat(
            [
                df_trajectory.iloc[:, -1].rename(ColumnsConfig.CUMULATIVE_TRAJECTORY),
                df_target.iloc[:, -1].rename(ColumnsConfig.CUMULATIVE_TARGET),
                df_budget.iloc[:, -1].rename(ColumnsConfig.CUMULATIVE_BUDGET),
                df_scaled_budget.iloc[:, -1].rename(
                    ColumnsConfig.CUMULATIVE_SCALED_BUDGET
                ),
                df_trajectory_exceedance.rename(
                    f"{ColumnsConfig.TRAJECTORY_EXCEEDANCE_YEAR}"
                ),
                df_target_exceedance.rename(f"{ColumnsConfig.TARGET_EXCEEDANCE_YEAR}"),
            ],
            axis=1,
        )
        df_company_data = df_company_data.join(df_scope_data).reset_index("scope")
        na_company_mask = df_company_data.scope.isna()
        if na_company_mask.any():
            # Happens when the benchmark doesn't cover the company's supplied scopes at all
            logger.warning(
                f"Dropping companies with no scope data: {df_company_data[na_company_mask].index.get_level_values(level='company_id').to_list()}"
            )
            df_company_data = df_company_data[~na_company_mask]
        df_company_data[ColumnsConfig.BENCHMARK_GLOBAL_BUDGET] = pd.Series(
            PA_([global_budget.m] * len(df_company_data), dtype=str(global_budget.u)),
            index=df_company_data.index,
        )
        # ICompanyAggregates wants this Quantity as a `str`
        df_company_data[ColumnsConfig.BENCHMARK_TEMP] = [
            str(benchmark_temperature)
        ] * len(df_company_data)
        return df_company_data

    def get_preprocessed_company_data(
        self, company_ids: List[str]
    ) -> List[ICompanyAggregates]:
        """Get all relevant data for a list of company ids. This method should return a list of ICompanyAggregates
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data and additional precalculated fields
        """
        company_data = self.company_data.get_company_data(company_ids)
        df_company_data = pd.DataFrame.from_records(
            [dict(c) for c in company_data]
        ).set_index(self.company_data.column_config.COMPANY_ID, drop=False)
        valid_company_ids = df_company_data.index.to_list()

        # Why does the following create ICompanyData?  Because get_company_fundamentals is getting base_year_production, ghg_s1s2, and ghg_s3 the hard way
        company_info_at_base_year = (
            self.company_data.get_company_intensity_and_production_at_base_year(
                valid_company_ids
            )
        )

        cd_pc = self.company_data.get_projection_controls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/128
            assert self.benchmark_projected_production is not None
            projected_production = (
                self.benchmark_projected_production.get_company_projected_production(
                    company_info_at_base_year
                )
            )
            target_year_loc = projected_production.columns.get_loc(cd_pc.TARGET_YEAR)
            projected_production = projected_production.iloc[:, 0 : target_year_loc + 1]  # noqa: E203

        assert self.benchmarks_projected_ei is not None
        df_company_data = self._process_company_data(
            df_company_data,
            projected_production=projected_production,
            projected_trajectories=self.company_data.get_company_projected_trajectories(
                valid_company_ids
            ),
            projected_targets=self.company_data.get_company_projected_targets(
                valid_company_ids
            ),
            base_year=cd_pc.BASE_YEAR,
            target_year=cd_pc.TARGET_YEAR,
            budgeted_ei=self.benchmarks_projected_ei.get_SDA_intensity_benchmarks(
                company_info_at_base_year
            ),
            benchmark_temperature=self.benchmarks_projected_ei.benchmark_temperature,
            global_budget=self.benchmarks_projected_ei.benchmark_global_budget,
        )

        # This was WICKED SLOW: aggregate_company_data = [ICompanyAggregates.parse_obj(company) for company in companies]
        aggregate_company_data = [
            ICompanyAggregates.from_ICompanyData(company, scope_company_data)
            for company in company_data
            for scope_company_data in df_company_data.loc[[company.company_id]][
                [
                    "cumulative_budget",
                    "cumulative_scaled_budget",
                    "cumulative_trajectory",
                    "cumulative_target",
                    "benchmark_temperature",
                    "benchmark_global_budget",
                    "scope",
                    "trajectory_exceedance_year",
                    "target_exceedance_year",
                ]
            ].to_dict(orient="records")
        ]
        return aggregate_company_data

    def _convert_df_to_model(
        self, df_company_data: pd.DataFrame
    ) -> List[ICompanyAggregates]:
        """Transforms Dataframe Company data and preprocessed values into list of ICompanyAggregates instances

        :param df_company_data: pandas Dataframe with targets
        :return: A list containing the targets
        """
        df_company_data = df_company_data.where(
            pd.notnull(df_company_data), None
        ).replace({np.nan: None})  # set NaN to None since NaN is float instance
        companies_data_dict = df_company_data.to_dict(orient="records")
        model_companies: List[ICompanyAggregates] = []
        for company_data in companies_data_dict:
            try:
                model_companies.append(ICompanyAggregates.model_validate(company_data))
            except ValidationError:
                logger.warning(
                    "(one of) the input(s) of company %s is invalid and will be skipped"
                    % company_data[self.company_data.column_config.COMPANY_NAME]
                )
                pass
        return model_companies

    @classmethod
    def _get_cumulative_emissions(
        cls, projected_ei: pd.DataFrame, projected_production: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the weighted sum of the projected emission
        :param projected_ei: Rows of projected emissions intensities indexed by (company_id, scope)
        :param projected_production: Rows of projected production amounts indexed by (company_id, scope)
        :return: cumulative emissions, by year, based on weighted sum of emissions intensity * production
        """
        # The old and slow way:
        # projected_ei_t = asPintDataFrame(projected_ei.T)
        # projected_prod_t = asPintDataFrame(projected_production.T)
        # projected_emissions_t = projected_ei_t.mul(projected_prod_t.loc[projected_ei_t.index, projected_ei.T.columns])
        # cumulative_emissions = projected_emissions_t.T.cumsum(axis=1).astype('pint[Mt CO2]')

        # Ensure that projected_production is ordered the same as projected_ei, preserving order of projected_ei
        # projected_production is constructed to be limited to the years we want to analyze
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
            proj_prod_t = asPintDataFrame(
                projected_production.loc[projected_ei.index].T
            )
            # Limit projected_ei to the year range of projected_production
            proj_ei_t = asPintDataFrame(projected_ei[proj_prod_t.index].T)
            units_CO2e = "t CO2e"
            # We use pd.concat because pd.combine reverts our PintArrays into object arrays :-/
            proj_CO2e_m_t = pd.concat(
                [
                    ITR.data.osc_units.align_production_to_bm(
                        proj_prod_t[col], proj_ei_t[col]
                    )
                    .mul(proj_ei_t[col])
                    .pint.m_as(units_CO2e)
                    for col in proj_ei_t.columns
                ],
                axis=1,
            )
            # pd.concat names parameter refers to index.names; there's no way to set columns.names
            proj_CO2e_m_t.columns.names = proj_ei_t.columns.names
            if ITR.HAS_UNCERTAINTIES:
                # Sum both the nominal and std_dev values, because these series are completely correlated
                # Note that NaNs in this dataframe will be nan+/-nan, showing up in both nom and err
                nom_CO2e_m_t = proj_CO2e_m_t.apply(ITR.nominal_values).cumsum()
                err_CO2e_m_t = proj_CO2e_m_t.apply(ITR.std_devs).cumsum()
                cumulative_emissions_m_t = nom_CO2e_m_t.combine(
                    err_CO2e_m_t, ITR.recombine_nom_and_std
                )
            else:
                cumulative_emissions_m_t = proj_CO2e_m_t.cumsum()
            return cumulative_emissions_m_t.T.astype(f"pint[{units_CO2e}]")

    @classmethod
    def _get_exceedance_year(
        self,
        df_subject: pd.DataFrame,
        df_budget: pd.DataFrame,
        base_year: int,
        target_year: int,
        budget_year: Optional[int],
    ) -> pd.Series:
        """:param df_subject: DataFrame of cumulative emissions values over time
        :param df_budget: DataFrame of cumulative emissions budget allowed over time
        :param budget_year: if not None, set the exceedence budget to that year; otherwise budget starts low and grows year-by-year
        :return: The furthest-out year where df_subject < df_budget, or np.nan if none

        Where the (df_subject-aligned) budget defines a value but df_subject doesn't have a value, return pd.NA
        Where the benchmark (df_budget) fails to provide a metric for the subject scope, return no rows
        """
        missing_subjects = df_budget.index.difference(df_subject.index)
        aligned_rows = df_budget.index.intersection(df_subject.index)
        # idxmax returns the first maximum of a series, but we want the last maximum of a series
        # Reversing the columns, the maximum remains the maximum, but the "first" is the furthest-out year

        df_subject = df_subject.loc[aligned_rows, ::-1].pint.dequantify()
        df_budget = df_budget.loc[aligned_rows, ::-1].pint.dequantify()
        # units are embedded in the column multi-index, so this check validates dequantify operation post-hoc
        assert (df_subject.columns == df_budget.columns).all()
        if budget_year:
            df_exceedance_budget = pd.DataFrame(
                {
                    (year, units): df_budget[(budget_year, units)]
                    for year, units in df_budget.columns
                    if year < budget_year
                }
            )
            df_budget.update(df_exceedance_budget)
        # pd.where operation requires DataFrames to be aligned
        df_aligned = df_subject.where(df_subject <= df_budget)
        # Drop the embedded units from the multi-index and find the first (meaning furthest-out) date of alignment
        df_aligned = df_aligned.droplevel(1, axis=1).apply(
            lambda x: x.first_valid_index(), axis=1
        )

        if len(missing_subjects):
            df_aligned = pd.concat(
                [
                    df_aligned,
                    pd.Series(
                        data=[pd.NA] * len(missing_subjects), index=missing_subjects
                    ),
                ]
            )
        df_exceedance = df_aligned.map(
            lambda x: base_year if ITR.isna(x) else pd.NA if x >= target_year else x
        ).astype("Int64")
        return df_exceedance
