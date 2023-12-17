from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Type

import pandas as pd

import ITR

from ..configs import ColumnsConfig, LoggingConfig, ProjectionControls  # noqa F401
from ..data.base_providers import BaseCompanyDataProvider
from ..data.osc_units import (
    fx_ctx,
)
from ..data.osc_units import EmissionsQuantity, Quantity_type, delta_degC_Quantity
from ..interfaces import (  # noqa F401
    EScope,
    ICompanyData,
    ICompanyEIProjection,
    ICompanyEIProjections,
    ICompanyEIProjectionsScopes,
    IEIRealization,
    IEmissionRealization,
    IHistoricData,
    IHistoricEIScopes,
    IHistoricEmissionsScopes,
    IProductionRealization,
    ITargetData,
)
from ..utils import requantify_df

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class NZDPU_CompanyDataProvider(BaseCompanyDataProvider):
    """
    Company data provider super class.
    Data container for company specific data. It expects both Fundamental (e.g. Company revenue, marktetcap etc) and
    emission and target data per company.

    Initialized CompanyDataProvider is required when setting up a data warehouse instance.
    """

    def __init__(
        self,
        excel_path: str,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
    ):
        self.nzdpu_start_year = None
        self.projection_controls = projection_controls
        # The initial population of companies' data
        if excel_path:
            self._own_data = True
            self._companies = self._init_from_nzdpu_company_data(excel_path)
            super().__init__(self._companies, column_config, projection_controls)
            # The perfection of historic ESG data (adding synthethic company sectors, dropping those with missing data)
            self._companies = self._convert_from_nzdpu_company_data()
        else:
            self._own_data = False
            self._companies = []

    def _init_from_nzdpu_company_data(self, excel_path: str):
        """
        Converts first sheet of Excel template to list of minimal ICompanyData objects (fundamental data, but no ESG data).
        All dataprovider features will be inhereted from Base.
        :param excel_path: file path to excel file
        """

        df_nzdpu_data = pd.read_excel("~/Downloads/nzdpu_sample_data/nzdpu_data_sample.xlsx", sheet_name=None, skiprows=[1])
        df_fundamentals = pd.DataFrame()
        df_esg = pd.DataFrame()

        try:
            df = df_nzdpu_data['METADATA']
        except KeyError as e:
            logger.error(f"Tab {input_data_sheet} is required in input Excel file.")
            raise KeyError

        # We build `esg_data_sheet` by stacking together SCOPE1, SCOPE2, and SCOPE3 disclosures
        try:
            df_scope1 = requantify_df(df_nzdpu_data['SCOPE 1 EMISSIONS'].drop(
                columns=["data_model", "scope_1_methodology", "scope_1_change_type", "scope_1_change_description"]
            ))
            df_scope2_location_based = requantify_df(df_nzdpu_data['SCOPE 2 LB EMISSIONS'].drop(
                columns=["data_model", "rationale_s2_lb_non_disclose", "scope_2_lb_methodology", "scope_2_lb_change_type", "scope_2_lb_change_description"]
            )).drop(columns="total_scope_2_lb_emissions_co2")
            df_scope2_market_based = requantify_df(df_nzdpu_data['SCOPE 2 MB EMISSIONS'].drop(
                columns=["data_model", "rationale_s2_mb_non_disclose", "scope_2_mb_methodology", "scope_2_mb_change_type", "scope_2_mb_change_description"]
            )).drop(columns="total_scope_2_mb_emissions_co2")
            df_scope3 = requantify_df(df_nzdpu_data['SCOPE 3 EMISSIONS'][
                ["legal_entity_identifier", "company_name", "reporting_year", "org_boundary_approach", "total_scope_3_emissions_ghg", "total_scope_3_emissions_ghg units"]
            ])

        except KeyError as e:
            logger.error(f"{e} raised looking for tab in input Excel file.")
            raise KeyError
        df_scope1[ColumnsConfig.COMPANY_ID] = df_scope1.legal_entity_identifier
        df_scope1.rename(columns={"legal_entity_identifier":ColumnsConfig.COMPANY_LEI, "reporting_year":"year"}, inplace=True)
        df_scope2_location_based[ColumnsConfig.COMPANY_ID] = df_scope2_location_based.legal_entity_identifier
        df_scope2_location_based.rename(
            columns={
                "legal_entity_identifier":ColumnsConfig.COMPANY_LEI,
                "total_scope_2_lb_emissions_ghg":"total_scope_2_emissions_ghg",
                "reporting_year":"year",
            },
            inplace=True,
        )
        df_scope2_market_based[ColumnsConfig.COMPANY_ID] = df_scope2_market_based.legal_entity_identifier
        df_scope2_market_based.rename(
            columns={
                "legal_entity_identifier":ColumnsConfig.COMPANY_LEI,
                "total_scope_2_mb_emissions_ghg":"total_scope_2_emissions_ghg",
                "reporting_year":"year",
            },
            inplace=True,
        )
        df_scope2 = df_scope2_location_based.set_index([ColumnsConfig.COMPANY_ID, "year"]).combine_first(
            df_scope2_market_based.set_index([ColumnsConfig.COMPANY_ID, "year"])
        ).reset_index()
        df_scope3[ColumnsConfig.COMPANY_ID] = df_scope3.legal_entity_identifier
        df_scope3.rename(
            columns={
                "legal_entity_identifier":ColumnsConfig.COMPANY_LEI,
                "reporting_year":"year",
            },
            inplace=True,
        )

        # testing if all data is in the same currency
        fundamental_metrics = [
            "company_market_cap",
            "company_revenue",
            "company_enterprise_value",
            "company_ev_plus_cash",
            "company_total_assets",
        ]

        # are there empty sectors?

        # testing if only valid sectors are provided

        # Checking missing company ids

        # Checking if there are not any missing market cap

        # For the missing Market Cap we should use the ratio below to get dummy market cap:
        #   (Avg for the Sector (Market Cap / Revenues) + Avg for the Sector (Market Cap / Assets)) 2
        if False:
            df_fundamentals["MCap_to_Reven"] = (
                df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP] / df_fundamentals[ColumnsConfig.COMPANY_REVENUE]
            )  # new temp column with ratio
            df_fundamentals["MCap_to_Assets"] = (
                df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP] / df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS]
            )  # new temp column with ratio
            df_fundamentals["AVG_MCap_to_Reven"] = df_fundamentals.groupby(ColumnsConfig.SECTOR)["MCap_to_Reven"].transform(
                "mean"
            )
            df_fundamentals["AVG_MCap_to_Assets"] = df_fundamentals.groupby(ColumnsConfig.SECTOR)[
                "MCap_to_Assets"
            ].transform("mean")
            # FIXME: Add uncertainty here!
            df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP] = df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP].fillna(
                0.5
                * (
                    df_fundamentals[ColumnsConfig.COMPANY_REVENUE] * df_fundamentals["AVG_MCap_to_Reven"]
                    + df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS] * df_fundamentals["AVG_MCap_to_Assets"]
                )
            )
            df_fundamentals.drop(
                columns=[
                    "MCap_to_Reven",
                    "MCap_to_Assets",
                    "AVG_MCap_to_Reven",
                    "AVG_MCap_to_Assets",
                ],
                inplace=True,
            )  # deleting temporary columns

            if missing_cap_ids:
                logger.warning(
                    f"Missing market capitalisation values are estimated for companies with ID: " f"{missing_cap_ids}."
                )

        try:
            target_abs_cols = ["legal_entity_identifier", "company_name", "reporting_year", "org_boundary_approach",
                               "tgt_abs_id", "tgt_abs_name", "tgt_abs_status", "tgt_abs_year_set" ]
            target_int_cols = ["legal_entity_identifier", "company_name", "reporting_year", "org_boundary_approach",
                               "tgt_int_id", "tgt_int_name", "tgt_int_status", "tgt_int_year_set" ]
            target_cols = [ "tgt_TYPE_coverage_scope", "tgt_TYPE_coverage_s3_cat",
                           "tgt_TYPE_coverage_perc_s1", "tgt_TYPE_coverage_perc_s2", "tgt_TYPE_coverage_perc_s1s2", "tgt_TYPE_coverage_perc_s3", "tgt_TYPE_coverage_perc_total", ]
            int_nom_denom = [ "tgt_TYPE_units_numerator", "tgt_TYPE_units_denom", "tgt_TYPE_units_denom_other", "tgt_TYPE_target_year", ]
            abs_targets = [ "tgt_abs_base_year_s1", "tgt_abs_units_s1", "tgt_abs_base_year_s2", "tgt_abs_units_s2", "tgt_abs_base_year_s1s2", "tgt_abs_units_s1s2",
                           "tgt_abs_base_year_s3", "tgt_abs_units_s3", "tgt_abs_base_year_total", "tgt_abs_units_total",
                           "tgt_abs_target_year", "tgt_abs_target_year_s1", "tgt_abs_target_year_s2", "tgt_abs_target_year_s1s2", "tgt_abs_target_year_s3", "tgt_abs_target_year_total", ]
            int_targets = [ "tgt_TYPE_base_year_s1", "tgt_TYPE_base_year_s2", "tgt_TYPE_base_year_s1s2",
                           "tgt_TYPE_base_year_s3", "tgt_TYPE_base_year_total",
                           "tgt_TYPE_target_year_s1", "tgt_TYPE_target_year_s2", "tgt_TYPE_target_year_s1s2", "tgt_TYPE_target_year_s3", "tgt_TYPE_target_year_total", ]
            target_pct = [ "tgt_TYPE_target_year_reduct_perc_s1", "tgt_TYPE_target_year_reduct_perc_s2", "tgt_TYPE_target_year_reduct_perc_s1s2", "tgt_TYPE_target_year_reduct_perc_s3", "tgt_TYPE_target_year_reduct_perc_total" ]
            target_comm = [ "tgt_TYPE_level_ambition", "tgt_TYPE_achieve_other_description", ]
            df_targets_abs = requantify_df(df_nzdpu_data["TARGETS (ABSOLUTE)"][
                # For now we don't bother with S3 categories
                target_abs_cols
                + [col.replace("_TYPE_", "_abs_") for col in target_cols]
                + abs_targets
                + [col.replace("_TYPE_", "_abs_") for col in target_pct]
                + [col.replace("_TYPE_", "_abs_") for col in target_comm]
            ])
            df_targets_phys = requantify_df(df_nzdpu_data["TARGETS (PHYS INTENSITY)"][
                # For now we don't bother with S3 categories
                target_int_cols
                + [col.replace("_TYPE_", "_phys_int_") for col in target_cols]
                + [col.replace("_TYPE_", "_phys_int_") for col in int_nom_denom]
                + [col.replace("_TYPE_", "_phys_int_")+"_int" for col in int_targets]
                + [col.replace("_TYPE_", "_phys_int_") for col in target_pct]
                + [col.replace("_TYPE_", "_int_") for col in target_comm]
            ])
            df_targets_econ = requantify_df(df_nzdpu_data["TARGETS (ECON INTENSITY)"][
                # For now we don't bother with S3 categories
                target_int_cols
                + [col.replace("_TYPE_", "_econ_int_") for col in target_cols]
                + [col.replace("_TYPE_", "_econ_int_") for col in int_nom_denom]
                + [col.replace("_TYPE_", "_econ_int_")+"_int" for col in int_targets]
                + [col.replace("_TYPE_", "_econ_int_") for col in target_pct]
                + [col.replace("_TYPE_", "_int_") for col in target_comm]
            ])
        except KeyError as e:
            logger.error(f"{e} raised when attempting to ingest target data in input Excel file.")
            raise

        breakpoint()

        self.df_esg = df_esg
        self.df_fundamentals = df_fundamentals
        # We don't want to process historic and target data yet
        return self._company_df_to_model(df_fundamentals, pd.DataFrame(), pd.DataFrame())

    def _convert_from_template_company_data(self) -> List[ICompanyData]:
        """
        Converts ESG sheet of Excel template to flesh out ICompanyData objects.
        :return: List of ICompanyData objects
        """
        df_fundamentals = self.df_fundamentals
        df_target_data = self.df_target_data
        if self.template_version > 1:
            # self.df_esg = self.df_esg[self.df_esg.company_id=='US3379321074']
            df_esg = self.df_esg

        def _fixup_name(x):
            prefix, _, suffix = x.partition("_")
            suffix = suffix.replace("ghg_", "")
            if suffix != "production":
                suffix = suffix.upper()
            return f"{suffix}-{prefix}"

        if self.template_version == 1:
            # The nightmare of naming columns 20xx_metric instead of metric_20xx...and potentially dealing with data from 1990s...
            historic_columns = [col for col in df_fundamentals.columns if col[:1].isdigit()]
            historic_scopes = ["S1", "S2", "S3", "S1S2", "S1S2S3", "production"]
            df_historic = df_fundamentals[["company_id"] + historic_columns].dropna(axis=1, how="all")
            df_fundamentals = df_fundamentals[df_fundamentals.columns.difference(historic_columns, sort=False)]

            df_historic = df_historic.rename(columns={col: _fixup_name(col) for col in historic_columns})
            df = pd.wide_to_long(
                df_historic,
                historic_scopes,
                i="company_id",
                j="year",
                sep="-",
                suffix=r"\d+",
            ).reset_index()
            df2 = (
                df.pivot(index="company_id", columns="year", values=historic_scopes)
                .stack(level=0, dropna=False)
                .reset_index()
                .rename(columns={"level_1": ColumnsConfig.SCOPE})
                .set_index("company_id")
                # Most things are emissions, but all things are strings
                .assign(variable=VariablesConfig.EMISSIONS)
            )
            # Fix the strings that are not emissions
            df2.loc[df2[ColumnsConfig.SCOPE] == "production", ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS

            df3 = df2.reset_index().set_index(["company_id", "variable", "scope"]).dropna(how="all")
            df3 = pd.concat(
                [
                    df3.xs(VariablesConfig.PRODUCTIONS, level=1, drop_level=False).apply(
                        lambda x: x.map(
                            lambda y: Q_(
                                float(y) if y is not pd.NA else np.nan,
                                df_fundamentals.loc[
                                    df_fundamentals.company_id == x.name[0],
                                    "production_metric",
                                ].squeeze(),
                            )
                        ),
                        axis=1,
                    ),
                    df3.xs(VariablesConfig.EMISSIONS, level=1, drop_level=False).apply(
                        lambda x: x.map(
                            lambda y: Q_(
                                float(y) if y is not pd.NA else np.nan,
                                df_fundamentals.loc[
                                    df_fundamentals.company_id == x.name[0],
                                    "emissions_metric",
                                ].squeeze(),
                            )
                        ),
                        axis=1,
                    ),
                ]
            )
            df4 = df3.xs(VariablesConfig.EMISSIONS, level=1) / df3.xs(
                (VariablesConfig.PRODUCTIONS, "production"), level=[1, 2]
            )
            df4["variable"] = VariablesConfig.EMISSIONS_INTENSITIES
            df4 = df4.reset_index().set_index(["company_id", "variable", "scope"])
            df5 = pd.concat([df3, df4])
            df_historic_data = df5
        else:
            # We are already much tidier, so don't need the wide_to_long conversion.
            esg_year_columns = df_esg.columns[df_esg.columns.get_loc(self.template_v2_start_year) :]
            df_esg_hasunits = df_esg.unit.notna()
            df_esg_badunits = df_esg[df_esg_hasunits].unit.map(lambda x: x not in ureg)
            badunits_idx = df_esg_badunits[df_esg_badunits].index
            if df_esg_badunits.any():
                logger.error(
                    f"The following row of data contain units that are not in the registry and will be removed from analysis\n{df_esg.loc[badunits_idx]}"
                )
                df_esg_hasunits.loc[badunits_idx] = False
            df_esg = df_esg[df_esg_hasunits]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_col = df_esg["unit"]
                for col in esg_year_columns:
                    # Convert ints to float as Work-around for Pandas GH#55824
                    # If we remove this extra conversion, we'll need to change Q_(m, u) to Q_(float(m), u)
                    # so as to convert not-yet-numeric string values to floating point numbers
                    df_esg[col] = df_esg[col].astype("float64")
                    df_esg[col] = df_esg[col].combine(
                        u_col,
                        lambda m, u: PintType(u).na_value if ITR.isna(m) else Q_(m, u),
                    )
            # All emissions metrics across multiple sectors should all resolve to some form of [mass] CO2
            em_metrics = df_esg[df_esg.metric.str.upper().isin(["S1", "S2", "S3", "S1S2", "S1S3", "S1S2S3"])]
            em_metrics_grouped = em_metrics.groupby(by=["company_id", "metric"])
            em_unit_nunique = em_metrics_grouped["unit"].nunique()
            if any(em_unit_nunique > 1):
                em_unit_ambig = em_unit_nunique[em_unit_nunique > 1].reset_index("metric")
                for company_id in em_unit_ambig.index.unique():
                    logger.warning(
                        f"Company {company_id} uses multiple units describing scopes "
                        f"{[s for s in em_unit_ambig.loc[[company_id]]['metric']]}"
                    )
                logger.warning("The ITR Tool will choose one and covert all to that")

            em_units = em_metrics.groupby(by=["company_id"], group_keys=True).first()
            # We update the metrics we were told with the metrics we are given
            df_fundamentals.loc[em_units.index, ColumnsConfig.EMISSIONS_METRIC] = em_units.unit

            # We solve while we still have valid report_date data.  After we group reports together to find the "best"
            # by averaging across report dates, the report_date becomes meaningless
            # FIXME: Check use of PRODUCTION_METRIC in _solve_intensities for multi-sector companies
            df_esg = self._solve_intensities(df_fundamentals, df_esg)

            # Recalculate if any of the above dropped rows from df_esg
            em_metrics = df_esg[df_esg.metric.str.upper().isin(["S1", "S2", "S3", "S1S2", "S1S3", "S1S2S3"])]

            # Convert CH4 to the GWP of CO2e
            ch4_idx = df_esg.loc[em_metrics.index].unit.str.contains("CH4")
            ch4_gwp = Q_(gwp.data["AR5GWP100"]["CH4"], "CO2e/CH4")
            ch4_to_co2e = df_esg.loc[em_metrics.index].loc[ch4_idx].unit.map(lambda x: x.replace("CH4", "CO2e"))
            df_esg.loc[ch4_to_co2e.index, "unit"] = ch4_to_co2e
            df_esg.loc[ch4_to_co2e.index, esg_year_columns] = df_esg.loc[ch4_to_co2e.index, esg_year_columns].apply(
                lambda x: ITR.asPintSeries(x).mul(ch4_gwp), axis=1
            )

            # Validate that all our em_metrics are, in fact, some kind of emissions quantity
            em_invalid = df_esg.loc[em_metrics.index].unit.map(
                lambda x: not isinstance(x, str) or not ureg(x).is_compatible_with("t CO2")
            )
            em_invalid_idx = em_invalid[em_invalid].index
            if len(em_invalid_idx) > 0:
                logger.error(
                    f"The following rows of data do not have proper emissions data (can be converted to t CO2e) and will be dropped from the analysis\n{df_esg.loc[em_invalid_idx]}"
                )
                df_esg = df_esg.loc[df_esg.index.difference(em_invalid_idx)]
                em_metrics = em_metrics.loc[em_metrics.index.difference(em_invalid_idx)]

            # We don't need units here anymore--they've been translated/transported everywhere we need them
            df_esg = df_esg.drop(columns="unit")

            submetric_sector_map = {
                "cement": "Cement",
                "clinker": "Cement",
                "chemicals": "Chemicals",
                "electricity": "Electricity Utilities",
                "generation": "Electricity Utilities",
                "gas": "Gas Utilities",
                "distribution": "Gas Utilities",
                "coal": "Coal",
                "lng": "Gas",
                "ng": "Gas",
                "oil": "Oil",
                # Note that 'revenue' is not sector-specific: Chemicals, Building Construction, Consumer Products, etc.
            }
            sector_submetric_keys = list(submetric_sector_map.keys())

            grouped_prod = (
                df_esg[df_esg.metric.isin(["production"])]
                .assign(submetric=lambda x: x["submetric"].str.lower())
                # .assign(sector=df_esg['company_id'].map(lambda x: df_fundamentals.loc[x].sector))
                .assign(
                    sector=lambda x: x[["company_id", "submetric"]].apply(
                        lambda y: submetric_sector_map.get(y.submetric, df_fundamentals.loc[y.company_id].sector),
                        axis=1,
                    )
                )
                # first collect things together down to sub-metric category
                .groupby(
                    by=[
                        ColumnsConfig.SECTOR,
                        ColumnsConfig.COMPANY_ID,
                        "metric",
                        "submetric",
                    ],
                    dropna=False,
                )[esg_year_columns]
                # then estimate values for each submetric (many/most of which will be NaN)
                .agg(_estimated_value)
                .reset_index(level="submetric")
            )

            # Now prioritize the submetrics we want: the best non-NaN values for each company in each column
            # For categoricals, fisrt listed is least and sorts to first in ascending order
            grouped_prod.submetric = pd.Categorical(
                grouped_prod["submetric"],
                ordered=True,
                categories=sector_submetric_keys
                + [
                    "operated",
                    "own",
                    "revenue",
                    "equity",
                    "",
                    "total",
                    "gross",
                    "net",
                    "full",
                    "*unrecognized*",
                ],
            )
            best_prod = grouped_prod.groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, "metric"]).apply(
                prioritize_submetric
            )
            # Comb out submetric field that we'll need later when sorting out sector data
            best_prod.submetric = best_prod.submetric.map(lambda x: x[0] if hasattr(x, "ndim") else x)
            best_prod[ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS

            # convert "nice" word descriptions of S3 emissions to category numbers
            s3_idx = df_esg.metric.str.upper().eq("S3")
            s3_dict_matches = df_esg[s3_idx].submetric.astype("string").str.lower().isin(s3_category_dict)
            s3_dict_idx = s3_dict_matches[s3_dict_matches].index
            df_esg.loc[s3_dict_idx, "submetric"] = (
                df_esg.loc[s3_dict_idx].submetric.astype("string").str.lower().map(s3_category_dict)
            )
            # FIXME: can we make more efficient by just using ':' as index on left-hand side?
            df_esg.loc[s3_idx.index.difference(s3_dict_idx), "submetric"] = df_esg.loc[
                s3_idx.index.difference(s3_dict_idx)
            ].submetric.map(maybe_other_s3_mappings)

            # We group, in order to prioritize, emissions according to boundary-like and/or category submetrics.
            grouped_em = (
                df_esg.loc[em_metrics.index]
                .assign(metric=df_esg.loc[em_metrics.index].metric.str.upper())
                .assign(
                    submetric=df_esg.loc[em_metrics.index].submetric.map(
                        lambda x: "" if ITR.isna(x) else str(x).lower()
                    )
                )
                # special submetrics define our sector (such as electricity -> Electricity Utilities)
                .assign(
                    sector=lambda x: x[["company_id", "submetric"]].apply(
                        lambda y: submetric_sector_map.get(y.submetric, df_fundamentals.loc[y.company_id].sector),
                        axis=1,
                    )
                )
                # Then group down to submetric across the report_dates, keeping the report dates in case we want to prioritize by report_date
                .set_index("report_date", append=True)
                .groupby(
                    by=[
                        ColumnsConfig.SECTOR,
                        ColumnsConfig.COMPANY_ID,
                        "metric",
                        "submetric",
                    ],
                    dropna=False,
                )[esg_year_columns]
                # ...averaging or estimating values for each submetric (many/most of which will be NaN)
                .agg(_estimated_value)
                .reset_index(level="submetric")
            )

            # Now prioritize the submetrics we want: the best non-NaN values for each company in each column
            s3_idx = grouped_em.index.get_level_values("metric") == "S3"
            non_s3_submetrics = sector_submetric_keys + [
                "own",
                "",
                "all",
                "combined",
                "total",
                "net",
                "location",
                "location-based",
                "market",
                "market-based",
                "*unrecognized*",
            ]
            grouped_non_s3 = grouped_em.loc[~s3_idx].copy()
            grouped_non_s3.submetric = pd.Categorical(
                grouped_non_s3["submetric"], ordered=True, categories=non_s3_submetrics
            )

            best_em = grouped_non_s3.groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, "metric"]).apply(
                prioritize_submetric
            )
            # Comb out submetric field that we'll need later when sorting out sector data
            best_em.submetric = best_em.submetric.map(lambda x: x[0] if hasattr(x, "ndim") else x)
            em_all_nan = best_em.drop(columns="submetric").apply(lambda x: x.map(lambda y: ITR.isna(y)).all(), axis=1)
            missing_em = best_em[em_all_nan]
            if len(missing_em):
                logger.warning(f"Emissions data missing for {missing_em.index}")
                best_em = best_em[~em_all_nan].copy()
            best_em[ColumnsConfig.VARIABLE] = VariablesConfig.EMISSIONS

            # We still need to group and prioritize S3 emissions, according to the benchmark requirements
            s3_submetrics = (
                sector_submetric_keys
                + ["", "all", "combined", "total", "3", "11"]
                + [k for k in s3_category_dict.keys() if k not in ["3", "11"]]
                + ["*unrecognized*"]
            )
            grouped_s3 = grouped_em.loc[s3_idx].copy()
            grouped_s3.submetric = pd.Categorical(grouped_s3["submetric"], ordered=True, categories=s3_submetrics)
            best_s3 = grouped_s3.groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, "metric"]).apply(
                prioritize_submetric
            )
            # Comb out submetric field that we'll need later when sorting out sector data
            best_s3.submetric = best_s3.submetric.map(lambda x: x[0] if hasattr(x, "ndim") else x)
            # x.submetric is np.nan or
            s3_all_nan = best_s3.apply(lambda x: x.drop("submetric").map(lambda y: ITR.isna(y)).all(), axis=1)
            missing_s3 = best_s3[s3_all_nan]
            if len(missing_s3):
                logger.warning(f"Scope 3 Emissions data missing for {missing_s3.index.droplevel('metric')}")
                # We cannot fill in missing data here, because we don't yet know what benchmark(s) will in use
                best_s3 = best_s3[~s3_all_nan].copy()
            best_s3[ColumnsConfig.VARIABLE] = VariablesConfig.EMISSIONS

            # We use the 'submetric' column to decide which new sectors we need to create
            company_sector_count = best_prod.groupby("company_id")["variable"].transform("count")
            company_sector_idx = best_prod[company_sector_count > 1].droplevel("metric").index
            new_prod = None
            new_em_to_allocate = new_em_allocated = None
            best_esg_em = pd.concat([best_em, best_s3]).sort_index()

            # If best_esg_em has a NaN submetric (not '' but NaN) it means somebody entered something that was not understood.  Flag that and Ignore it.
            ignored_em_idx = best_esg_em.submetric.isna()
            if ignored_em_idx.any():
                logger.error(
                    f"Unsupported submetrics appearing in\n{best_esg_em[ignored_em_idx].index} will be ignored"
                )
                best_esg_em = best_esg_em[~ignored_em_idx]

            if company_sector_idx.empty:
                best_prod = best_prod.droplevel("sector").drop(columns="submetric")
                best_esg_em = best_esg_em.droplevel("sector").drop(columns="submetric")
            else:
                company_em_sector = pd.MultiIndex.from_tuples(
                    [idx for idx in company_sector_idx if idx in best_esg_em.index],
                    names=["sector", "company_id"],
                )

                # Add new multi-sector companies, giving us choice of company_id, sector, company_id+sector (aka new_company_id)
                new_company_ids = company_sector_idx.to_frame(index=False).assign(
                    new_company_id=lambda x: x.company_id + "+" + x.sector
                )

                # Get completely new rows of data to add to df_fundamentals using new_company_id across each newly assigned sector
                new_fundamentals = (
                    df_fundamentals.loc[new_company_ids.company_id]
                    .reset_index(drop=True)
                    .assign(
                        sector=new_company_ids.sector,
                        company_id=new_company_ids.new_company_id,
                    )
                    .set_index("company_id", drop=False)
                )
                # Get rid of old rows that will become unattached to EMISSIONS_METRIC and PRODUCTION_METRIC
                df_fundamentals = pd.concat(
                    [
                        df_fundamentals[~df_fundamentals.company_id.isin(new_company_ids.company_id)],
                        new_fundamentals,
                    ]
                )

                prod_to_drop = best_prod.index.get_level_values("company_id").isin(new_company_ids.company_id)
                new_prod = (
                    best_prod[prod_to_drop]
                    .reset_index()
                    .query("submetric.isin(@submetric_sector_map)")
                    .merge(new_company_ids, on=["company_id", "sector"])
                    .drop(columns=["sector", "company_id", "submetric"])
                    .rename(columns={"new_company_id": "company_id"})
                    .set_index(["company_id", "metric"])
                )
                best_prod = best_prod[~prod_to_drop].droplevel("sector").drop(columns=["submetric"])

                # We must now handle three cases of emissions disclosures on a per-company basis:
                # (1) All-sector emissions that must be allocated across sectors.  In this case we allocate the full amount to each
                #     sector and it's divided down later in `update_benchmarks` (work defined in work_dict)
                # (2) Emissions tied to a specific sector
                # (3) A combination of (1) and (2)
                # These are all the emissions that need to be sorted into case 1, 2, or 3
                em_new_cases = best_esg_em.index.get_level_values("company_id").isin(new_company_ids.company_id)

                # Case 1 emissions need to be prorated across sectors using benchmark alignment method
                case_1 = best_esg_em.submetric[em_new_cases & ~best_esg_em.submetric.isin(sector_submetric_keys)]
                # Case 2 emissions are good as is; no benchmark alignment needed
                case_2 = best_esg_em.submetric[em_new_cases & best_esg_em.submetric.isin(sector_submetric_keys)]
                # Case 3 ambiguous overlap of emissions (i.e., Scope 3 general to Utilities (it's really just gas) and Scope 3 gas specific to Gas Utilities
                case_3 = best_esg_em.submetric[
                    best_esg_em.droplevel("sector").index.isin(
                        case_2.droplevel("sector").index.intersection(case_1.droplevel("sector").index)
                    )
                ]
                if not case_3.empty:
                    # Shift out of general (case_1) and leave in specific (case_2)
                    case_1 = case_1.loc[~case_1.index.isin(case_3.index)]

                # Case 4: case_1 scopes containing case_2 scopes that need to be removed before remaining scopes can be allocated
                # Example: We have S1 allocated to electricity and gas, but S2 and S3 are general.  To allocate S1S2S3 we need to subtract out S1, allocate remaining to S2 and S3 across Electricity and Gas sectors
                # Eni's Plenitude and power is an example where S1S2S3 > S1+S2+S3 (due to lifecycle emissions concept).  FIXME: don't know how to deal with that!
                case_4_df = case_1.reset_index("metric").merge(
                    case_2.reset_index("metric"),
                    on=["sector", "company_id"],
                    suffixes=[None, "_2"],
                )
                case_4 = case_4_df[case_4_df.apply(lambda x: x.metric_2 in x.metric, axis=1)].set_index(
                    "metric", append=True
                )
                if not case_4.empty:
                    logger.error(
                        f"Dropping attempt to disentangle embedded submetrics found in sector/scope assignment dataframe:\n{best_esg_em.submetric[case_4.index]}"
                    )
                    case_1 = case_1.loc[~case_1.index.isin(case_4.index)]

                em_needs_allocation = best_esg_em.index.isin(case_1.index)
                new_em_to_allocate = (
                    best_esg_em.reset_index()[em_needs_allocation]
                    # This gives us combinatorial product of common scope emissions across all types of production
                    # We'll use benchmark data to compute what fraction each type of production should get
                    .drop(columns="sector")
                    .merge(new_company_ids, on="company_id")
                    .drop(columns=["sector", "company_id", "submetric"])
                    .rename(columns={"new_company_id": "company_id"})
                    .set_index(["company_id", "metric"])
                )
                # Stash this index for later use once we have the benchmark with which to align
                self._bm_allocation_index = new_em_to_allocate.index
                em_has_allocation = best_esg_em.index.isin(case_2.index)
                new_em_allocated = (
                    best_esg_em.reset_index()[em_has_allocation]
                    .merge(new_company_ids, on=["sector", "company_id"])
                    .drop(columns=["sector", "company_id", "submetric"])
                    .rename(columns={"new_company_id": "company_id"})
                    .set_index(["company_id", "metric"])
                )
                best_esg_em = (
                    best_esg_em[~(em_needs_allocation | em_has_allocation)]
                    .droplevel("sector")
                    .drop(columns="submetric")
                )

            prod_df = pd.concat([best_prod, new_prod]).droplevel("metric")
            base_year_loc = prod_df.columns.get_loc(self.projection_controls.BASE_YEAR)
            base_year_na = prod_df.iloc[:, base_year_loc].isna()
            if base_year_na.any():
                logger.warning(
                    "The following companies lack base year production info (will be ignored):\n"
                    f"{prod_df[base_year_na].index.to_list()}"
                )
                # We could backfill instead of dropping companies...
                # prod_df.iloc[:, base_year_loc:-1] = prod_df.iloc[:, base_year_loc:-1].bfill(axis=1)
                prod_df = prod_df[~base_year_na]
                if len(prod_df) == 0:
                    logger.error("No companies left to analyze...aborting")
                    assert False
            prod_base_year = prod_df.iloc[:, base_year_loc]
            prod_metrics = prod_base_year.map(lambda x: f"{x.u:~P}")
            # We update the metrics we were told with the metrics we are given
            df_fundamentals.loc[prod_metrics.index, ColumnsConfig.PRODUCTION_METRIC] = prod_metrics

            # After this point we can gripe if missing emissions and/or production metrics
            missing_esg_metrics_df = df_fundamentals[ColumnsConfig.COMPANY_ID][
                df_fundamentals[ColumnsConfig.EMISSIONS_METRIC].isnull()
                | df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].isnull()
            ]
            if len(missing_esg_metrics_df) > 0:
                logger.warning(
                    f"Missing ESG metrics for companies with ID (will be ignored): " f"{missing_esg_metrics_df.index}."
                )
                df_fundamentals = df_fundamentals[~df_fundamentals.index.isin(missing_esg_metrics_df.index)]
                df_esg = df_esg[~df_esg.company_id.isin(missing_esg_metrics_df.index)]

            # We don't yet know our benchmark, so we cannot yet Use benchmark data to align their respective weights
            df3 = (
                pd.concat(
                    [
                        best_prod,
                        new_prod,
                        best_esg_em,
                        new_em_to_allocate,
                        new_em_allocated,
                    ]
                )
                .reset_index(level="metric")
                .rename(columns={"metric": "scope"})
                .set_index([ColumnsConfig.VARIABLE, "scope"], append=True)
            ).sort_index()

            # XS is how we match labels in indexes.  Here 'variable' is level=1, (company_id=0, scope/production=2)
            # By knocking out 'production', we don't get production / production in the calculations, only emissions (all scopes in data) / production
            assert "sector" not in df3.columns and "submetric" not in df3.columns
            if len(df3.index):
                assert "sector" not in df3.index.names and "submetric" not in df3.index.names

            # Avoid division by zero problems with zero-valued production metrics
            # Note that we should be filtering out NA production values before this point,
            # but we want this to be robust in case NA production values arrive here somehow
            df3_num_t = ITR.asPintDataFrame(df3.xs(VariablesConfig.EMISSIONS, level=1).T)
            df3_denom_t = ITR.asPintDataFrame(df3.xs((VariablesConfig.PRODUCTIONS, "production"), level=[1, 2]).T)
            df3_null = df3_denom_t.dtypes == object
            df3_null_idx = df3_null[df3_null].index
            if len(df3_null_idx):
                logger.warning(f"Dropping NULL-valued production data for these indexes\n{df3_null_idx}")
                df3_num_t = df3_num_t[~df3_null_idx]
                df3_denom_t = df3_denom_t[~df3_null_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
                df4 = (
                    df3_num_t
                    * df3_denom_t.rdiv(1.0).apply(
                        lambda x: x.map(
                            lambda y: x.dtype.na_value
                            if ITR.isna(y)
                            else Q_(0, x.dtype.units)
                            if np.isinf(ITR.nominal_values(y.m))
                            else y
                        )
                    )
                ).T
            df4["variable"] = VariablesConfig.EMISSIONS_INTENSITIES
            df4 = df4.reset_index().set_index(["company_id", "variable", "scope"])
            # Build df5 from PintArrays, not object types
            df3_num_t = pd.concat({VariablesConfig.EMISSIONS: df3_num_t}, names=["variable"], axis=1)
            df3_num_t.columns = df3_num_t.columns.reorder_levels(["company_id", "variable", "scope"])
            df3_denom_t = pd.concat({VariablesConfig.PRODUCTIONS: df3_denom_t}, names=["variable"], axis=1)
            df3_denom_t = pd.concat({"production": df3_denom_t}, names=["scope"], axis=1)
            df3_denom_t.columns = df3_denom_t.columns.reorder_levels(["company_id", "variable", "scope"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
                df5 = pd.concat([df3_num_t.T, df3_denom_t.T, df4])

            df_historic_data = df5

        # df_target_data now ready for conversion to model for each company
        df_target_data = self._validate_target_data(df_target_data)

        # df_historic now ready for conversion to model for each company
        self.historic_years = [column for column in df_historic_data.columns if isinstance(column, int)]

        def get_scoped_df(df, scope, names):
            mask = df.scope.eq(scope)
            return df.loc[mask[mask].index].set_index(names)

        def fill_blank_or_missing_scopes(df, scope_a, scope_b, scope_ab, index_names, historic_years):
            # Translate from long format, where each scope is on its own line, to common index
            df_a = get_scoped_df(df, scope_a, index_names)
            df_b = get_scoped_df(df, scope_b, index_names)
            df_ab = get_scoped_df(df, scope_ab, index_names).set_index("scope", append=True)
            # This adds rows of SCOPE_AB data that could be created by adding SCOPE_A and SCOPE_B rows
            new_ab_idx = df_a.index.intersection(df_b.index)
            new_ab = df_a.loc[new_ab_idx, historic_years] + df_b.loc[new_ab_idx, historic_years]
            new_ab.insert(0, "scope", scope_ab)
            new_ab.set_index("scope", append=True, inplace=True)
            df_ab[df_ab.map(ITR.isna)] = new_ab.loc[new_ab.index.intersection(df_ab.index)]
            # DF_AB has gaps filled, but not whole new rows that did not exist before
            # Drop rows in NEW_AB already covered by DF_AB and consolidate
            new_ab.drop(index=df_ab.index, inplace=True, errors="ignore")
            # Now update original dataframe with our new data
            df.set_index(index_names + ["scope"], inplace=True)
            df.update(df_ab)
            df = pd.concat([df, new_ab]).reset_index()
            return df

        df = df_historic_data.reset_index()
        index_names = ["company_id", "variable"]
        df = fill_blank_or_missing_scopes(df, "S1", "S2", "S1S2", index_names, self.historic_years)
        df = fill_blank_or_missing_scopes(df, "S1S2", "S3", "S1S2S3", index_names, self.historic_years)
        df_historic_data = df.set_index(["company_id", "variable", "scope"]).sort_index()
        # We might run `fill_blank_or_missing_scopes` again if we get newly estimated S3 data from an as-yet unknown benchmark

        # Drop from our companies list the companies dropped in df_fundamentals
        self._companies = [c for c in self._companies if c.company_id in df_fundamentals.index]
        # Add to our companies list the companies added in df_fundamentals
        self._companies.extend(
            self._company_df_to_model(
                df_fundamentals[df_fundamentals.company_id.str.contains("+", regex=False)],
                pd.DataFrame(),
                pd.DataFrame(),
            )
        )
        for company in self._companies:
            row = df_fundamentals.loc[company.company_id]
            company.emissions_metric = EmissionsMetric(row[ColumnsConfig.EMISSIONS_METRIC])
            company.production_metric = ProductionMetric(row[ColumnsConfig.PRODUCTION_METRIC])
        # And keep df_fundamentals in sync
        self.df_fundamentals = df_fundamentals

        # company_id, netzero_year, target_type, target_scope, target_start_year, target_base_year, target_base_year_qty, target_base_year_unit, target_year, target_reduction_ambition
        return self._company_df_to_model(None, df_target_data, df_historic_data)
