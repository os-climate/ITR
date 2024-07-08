import datetime
import logging
import re
import warnings  # needed until apply behaves better with Pint quantities in arrays
from typing import Dict, List, Optional, Type

import globalwarmingpotentials as gwp
import numpy as np
import pandas as pd
import pint
from pydantic import ValidationError

import ITR

from ..configs import (
    ColumnsConfig,
    LoggingConfig,
    ProjectionControls,
    SectorsConfig,
    TabsConfig,
    VariablesConfig,
)
from ..data import PA_, Q_, PintType, ureg
from ..data.base_providers import BaseCompanyDataProvider
from ..data.osc_units import (
    EmissionsMetric,
    ProductionMetric,
    asPintDataFrame,
    asPintSeries,
    fx_ctx,
)
from ..interfaces import (
    EScope,
    ICompanyData,
    IEIRealization,
    IEmissionRealization,
    IHistoricData,
    IHistoricEIScopes,
    IHistoricEmissionsScopes,
    IProductionRealization,
    ITargetData,
)
from ..utils import get_project_root

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)

pkg_root = get_project_root()
df_country_regions = pd.read_csv(f"{pkg_root}/data/input/country_region_info.csv")


def ITR_country_to_region(country: str) -> str:
    if not isinstance(country, str):
        if np.isnan(country):
            return "Global"
        raise ValueError(f"ITR_country_to_region received invalid valud {country}")
    if len(country) == 2:
        if country == "UK":
            country = "GB"
        elif country == "SP":
            country = "ES"
        elif country == "LX":
            country = "LU"
        elif country == "JN":
            country = "JP"
        elif country == "PO":
            country = "PT"
        regions = df_country_regions[
            df_country_regions.alpha_2 == country
        ].region_ar6_10
    elif len(country) == 3:
        regions = df_country_regions[
            df_country_regions.alpha_3 == country
        ].region_ar6_10
    else:
        if country in df_country_regions.name.values:
            regions = df_country_regions[
                df_country_regions.name == country
            ].region_ar6_10
        elif country in df_country_regions.common_name.values:
            regions = df_country_regions[
                df_country_regions.common_name == country
            ].region_ar6_10
        elif country == "Great Britain":
            return "Europe"
        else:
            raise ValueError(f"country {country} not found")
    if len(regions) == 0:
        raise ValueError(f"country {country} not found")
    region = regions.squeeze()
    if region in ["North America", "Europe"]:
        return region
    if "Asia" in region:
        return "Asia"
    return "Global"


# FIXME: circa line 480 of pint_array is code for master_scalar that shows how to decide if a thing has units


def _estimated_value(y: pd.Series) -> pint.Quantity:
    """:param y : a pd.Series that arrives via a pd.GroupBy operation.  The elements of the series are all data (or np.nan) matching a metric/sub-metric.

    :return: A Quantity which could be either the first or only element from the pd.Series, or an estimate of correct answer.
        The estimate is based on the mean of the non-null entries.
        It could be changed to output the last (most recent) value (if the inputs arrive sorted)
    """
    try:
        if isinstance(y, pd.DataFrame):
            # Something went wrong with the GroupBy operation and we got a pd.DataFrame
            # insted of being called column-by-column.
            logger.error(
                "Cannot estimate value of whole DataFrame; Something went wrong with GroupBy operation"
            )
            raise ValueError
        y = y[y.map(lambda x: not ITR.isna(x))]
    except TypeError:
        # FIXME: can get here if we have garbage units in one row (such as 't/MWh') that don't match another ('t CO2e/MWh')
        # Need to deal with that...
        logger.error(f"type_error({y}), so returning {y.values}[0]")
        return y.iloc[0]
    if len(y) == 0:
        return np.nan
    if len(y) == 1:
        # If there's only one non-NaN input, return that one
        return y.iloc[0]
    if not y.map(lambda x: isinstance(x, pint.Quantity)).all():
        # In the non-Quantity case, we might be "averaging" string values
        z = y.unique()
        if len(z) == 1:
            return z[0]
        logger.error(f"Invalid attempt to estimate average of these items: {z}")
        raise ValueError
    # Let PintArray do all the work of harmonizing compatible units
    z = PA_._from_sequence(y)
    if ITR.HAS_UNCERTAINTIES:
        wavg = ITR.umean(z.quantity.m)
    else:
        wavg = np.mean(z.quantity.m)
    est_value = Q_(wavg, z.quantity.u)
    return est_value


# FIXME: Should make this work with a pure PintArray


# Callers have already reduced multiple SUBMETRICS to a single rows using _estimated_value.
# However, pd.Categorize normalization might set several submetrics to NaN.  Ideally, we
# have more highly prioritized metrics than that, but if not, we can still fix things here.
def prioritize_submetric(x: pd.DataFrame) -> pd.Series:
    """:param x : pd.DataFrame.  The index of the DataFrame is (SECTOR, COMPANY_ID, SCOPE)

    :return: y : Based on the SECTOR, pick the highest priority Series based on the SUBMETRIC
    """
    if len(x) == 1:
        # Nothing to prioritize
        return x.iloc[0]

    # NaN values in pd.Categorical means we did not understand the prioritization of the submetric; *unrecognized* pushes to bottom
    x.submetric = x.submetric.fillna("*unrecognized*")
    x = x.sort_values("submetric")
    sector = x.index.get_level_values(0)[0]
    # Note that we only pull out a sector row if submetric is not NaN
    if sector == "Electricity Utilities":
        submetric_idx = x.submetric == "3"
        y = x[submetric_idx]
        x = x[~submetric_idx]
    elif sector in [
        "Autos",
        "Energy",
        "Oil & Gas",
        "Coal",
        "Oil",
        "Gas",
        "Gas Utilities",
    ]:
        submetric_idx = x.submetric == "11"
        y = x[submetric_idx]
        x = x[~submetric_idx]
    else:
        y = pd.DataFrame([], columns=x.columns)

    while not x.empty:
        submetric_idx = x.submetric == x.iloc[0].submetric
        z = x[submetric_idx]
        x = x[~submetric_idx]
        if len(z) > 1:
            z = z.groupby(
                by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, "metric"],
                dropna=False,
            ).agg(_estimated_value)
        if y.empty:
            y = z
        else:
            # Where TRUE, keep original value
            # FIXME: Pandas 2.1 allows inplace=True
            y = y.where(y.isna() <= z.isna(), z)
    assert not any(y.submetric == "*unrecognized*")
    return y.squeeze()


# Documentation for `number` can be found at f"https://ghgprotocol.org/sites/default/files/standards_supporting/Chapter{number}.pdf"
# Appendix A covers Sampling
# Appendix B covers Scenario Uncertainty
# Appendix C covers Intensity Metrics
# Appendix D contains Summary Tables
# Appendix documentation for `letter` can be found at f"https://ghgprotocol.org/sites/default/files/standards_supporting/Appendix{letter}.pdf"
s3_category_rdict = {
    "1": "Purchased goods and services",
    "2": "Capital goods",
    "3": "Fuel- and energy-related activities",
    "4": "Upstream transportation and distribution",
    "5": "Waste generated in operations",
    "6": "Business travel",
    "7": "Employee commuting",
    "8": "Upstream leased assets",
    "9": "Downstream transportation and distribution",
    "10": "Processing of sold products",
    "11": "Use of sold products",
    "12": "End-of-life treatment of sold products",
    "13": "Downstream leased assets",
    "14": "Franchises",
    "15": "Investments",
    "4,9": "Transportation",
}
s3_category_rdict_alt1 = {
    "3": "Fuel and energy-related activities",
    "4": "Upstream transportation",
    "8": "Leased assets (upstream)",
    "9": "Downstream transportation",
    "12": "End of life treatment",
    "13": "Leased assets (downstream)",
}
s3_category_dict = {v.lower(): k for k, v in s3_category_rdict.items()}
for k, v in s3_category_rdict_alt1.items():
    s3_category_dict[v.lower()] = k


def maybe_other_s3_mappings(x: str):
    if ITR.isna(x):
        return ""
    if isinstance(x, int):
        return str(x)
    if m := re.match(r"^Cat (\d+):", x, flags=re.IGNORECASE):
        return m.group(1)
    return x


# FIXME: Should we change this to derive from ExcelProviderCompany?
class TemplateProviderCompany(BaseCompanyDataProvider):
    """Data provider skeleton for CSV files. This class serves primarily for testing purposes only!
    As of Feb 2022, we are testing!!

    :param excel_path: A path to the Excel file with the company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    """

    def __init__(
        self,
        excel_path: str,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
    ):
        self.template_v2_start_year = None
        self.projection_controls = projection_controls
        # The initial population of companies' data
        if excel_path:
            self._own_data = True
            self._companies = self._init_from_template_company_data(excel_path)
            super().__init__(self._companies, column_config, projection_controls)
            # The perfection of historic ESG data (adding synthethic company sectors, dropping those with missing data)
            self._companies = self._convert_from_template_company_data()
        else:
            self._own_data = False
            self._companies = []

    # When rows of data are expressed in terms of scope intensities, solve for the implied production
    # This function is called before we've decided on "best" production, and indeed generates candidates for "best" emissions
    def _solve_intensities(
        self, df_fundamentals: pd.DataFrame, df_esg: pd.DataFrame
    ) -> pd.DataFrame:
        # We have organized all emissions and productions data given, and inferred intensities from that.
        # But we may yet have been giving emissions and emissions intensities without production numbers,
        # or emissions intensities and production numbers without absolute emissions numbers.
        # We calculate those and append to the result.
        df_esg_has_intensity = df_esg[
            df_esg.metric.str.contains("intensity", regex=False)
        ]
        df_esg_has_intensity = df_esg_has_intensity.assign(
            scope=lambda x: list(map(lambda y: y[0], x["metric"].str.split(" ")))
        )
        # https://stackoverflow.com/a/61021228/1291237
        compare_cols = ["company_id", "report_date"]
        has_reported_production_mask = pd.Series(
            list(zip(*[df_esg[c] for c in compare_cols]))
        ).isin(
            list(
                zip(*[df_esg[df_esg.metric.eq("production")][c] for c in compare_cols])
            )
        )
        df_esg_missing_production = df_esg[~has_reported_production_mask.values]
        start_year_loc = df_esg.columns.get_loc(self.template_v2_start_year)
        esg_year_columns = df_esg.columns[start_year_loc:]

        if len(df_esg_missing_production):
            # Generate production metrics from Emissions / EI, index by COMPANY_ID, REPORT_DATE, and SUBMETRIC
            df_intensities = df_esg_has_intensity.rename(
                columns={"metric": "intensity_metric", "scope": "metric"}
            )
            df_intensities = df_intensities.reset_index(drop=False).set_index(
                [
                    ColumnsConfig.COMPANY_ID,
                    "metric",
                    ColumnsConfig.TEMPLATE_REPORT_DATE,
                    "submetric",
                ]
            )
            df_emissions = df_esg_missing_production.set_index(
                [
                    ColumnsConfig.COMPANY_ID,
                    "metric",
                    ColumnsConfig.TEMPLATE_REPORT_DATE,
                    "submetric",
                ]
            )
            common_idx = df_emissions.index.intersection(df_intensities.index)
            df_emissions = df_emissions.loc[common_idx]
            df_intensities = df_intensities.loc[common_idx]
            df_emissions.loc[:, "index"] = df_intensities.loc[:, "index"]
            df_intensities = df_intensities.set_index("index", append=True).loc[
                :, esg_year_columns
            ]
            df_emissions = df_emissions.set_index("index", append=True).loc[
                :, esg_year_columns
            ]
            df_intensities_t = asPintDataFrame(df_intensities.T)
            df_emissions_t = asPintDataFrame(df_emissions.T)
            df_productions_t = df_emissions_t.divide(df_intensities_t)
            # FIXME: need to reconcile what happens if multiple scopes all yield the same production metrics
            df_productions_t = df_productions_t.apply(
                lambda x: x.astype(
                    f"pint[{ureg(str(x.dtype.units)).to_reduced_units().u}]"
                )
            ).dropna(axis=1, how="all")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # pint units don't like being twisted from columns to rows, but it's ok
                df_productions = df_productions_t.T.droplevel(
                    ["company_id", "metric", "report_date", "submetric"]
                )
            production_idx = df_productions.index
            df_esg.loc[production_idx] = pd.concat(
                [df_esg.loc[production_idx].iloc[:, 0:start_year_loc], df_productions],
                axis=1,
            )
            df_esg.loc[production_idx, "metric"] = "production"
        df_esg_has_intensity = df_esg[
            df_esg.metric.str.contains("intensity", regex=False)
        ]
        df_esg_has_intensity = df_esg_has_intensity.assign(
            scope=lambda x: list(map(lambda y: y[0], x["metric"].str.split(" ")))
        )
        if len(df_esg_has_intensity):
            # Gather all "unique" production metrics together.  Alas, production submetrics don't always align with Emissions/EI submetrics
            # FIXME: several boundary-related words are used in submetric prioritization, though they never appear as submetircs; for now, drop boundary
            df1 = df_esg[df_esg.metric.eq("production")].set_index(
                [
                    ColumnsConfig.COMPANY_ID,
                    ColumnsConfig.TEMPLATE_REPORT_DATE,
                    "submetric",
                ]
            )
            if "boundary" in df1.columns:
                df1.drop(columns="boundary", inplace=True)
            df1 = (
                df1.drop(columns="unit")
                .groupby(
                    by=[
                        ColumnsConfig.COMPANY_ID,
                        ColumnsConfig.TEMPLATE_REPORT_DATE,
                        "submetric",
                    ],
                    dropna=False,
                )
                .agg(_estimated_value)
            )
            df1["sub_count"] = df1.groupby(
                [ColumnsConfig.COMPANY_ID, ColumnsConfig.TEMPLATE_REPORT_DATE]
            )[df1.columns[0]].transform("count")
            cols = df1.columns.tolist()
            df1 = df1[cols[-1:] + cols[:-1]]
            # Convert EI to Emissions using production
            # Save index...we'll need it later to bring values back to df_esg
            # Hide all our values except those we need to use/join with later
            df2 = (
                df_esg_has_intensity.reset_index()
                .set_index(
                    [
                        ColumnsConfig.COMPANY_ID,
                        ColumnsConfig.TEMPLATE_REPORT_DATE,
                        "submetric",
                        ColumnsConfig.SCOPE,
                        "index",
                    ]
                )
                .loc[:, esg_year_columns]
            )
            # Now we have to reconcile the fact that production submetrics/boundaries may
            # or may not align with emissions(intensity) submetrics/boundaries
            # There are two cases we handle: (1) unique production rows match all intensity rows,
            # and (2) production rows with submetrics match intensity rows with same submetrics
            # There may be intensities for multiple scopes, which all multiply against the same
            # production number but which produce per-scope emissions values
            df1_case1 = (
                df1[df1.sub_count == 1]
                .droplevel("submetric")
                .drop(columns="sub_count")
                .merge(
                    df2.reset_index(["submetric", ColumnsConfig.SCOPE, "index"])[
                        ["submetric", ColumnsConfig.SCOPE, "index"]
                    ],
                    left_index=True,
                    right_index=True,
                )
                .set_index(["submetric", ColumnsConfig.SCOPE, "index"], append=True)
            )
            df1_case2 = (
                df1[df1.sub_count > 1]
                .drop(columns="sub_count")
                .merge(
                    df2.reset_index([ColumnsConfig.SCOPE, "index"])[
                        [ColumnsConfig.SCOPE, "index"]
                    ],
                    left_index=True,
                    right_index=True,
                )
                .set_index([ColumnsConfig.SCOPE, "index"], append=True)
            )
            case1_idx = df2.index.intersection(df1_case1.index)
            case2_idx = df2.index.intersection(df1_case2.index)
            # Mulitiplication of quantities results in concatenation of units, but NaNs might not be wrapped with their own units.
            # When we multiply naked NaNs by quantities, we get a defective concatenation.  Fix this by transposing df3 so we can use
            # PintArrays.  Those multiplications will create proper unit math for both numbers and NaNs
            df3_t = pd.concat(
                [
                    asPintDataFrame(df2.loc[case1_idx].T).mul(
                        asPintDataFrame(df1_case1.loc[case1_idx, esg_year_columns].T)
                    ),
                    asPintDataFrame(df2.loc[case2_idx].T).mul(
                        asPintDataFrame(df1_case2.loc[case2_idx, esg_year_columns].T)
                    ),
                ],
                axis=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if ITR.HAS_UNCERTAINTIES:
                    df4 = df3_t.astype(
                        "pint[t CO2e]"
                    ).T  # .drop_duplicates() # When we have uncertainties, multiple observations influence the observed error term
                    # Also https://github.com/pandas-dev/pandas/issues/12693
                else:
                    df4 = df3_t.astype("pint[t CO2e]").T.drop_duplicates()
            df5 = df4.droplevel(
                [ColumnsConfig.COMPANY_ID, ColumnsConfig.TEMPLATE_REPORT_DATE]
            ).swaplevel()  # .sort_index()
            df_esg.loc[df5.index.get_level_values("index"), "metric"] = (
                df5.index.get_level_values("scope")
            )
            df5.index = df5.index.droplevel(["scope", "submetric"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Change from emissions intensity metrics to emissions metrics; we know this will trigger a warning
                df_esg.loc[df5.index, esg_year_columns] = df5[esg_year_columns]
            # Set emissions metrics we just derived (if necessary), both in df_fundamentals (for targets, globally) and df_esg (for projections etc)
            for idx in df5.index:
                # FIXME: There's got to be a better way...
                unit = str(df_esg.loc[idx].ffill().iloc[-1].u)
                if ITR.isna(
                    df_fundamentals.loc[
                        df_esg.loc[idx].company_id, ColumnsConfig.EMISSIONS_METRIC
                    ]
                ):
                    df_fundamentals.loc[
                        df_esg.loc[idx].company_id, ColumnsConfig.EMISSIONS_METRIC
                    ] = unit
                df_esg.loc[idx, "unit"] = unit
        return df_esg

    def _init_from_template_company_data(self, excel_path: str):
        """Converts first sheet of Excel template to list of minimal ICompanyData objects (fundamental data, but no ESG data).
        All dataprovider features will be inhereted from Base.
        :param excel_path: file path to excel file
        """
        self.template_version = 1

        df_company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)

        if (
            TabsConfig.TEMPLATE_INPUT_DATA_V2
            and TabsConfig.TEMPLATE_INPUT_DATA_V2 in df_company_data
        ):
            self.template_version = 2
            input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA_V2
        else:
            input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA
        try:
            df = df_company_data[input_data_sheet]
        except KeyError as exc:
            logger.error(f"Tab {input_data_sheet} is required in input Excel file.")
            raise exc
        if self.template_version == 2:
            esg_data_sheet = TabsConfig.TEMPLATE_ESG_DATA_V2
            try:
                df_esg = (
                    df_company_data[esg_data_sheet].drop(columns="company_lei").copy()
                )  # .iloc[246:291]  # .iloc[1265:1381]  # .iloc[0:45]
            except KeyError as exc:
                logger.error(f"Tab {esg_data_sheet} is required in input Excel file.")
                raise exc
            # Change year column names to integers if they come in as strings
            df_esg.rename(
                columns=lambda x: int(x)
                if isinstance(x, str) and x >= "1000" and x <= "2999"
                else x,
                inplace=True,
            )
            if "base_year" in df_esg.columns:
                self.template_v2_start_year = df_esg.columns[
                    df_esg.columns.get_loc("base_year") + 1
                ]
            else:
                self.template_v2_start_year = df_esg.columns[
                    df_esg.columns.map(lambda col: isinstance(col, int))
                ][0]
            # Make sure that if all NaN these columns are not represented as float64
            df_esg.submetric = df_esg.submetric.astype("string").str.strip().fillna("")
            if "boundary" in df_esg.columns:
                df_esg["boundary"] = (
                    df_esg["boundary"].astype("string").str.strip().fillna("")
                )
            # In the V2 template, the COMPANY_NAME and COMPANY_ID are merged cells and need to be filled forward
            # For convenience, we also fill forward Report Date, which is often expressed in the first row of a fresh report,
            # but sometimes omitted in subsequent rows (because it's "redundant information")
            # COMPANY_LEI has been dropped for now...may make a comeback later...

            # https://stackoverflow.com/a/74193599/1291237
            with warnings.catch_warnings():
                # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
                # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
                # See also: https://stackoverflow.com/q/74057367/859591
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=(
                        ".*will attempt to set the values inplace instead of always setting a new array. "
                        "To retain the old behavior, use either.*"
                    ),
                )

                df_esg.loc[
                    :,
                    [
                        ColumnsConfig.COMPANY_NAME,
                        ColumnsConfig.COMPANY_ID,
                        ColumnsConfig.TEMPLATE_REPORT_DATE,
                    ],
                ] = df_esg[
                    [
                        ColumnsConfig.COMPANY_NAME,
                        ColumnsConfig.COMPANY_ID,
                        ColumnsConfig.TEMPLATE_REPORT_DATE,
                    ]
                ].ffill()

        # NA in exposure is how we drop rows we want to ignore
        df = df[df.exposure.notna()].copy()

        # TODO: Fix market_cap column naming inconsistency
        df.rename(
            columns={
                "revenue": "company_revenue",
                "market_cap": "company_market_cap",
                "ev": "company_enterprise_value",
                "evic": "company_ev_plus_cash",
                "assets": "company_total_assets",
            },
            inplace=True,
        )
        df.loc[df.region.isnull(), "region"] = df.country.map(ITR_country_to_region)

        df_fundamentals = df.set_index(
            ColumnsConfig.COMPANY_ID, drop=False
        ).convert_dtypes()

        if self.template_version == 2:
            # Ensure our df_esg rows connect back to fundamental data
            # the one single advantage of template_version==1 is that
            # fundamental data and esg data are all part of the same rows
            # so no need to do this integrity check/correction
            esg_missing_fundamentals = ~df_esg.company_id.isin(df_fundamentals.index)
            if esg_missing_fundamentals.any():
                logger.error(
                    f"The following companies have ESG data defined but no fundamental data and will be removed from further analysis:\n{df_esg[esg_missing_fundamentals].company_id.unique()}"  # noqa: E501
                )
                df_esg = df_esg[~esg_missing_fundamentals]

        # testing if all data is in the same currency
        fundamental_metrics = [
            "company_market_cap",
            "company_revenue",
            "company_enterprise_value",
            "company_ev_plus_cash",
            "company_total_assets",
        ]
        col_num = df_fundamentals.columns.get_loc("report_date")
        missing_fundamental_metrics = [
            fm
            for fm in fundamental_metrics
            if fm not in df_fundamentals.columns[col_num + 1 :]  # noqa: E203
        ]
        if len(missing_fundamental_metrics) > 0:
            raise KeyError(
                f"Expected fundamental metrics {missing_fundamental_metrics}"
            )
        if ColumnsConfig.TEMPLATE_FX_QUOTE in df_fundamentals.columns:
            fx_quote = df_fundamentals[ColumnsConfig.TEMPLATE_FX_QUOTE].notna()
            if (
                len(
                    df_fundamentals.loc[
                        ~fx_quote, ColumnsConfig.COMPANY_CURRENCY
                    ].unique()
                )
                > 1
                or len(
                    df_fundamentals.loc[
                        fx_quote, ColumnsConfig.TEMPLATE_FX_QUOTE
                    ].unique()
                )
                > 1
                or (
                    fx_quote.any()
                    and (~fx_quote).any()
                    and not (
                        df_fundamentals.loc[
                            ~fx_quote, ColumnsConfig.COMPANY_CURRENCY
                        ].iloc[0]
                        == df_fundamentals.loc[
                            fx_quote, ColumnsConfig.TEMPLATE_FX_QUOTE
                        ].iloc[0]
                    )
                )
            ):
                error_message = "All data should be in the same currency."
                logger.error(error_message)
                raise ValueError(error_message)
            elif fx_quote.any():
                # create context for currency conversions
                # df_fundamentals defines 'report_date', 'currency', 'fx_quote', and 'fx_rate'
                # our base currency default is USD, but european reports may be denominated in EUR.  We crosswalk from report_base to pint_base currency.

                def convert_prefix_to_scalar(x):
                    try:
                        prefix, unit, suffix = ureg.parse_unit_name(x)[0]
                    except IndexError:
                        raise f"Currency {x} is not in the registry; please update registry or remove data row"
                    return unit, ureg._prefixes[prefix].value if prefix else 1.0

                fx_df = (
                    df_fundamentals.loc[
                        fx_quote, ["report_date", "currency", "fx_quote", "fx_rate"]
                    ]
                    .drop_duplicates()
                    .set_index("report_date")
                )
                fx_conversion_mask = fx_df.currency != fx_df.fx_quote
                assert fx_df[~fx_conversion_mask].fx_rate.eq(1.0).all()
                fx_df = fx_df[fx_conversion_mask].assign(
                    currency_tuple=lambda x: x["currency"].map(
                        convert_prefix_to_scalar
                    ),
                    fx_quote_tuple=lambda x: x["fx_quote"].map(
                        convert_prefix_to_scalar
                    ),
                )

                # FIXME: These simple rules don't take into account different conversion rates at different time periods.
                fx_df.apply(
                    lambda x: (
                        fx_ctx.redefine(
                            f"{x.currency_tuple[0]} = {x.fx_rate * x.fx_quote_tuple[1] / x.currency_tuple[1]} {x.fx_quote_tuple[0]}"
                        )
                        if x.currency_tuple[0] != "USD"
                        else fx_ctx.redefine(
                            f"{x.fx_quote_tuple[0]} = {x.currency_tuple[1]/(x.fx_rate * x.fx_quote_tuple[1])} {x.currency_tuple[0]}"
                        )
                    ),
                    axis=1,
                )
                ureg.add_context(fx_ctx)
                ureg.enable_contexts("FX")

                for col in fundamental_metrics:
                    # PintPandas 0.3 (without OS-Climate enhancements) cannot deal with Float64DTypes that contain pd.NA
                    df_fundamentals[col] = df_fundamentals[col].astype("float64")
                    df_fundamentals[f"{col}_base"] = df_fundamentals[col]
                    df_fundamentals.loc[fx_quote, col] = (
                        df_fundamentals.loc[fx_quote, "fx_rate"].astype("float64")
                        * df_fundamentals.loc[fx_quote, f"{col}_base"]
                    )
                    quote_currency, quote_scalar = convert_prefix_to_scalar(
                        df_fundamentals.loc[
                            fx_quote, ColumnsConfig.TEMPLATE_FX_QUOTE
                        ].iloc[0]
                    )
                    df_fundamentals[col] = (
                        df_fundamentals[col]
                        .mul(quote_scalar)
                        .astype(f"pint[{quote_currency}]")
                    )
            else:
                # Degenerate case where we have fx_quote column and no actual fx_quote conversions to do
                for col in fundamental_metrics:
                    # PintPandas 0.3 (without OS-Climate enhancements) cannot deal with Float64DTypes that contain pd.NA
                    df_fundamentals[col] = (
                        df_fundamentals[col]
                        .astype("float64")
                        .astype(
                            f"pint[{df_fundamentals[ColumnsConfig.COMPANY_CURRENCY].iloc[0]}]"
                        )
                    )
        else:
            if len(df_fundamentals[ColumnsConfig.COMPANY_CURRENCY].unique()) != 1:
                error_message = "All data should be in the same currency."
                logger.error(error_message)
                raise ValueError(error_message)
            for col in fundamental_metrics:
                # PintPandas 0.3 (without OS-Climate enhancements) cannot deal with Float64DTypes that contain pd.NA
                df_fundamentals[col] = (
                    df_fundamentals[col]
                    .astype("float64")
                    .astype(
                        f"pint[{df_fundamentals[ColumnsConfig.COMPANY_CURRENCY].iloc[0]}]"
                    )
                )

        # are there empty sectors?
        comp_with_missing_sectors = df_fundamentals[ColumnsConfig.COMPANY_ID][
            df_fundamentals[ColumnsConfig.SECTOR].isnull()
        ].to_list()
        if comp_with_missing_sectors:
            error_message = (
                f"For {comp_with_missing_sectors} companies the sector column is empty."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # testing if only valid sectors are provided
        sectors_from_df = df_fundamentals[ColumnsConfig.SECTOR].unique()
        configured_sectors = SectorsConfig.get_configured_sectors()
        not_configured_sectors = [
            sec for sec in sectors_from_df if sec not in configured_sectors
        ]
        if not_configured_sectors:
            error_message = f"Sector {not_configured_sectors} is not covered by the ITR tool currently."
            logger.error(error_message)
            raise ValueError(error_message)

        # Checking missing company ids
        missing_company_ids = df_fundamentals[ColumnsConfig.COMPANY_ID].isnull().any()
        if missing_company_ids:
            error_message = "Missing company ids"
            logger.error(error_message)
            raise ValueError(error_message)

        # Checking if there are not many missing market cap
        missing_cap_ids = df_fundamentals[ColumnsConfig.COMPANY_ID][
            df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP].isnull()
        ].to_list()
        # For the missing Market Cap we should use the ratio below to get dummy market cap:
        #   (Avg for the Sector (Market Cap / Revenues) + Avg for the Sector (Market Cap / Assets)) 2
        df_fundamentals["MCap_to_Reven"] = (
            df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP]
            / df_fundamentals[ColumnsConfig.COMPANY_REVENUE]
        )  # new temp column with ratio
        df_fundamentals["MCap_to_Assets"] = (
            df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP]
            / df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS]
        )  # new temp column with ratio
        df_fundamentals["AVG_MCap_to_Reven"] = df_fundamentals.groupby(
            ColumnsConfig.SECTOR
        )["MCap_to_Reven"].transform("mean")
        df_fundamentals["AVG_MCap_to_Assets"] = df_fundamentals.groupby(
            ColumnsConfig.SECTOR
        )["MCap_to_Assets"].transform("mean")
        # FIXME: Add uncertainty here!
        df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP] = df_fundamentals[
            ColumnsConfig.COMPANY_MARKET_CAP
        ].fillna(
            0.5
            * (
                df_fundamentals[ColumnsConfig.COMPANY_REVENUE]
                * df_fundamentals["AVG_MCap_to_Reven"]
                + df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS]
                * df_fundamentals["AVG_MCap_to_Assets"]
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
                f"Missing market capitalisation values are estimated for companies with ID: "
                f"{missing_cap_ids}."
            )

        test_target_sheet = TabsConfig.TEMPLATE_TARGET_DATA
        try:
            self.df_target_data = (
                df_company_data[test_target_sheet]
                .set_index("company_id")
                .convert_dtypes()
            )
        except KeyError:
            logger.error(f"Tab {test_target_sheet} is required in input Excel file.")
            raise

        if self.template_version == 1:
            # ignore company data that does not come with emissions and/or production metrics
            missing_esg_metrics_df = df_fundamentals[ColumnsConfig.COMPANY_ID][
                df_fundamentals[ColumnsConfig.EMISSIONS_METRIC].isnull()
                | df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].isnull()
            ]
            if len(missing_esg_metrics_df) > 0:
                logger.warning(
                    f"Missing ESG metrics for companies with ID (will be ignored): "
                    f"{missing_esg_metrics_df.index}."
                )
                df_fundamentals = df_fundamentals[
                    ~df_fundamentals.index.isin(missing_esg_metrics_df.index)
                ]
            # Template V1 does not use df_esg
            # self.df_esg = None
        else:
            # df_esg = df_esg.iloc[0:45]

            # Disable rows we do not yet handle
            df_esg = df_esg[~df_esg.metric.isin(["generation", "consumption"])]
            if ColumnsConfig.BASE_YEAR in df_esg.columns:
                df_esg = df_esg[
                    df_esg[ColumnsConfig.BASE_YEAR].map(
                        lambda x: not isinstance(x, str) or x.lower() != "x"
                    )
                ]
            if "submetric" in df_esg.columns:
                df_esg = df_esg[
                    df_esg.submetric.map(
                        lambda x: not isinstance(x, str) or x.lower() != "ignore"
                    )
                ]
            # FIXME: Should we move more df_esg work up here?
            self.df_esg = df_esg
        self.df_fundamentals = df_fundamentals
        # We don't want to process historic and target data yet
        return self._company_df_to_model(
            df_fundamentals, pd.DataFrame(), pd.DataFrame()
        )

    def _convert_from_template_company_data(self) -> List[ICompanyData]:
        """Converts ESG sheet of Excel template to flesh out ICompanyData objects.
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
            historic_columns = [
                col for col in df_fundamentals.columns if col[:1].isdigit()
            ]
            historic_scopes = ["S1", "S2", "S3", "S1S2", "S1S2S3", "production"]
            df_historic = df_fundamentals[["company_id"] + historic_columns].dropna(
                axis=1, how="all"
            )
            df_fundamentals = df_fundamentals[
                df_fundamentals.columns.difference(historic_columns, sort=False)
            ]

            df_historic = df_historic.rename(
                columns={col: _fixup_name(col) for col in historic_columns}
            )
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
            df2.loc[
                df2[ColumnsConfig.SCOPE] == "production", ColumnsConfig.VARIABLE
            ] = VariablesConfig.PRODUCTIONS

            df3 = (
                df2.reset_index()
                .set_index(["company_id", "variable", "scope"])
                .dropna(how="all")
            )
            df3 = pd.concat(
                [
                    df3.xs(
                        VariablesConfig.PRODUCTIONS, level=1, drop_level=False
                    ).apply(
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
            esg_year_columns = df_esg.columns[
                df_esg.columns.get_loc(self.template_v2_start_year) :
            ]  # noqa: E203
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
                # Quantify columns where units are specified (incl `base_year` if specified)
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
                if "base_year" in df_esg.columns:
                    df_esg.loc[pd.notna(u_col), "base_year"] = (
                        df_esg.loc[pd.notna(u_col), "base_year"]
                        .astype("float64")
                        .combine(
                            u_col,
                            lambda m, u: PintType(u).na_value
                            if ITR.isna(m)
                            else Q_(m, u),
                        )
                    )
            # All emissions metrics across multiple sectors should all resolve to some form of [mass] CO2
            em_metrics = df_esg[
                df_esg.metric.str.upper().isin(
                    ["S1", "S2", "S3", "S1S2", "S1S3", "S1S2S3"]
                )
            ]
            em_metrics_grouped = em_metrics.groupby(by=["company_id", "metric"])
            em_unit_nunique = em_metrics_grouped["unit"].nunique()
            if any(em_unit_nunique > 1):
                em_unit_ambig = em_unit_nunique[em_unit_nunique > 1].reset_index(
                    "metric"
                )
                for company_id in em_unit_ambig.index.unique():
                    logger.warning(
                        f"Company {company_id} uses multiple units describing scopes "
                        f"{[s for s in em_unit_ambig.loc[[company_id]]['metric']]}"
                    )
                logger.warning("The ITR Tool will choose one and covert all to that")

            em_units = em_metrics.groupby(by=["company_id"], group_keys=True).first()
            # We update the metrics we were told with the metrics we are given
            df_fundamentals.loc[em_units.index, ColumnsConfig.EMISSIONS_METRIC] = (
                em_units.unit
            )

            # We solve while we still have valid report_date data.  After we group reports together to find the "best"
            # by averaging across report dates, the report_date becomes meaningless
            # FIXME: Check use of PRODUCTION_METRIC in _solve_intensities for multi-sector companies
            df_esg = self._solve_intensities(df_fundamentals, df_esg)

            # Recalculate if any of the above dropped rows from df_esg
            em_metrics = df_esg[
                df_esg.metric.str.upper().isin(
                    ["S1", "S2", "S3", "S1S2", "S1S3", "S1S2S3"]
                )
            ]

            # Convert CH4 to the GWP of CO2e; because we do this only for em_metrics, intensity metrics don't confuse us
            df = df_esg.loc[em_metrics.index]
            ch4_idx = df.unit.str.contains("CH4")
            ch4_gwp = Q_(gwp.data["AR5GWP100"]["CH4"], "CO2e/CH4")
            ch4_to_co2e = df.loc[ch4_idx].unit.map(lambda x: x.replace("CH4", "CO2e"))
            df_esg.loc[ch4_to_co2e.index, "unit"] = ch4_to_co2e
            df_esg.loc[ch4_to_co2e.index, esg_year_columns] = df_esg.loc[
                ch4_to_co2e.index, esg_year_columns
            ].apply(lambda x: asPintSeries(x).mul(ch4_gwp), axis=1)

            # Validate that all our em_metrics are, in fact, some kind of emissions quantity
            em_invalid = df_esg.loc[em_metrics.index].unit.map(
                lambda x: not isinstance(x, str)
                or not ureg(x).is_compatible_with("t CO2")
            )
            em_invalid_idx = em_invalid[em_invalid].index
            if len(em_invalid_idx) > 0:
                logger.error(
                    f"The following rows of data do not have proper emissions data (can be converted to t CO2e) and will be dropped from the analysis\n{df_esg.loc[em_invalid_idx]}"  # noqa: E501
                )
                df_esg = df_esg.loc[df_esg.index.difference(em_invalid_idx)]
                em_metrics = em_metrics.loc[em_metrics.index.difference(em_invalid_idx)]

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
                        lambda y: submetric_sector_map.get(
                            y.submetric, df_fundamentals.loc[y.company_id].sector
                        ),
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
            best_prod = grouped_prod.groupby(
                by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, "metric"]
            ).apply(prioritize_submetric)
            # Comb out submetric field that we'll need later when sorting out sector data
            best_prod.submetric = best_prod.submetric.map(
                lambda x: x[0] if hasattr(x, "ndim") else x
            )
            best_prod[ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS

            # convert "nice" word descriptions of S3 emissions to category numbers
            s3_idx = df_esg.metric.str.upper().eq("S3")
            s3_dict_matches = (
                df_esg[s3_idx]
                .submetric.astype("string")
                .str.lower()
                .isin(s3_category_dict)
            )
            s3_dict_idx = s3_dict_matches[s3_dict_matches].index
            df_esg.loc[s3_dict_idx, "submetric"] = (
                df_esg.loc[s3_dict_idx]
                .submetric.astype("string")
                .str.lower()
                .map(s3_category_dict)
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
                        lambda y: submetric_sector_map.get(
                            y.submetric, df_fundamentals.loc[y.company_id].sector
                        ),
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

            best_em = grouped_non_s3.groupby(
                by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, "metric"]
            ).apply(prioritize_submetric)
            # Comb out submetric field that we'll need later when sorting out sector data
            best_em.submetric = best_em.submetric.map(
                lambda x: x[0] if hasattr(x, "ndim") else x
            )
            em_all_nan = best_em.drop(columns="submetric").apply(
                lambda x: x.map(lambda y: ITR.isna(y)).all(), axis=1
            )
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
            grouped_s3.submetric = pd.Categorical(
                grouped_s3["submetric"], ordered=True, categories=s3_submetrics
            )
            best_s3 = grouped_s3.groupby(
                by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, "metric"]
            ).apply(prioritize_submetric)
            # Comb out submetric field that we'll need later when sorting out sector data
            best_s3.submetric = best_s3.submetric.map(
                lambda x: x[0] if hasattr(x, "ndim") else x
            )
            # x.submetric is np.nan or
            s3_all_nan = best_s3.apply(
                lambda x: x.drop("submetric").map(lambda y: ITR.isna(y)).all(), axis=1
            )
            missing_s3 = best_s3[s3_all_nan]
            if len(missing_s3):
                logger.warning(
                    f"Scope 3 Emissions data missing for {missing_s3.index.droplevel('metric')}"
                )
                # We cannot fill in missing data here, because we don't yet know what benchmark(s) will in use
                best_s3 = best_s3[~s3_all_nan].copy()
            best_s3[ColumnsConfig.VARIABLE] = VariablesConfig.EMISSIONS

            # We use the 'submetric' column to decide which new sectors we need to create
            company_sector_count = best_prod.groupby("company_id")[
                "variable"
            ].transform("count")
            company_sector_idx = (
                best_prod[company_sector_count > 1].droplevel("metric").index
            )
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
                        df_fundamentals[
                            ~df_fundamentals.company_id.isin(new_company_ids.company_id)
                        ],
                        new_fundamentals,
                    ]
                )

                prod_to_drop = best_prod.index.get_level_values("company_id").isin(
                    new_company_ids.company_id
                )
                new_prod = (
                    best_prod[prod_to_drop]
                    .reset_index()
                    .query("submetric.isin(@submetric_sector_map)")
                    .merge(new_company_ids, on=["company_id", "sector"])
                    .drop(columns=["sector", "company_id", "submetric"])
                    .rename(columns={"new_company_id": "company_id"})
                    .set_index(["company_id", "metric"])
                )
                best_prod = (
                    best_prod[~prod_to_drop]
                    .droplevel("sector")
                    .drop(columns=["submetric"])
                )

                # We must now handle three cases of emissions disclosures on a per-company basis:
                # (1) All-sector emissions that must be allocated across sectors.  In this case we allocate the full amount to each
                #     sector and it's divided down later in `update_benchmarks` (work defined in work_dict)
                # (2) Emissions tied to a specific sector
                # (3) A combination of (1) and (2)
                # These are all the emissions that need to be sorted into case 1, 2, or 3
                em_new_cases = best_esg_em.index.get_level_values("company_id").isin(
                    new_company_ids.company_id
                )

                # Case 1 emissions need to be prorated across sectors using benchmark alignment method
                case_1 = best_esg_em.submetric[
                    em_new_cases & ~best_esg_em.submetric.isin(sector_submetric_keys)
                ]
                # Case 2 emissions are good as is; no benchmark alignment needed
                case_2 = best_esg_em.submetric[
                    em_new_cases & best_esg_em.submetric.isin(sector_submetric_keys)
                ]
                # Case 3 ambiguous overlap of emissions (i.e., Scope 3 general to Utilities (it's really just gas) and Scope 3 gas specific to Gas Utilities
                case_3 = best_esg_em.submetric[
                    best_esg_em.droplevel("sector").index.isin(
                        case_2.droplevel("sector").index.intersection(
                            case_1.droplevel("sector").index
                        )
                    )
                ]
                if not case_3.empty:
                    # Shift out of general (case_1) and leave in specific (case_2)
                    case_1 = case_1.loc[~case_1.index.isin(case_3.index)]

                # Case 4: case_1 scopes containing case_2 scopes that need to be removed before
                # remaining scopes can be allocated
                # Example: We have S1 allocated to electricity and gas, but S2 and S3 are general.
                # To allocate S1S2S3 we need to subtract out S1, allocate remaining to S2 and S3
                # across Electricity and Gas sectors
                # Eni's Plenitude and power is an example where S1S2S3 > S1+S2+S3 (due to lifecycle emissions concept).
                # FIXME: don't know how to deal with that!
                case_4_df = case_1.reset_index("metric").merge(
                    case_2.reset_index("metric"),
                    on=["sector", "company_id"],
                    suffixes=[None, "_2"],
                )
                case_4 = case_4_df[
                    case_4_df.apply(lambda x: x.metric_2 in x.metric, axis=1)
                ].set_index("metric", append=True)
                if not case_4.empty:
                    logger.error(
                        f"Dropping attempt to disentangle embedded submetrics found in sector/scope assignment dataframe:\n{best_esg_em.submetric[case_4.index]}"  # noqa: E501
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
            df_fundamentals.loc[prod_metrics.index, ColumnsConfig.PRODUCTION_METRIC] = (
                prod_metrics
            )

            # After this point we can gripe if missing emissions and/or production metrics
            missing_esg_metrics_df = df_fundamentals[ColumnsConfig.COMPANY_ID][
                df_fundamentals[ColumnsConfig.EMISSIONS_METRIC].isnull()
                | df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].isnull()
            ]
            if len(missing_esg_metrics_df) > 0:
                logger.warning(
                    f"Missing ESG metrics for companies with ID (will be ignored): "
                    f"{missing_esg_metrics_df.index}."
                )
                df_fundamentals = df_fundamentals[
                    ~df_fundamentals.index.isin(missing_esg_metrics_df.index)
                ]
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
                assert (
                    "sector" not in df3.index.names
                    and "submetric" not in df3.index.names
                )

            # Avoid division by zero problems with zero-valued production metrics
            # Note that we should be filtering out NA production values before this point,
            # but we want this to be robust in case NA production values arrive here somehow
            df3_num_t = asPintDataFrame(df3.xs(VariablesConfig.EMISSIONS, level=1).T)
            df3_denom_t = asPintDataFrame(
                df3.xs((VariablesConfig.PRODUCTIONS, "production"), level=[1, 2]).T
            )
            df3_null = df3_denom_t.dtypes == object
            df3_null_idx = df3_null[df3_null].index
            if len(df3_null_idx):
                logger.warning(
                    f"Dropping NULL-valued production data for these indexes\n{df3_null_idx}"
                )
                df3_num_t = df3_num_t[~df3_null_idx]
                df3_denom_t = df3_denom_t[~df3_null_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
                df4 = (
                    df3_num_t
                    * df3_denom_t.rdiv(1.0).apply(
                        lambda x: x.map(
                            lambda y: (
                                x.dtype.na_value
                                if ITR.isna(y)
                                else Q_(0, x.dtype.units)
                                if np.isinf(ITR.nominal_values(y.m))
                                else y
                            )
                        )
                    )
                ).T
            df4["variable"] = VariablesConfig.EMISSIONS_INTENSITIES
            df4 = df4.reset_index().set_index(["company_id", "variable", "scope"])
            # Build df5 from PintArrays, not object types
            df3_num_t = pd.concat(
                {VariablesConfig.EMISSIONS: df3_num_t}, names=["variable"], axis=1
            )
            df3_num_t.columns = df3_num_t.columns.reorder_levels(
                ["company_id", "variable", "scope"]
            )
            df3_denom_t = pd.concat(
                {VariablesConfig.PRODUCTIONS: df3_denom_t}, names=["variable"], axis=1
            )
            df3_denom_t = pd.concat(
                {"production": df3_denom_t}, names=["scope"], axis=1
            )
            df3_denom_t.columns = df3_denom_t.columns.reorder_levels(
                ["company_id", "variable", "scope"]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
                df5 = pd.concat([df3_num_t.T, df3_denom_t.T, df4])

            df_historic_data = df5

        # df_target_data now ready for conversion to model for each company
        df_target_data = self._validate_target_data(df_target_data)

        # df_historic now ready for conversion to model for each company
        self.historic_years = [
            column for column in df_historic_data.columns if isinstance(column, int)
        ]

        def get_scoped_df(df, scope, names):
            mask = df.scope.eq(scope)
            return df.loc[mask[mask].index].set_index(names)

        def fill_blank_or_missing_scopes(
            df, scope_a, scope_b, scope_ab, index_names, historic_years
        ):
            # Translate from long format, where each scope is on its own line, to common index
            df_a = get_scoped_df(df, scope_a, index_names)
            df_b = get_scoped_df(df, scope_b, index_names)
            df_ab = get_scoped_df(df, scope_ab, index_names).set_index(
                "scope", append=True
            )
            # This adds rows of SCOPE_AB data that could be created by adding SCOPE_A and SCOPE_B rows
            new_ab_idx = df_a.index.intersection(df_b.index)
            new_ab = (
                df_a.loc[new_ab_idx, historic_years]
                + df_b.loc[new_ab_idx, historic_years]
            )
            new_ab.insert(0, "scope", scope_ab)
            new_ab.set_index("scope", append=True, inplace=True)
            df_ab[df_ab.map(ITR.isna)] = new_ab.loc[
                new_ab.index.intersection(df_ab.index)
            ]
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
        df = fill_blank_or_missing_scopes(
            df, "S1", "S2", "S1S2", index_names, self.historic_years
        )
        df = fill_blank_or_missing_scopes(
            df, "S1S2", "S3", "S1S2S3", index_names, self.historic_years
        )
        df_historic_data = df.set_index(
            ["company_id", "variable", "scope"]
        ).sort_index()
        # We might run `fill_blank_or_missing_scopes` again if we get newly estimated S3 data from an as-yet unknown benchmark

        # Drop from our companies list the companies dropped in df_fundamentals
        self._companies = [
            c for c in self._companies if c.company_id in df_fundamentals.index
        ]
        # Add to our companies list the companies added in df_fundamentals
        self._companies.extend(
            self._company_df_to_model(
                df_fundamentals[
                    df_fundamentals.company_id.str.contains("+", regex=False)
                ],
                pd.DataFrame(),
                pd.DataFrame(),
            )
        )

        if self.template_version > 1:
            # Now we ensure that we convert production metrics to ones that benchmarks can handle
            # Each different company may have its own way of converting from "widgets" to canonical
            # production units with their own factors of conversion (for example Renault Group
            # treats a "vehicle" as having a useful life of 150,000 km; OECM and TPI both us
            # passenger km (pkm) as the unit of production for Automobiles
            prod_rows = df_esg[df_esg.metric.eq("production")]
            for idx, row in df_esg[df_esg.metric.eq("equivalence")].iterrows():
                prod_row = prod_rows[prod_rows.company_id.eq(row.company_id)]
                prod_unit = prod_row.unit.item()
                prod_dim = f"[{prod_unit}]"
                result_unit = str(ureg(f"({prod_unit}) * ({row.unit})").u)
                # Now all rows that match this company_id must be renormalized
                for idx2, row2 in df_historic_data.loc[row.company_id].iterrows():
                    if row2.name[0] == "Emissions":
                        continue
                    if prod_dim in row2.iloc[0].dimensionality.keys():
                        if row2.name[0] == "Productions":
                            df_historic_data.loc[
                                (row.company_id, *idx2), esg_year_columns
                            ] = row[esg_year_columns] * row2[esg_year_columns]
                        elif row2.name[0] == "Emissions Intensities":
                            df_historic_data.loc[
                                (row.company_id, *idx2), esg_year_columns
                            ] = row2[esg_year_columns] / row[esg_year_columns]
                        else:
                            raise ValueError
                # And target data as well
                for idx2, row2 in df_target_data.loc[row.company_id].iterrows():
                    if (
                        prod_dim
                        in ureg(row2.target_base_year_unit).dimensionality.keys()
                    ):
                        normalized_qty = (
                            Q_(
                                row2["target_base_year_qty"],
                                row2["target_base_year_unit"],
                            )
                            / row["base_year"]
                        )
                        df_target_data.loc[idx2, "target_base_year_qty"] = (
                            normalized_qty.m
                        )
                        df_target_data.loc[idx2, "target_base_year_unit"] = str(
                            normalized_qty.u
                        )
                # And fundamental data as well
                df_fundamentals.loc[row.company_id, ColumnsConfig.PRODUCTION_METRIC] = (
                    result_unit
                )

            # We don't need units here anymore--they've been translated/transported everywhere we need them
            df_esg = df_esg.drop(columns="unit")

        for company in self._companies:
            row = df_fundamentals.loc[company.company_id]
            company.emissions_metric = EmissionsMetric(
                row[ColumnsConfig.EMISSIONS_METRIC]
            )
            company.production_metric = ProductionMetric(
                row[ColumnsConfig.PRODUCTION_METRIC]
            )
        # And keep df_fundamentals in sync
        self.df_fundamentals = df_fundamentals

        # company_id, netzero_year, target_type, target_scope, target_start_year,
        # target_base_year, target_base_year_qty, target_base_year_unit, target_year,
        # target_reduction_ambition
        return self._company_df_to_model(None, df_target_data, df_historic_data)

    def _validate_target_data(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """Performs checks on the supplied target data. Some values are put in to make the tool function.
        :param target_data:
        :return:
        """

        def unique_ids(mask):
            return target_data[mask].index.unique().tolist()

        # TODO: need to fix Pydantic definition or data to allow optional int.  In the mean time...
        c_ids_without = {}
        for attr in ["start_year", "base_year", "base_year_qty"]:
            mask = target_data[f"target_{attr}"].isna()
            if mask.any():
                c_ids_without[attr] = unique_ids(mask)
                if attr == "start_year":
                    target_data.loc[mask, f"target_{attr}"] = 2021
                    logger.warning(
                        f"Missing target_{attr} set to 2021 for companies with ID: {c_ids_without[attr]}"
                    )

                    setattr(
                        target_data,
                        f"target_{attr}",
                        getattr(target_data, f"target_{attr}").map(
                            lambda x: int(x.year)
                            if isinstance(x, datetime.datetime)
                            else x
                        ),
                    )
                else:
                    logger.warning(
                        f"Missing target_{attr} for companies with ID: {c_ids_without[attr]}"
                    )
                    target_data = target_data[~mask]

        # Convert CH4 to the GWP of CO2e; don't convert CH4->CO2e in intensity targets
        target_data.reset_index(inplace=True)
        ch4_idx = target_data.target_base_year_unit.str.contains("CH4")
        ch4_gwp = Q_(gwp.data["AR5GWP100"]["CH4"], "CO2e/CH4")
        ch4_maybe_co2e = target_data.loc[ch4_idx].target_base_year_unit.map(
            lambda x: (
                x.replace("CH4", "CO2e")
                if len(dims := ureg.parse_units(x).dimensionality) == 2
                and "[mass]" in dims
                else x
            )
        )
        ch4_is_co2e = target_data.loc[ch4_idx].target_base_year_unit != ch4_maybe_co2e
        ch4_to_co2e = ch4_is_co2e[ch4_is_co2e]
        target_data.loc[ch4_to_co2e.index, "target_base_year_unit"] = (
            ch4_maybe_co2e.loc[ch4_to_co2e.index]
        )
        target_data.loc[ch4_to_co2e.index, "target_base_year_qty"] *= ch4_gwp.m
        target_data.set_index("company_id", inplace=True)

        target_data.loc[
            target_data.target_type.str.lower().str.contains("absolute"), "target_type"
        ] = "absolute"
        target_data.loc[
            target_data.target_type.str.lower().str.contains("intensity"), "target_type"
        ] = "intensity"
        mask = ~target_data.target_type.isin(["absolute", "intensity"])
        if mask.any():
            c_ids_with_invalid_target_type = unique_ids(mask)
            logger.warning(
                f"Invalid target types {target_data[mask].target_type} among companies with ID: {c_ids_with_invalid_target_type}"
            )
            target_data = target_data[~mask]

        mask = target_data["netzero_year"] > ProjectionControls.TARGET_YEAR
        if mask.any():
            c_ids_invalid_netzero_year = unique_ids(mask)
            warning_message = f"Invalid net-zero target years (>{ProjectionControls.TARGET_YEAR}) are entered for companies with ID: {c_ids_invalid_netzero_year}"  # noqa: E501
            logger.warning(warning_message)
            target_data = target_data[~mask]

        mask = target_data.netzero_year.isna()
        if mask.any():
            c_ids_without_netzero_year = unique_ids(mask)
            warning_message = (
                f"Companies without netzero targets: {c_ids_without_netzero_year}"
            )
            # target_data.loc[mask, 'netzero_year'] = ProjectionControls.TARGET_YEAR

        c_ids_with_nonnumeric_target = target_data[
            target_data["target_reduction_ambition"].map(lambda x: isinstance(x, str))
        ].index.tolist()
        if c_ids_with_nonnumeric_target:
            error_message = (
                "Non-numeric target reduction ambition is invalid; please fix companies with ID: "
                f"{c_ids_with_nonnumeric_target}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        c_ids_with_increase_target = list(
            target_data[target_data["target_reduction_ambition"] < 0].index
        )
        if c_ids_with_increase_target:
            error_message = (
                "Negative target reduction ambition is invalid and entered for companies with ID: "
                f"{c_ids_with_increase_target}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # https://stackoverflow.com/a/74193599/1291237
        with warnings.catch_warnings():
            # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
            # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
            # See also: https://stackoverflow.com/q/74057367/859591
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=(
                    ".*will attempt to set the values inplace instead of always setting a new array. "
                    "To retain the old behavior, use either.*"
                ),
            )

            target_data = target_data.assign(
                target_scope=target_data.target_scope.replace(
                    r"[\n\r]+", "+", regex=True
                )
                .replace(r"\bs([123])", r"S\1", regex=True)
                .str.strip()
                .replace(r" ?\+ ?", "+", regex=True)
            )

        return target_data

    def _company_df_to_model(
        self,
        df_fundamentals: pd.DataFrame,
        df_target_data: pd.DataFrame,
        df_historic_data: pd.DataFrame,
    ) -> List[ICompanyData]:
        """Transforms target Dataframe into list of ICompanyData instances.
        We don't necessarily have enough info to do target projections at this stage.

        :param df_fundamentals: pandas Dataframe with fundamental data; if None, use self._companies
        :param df_target_data: pandas Dataframe with target data; could be empty if we are partially initialized
        :param df_historic_data: pandas Dataframe with historic emissions, intensity, and production information; could be empty
        :return: A list containing the ICompanyData objects
        """
        if df_fundamentals is not None:
            companies_data_dict = df_fundamentals.to_dict(orient="records")
        else:
            companies_data_dict = [dict(c) for c in self._companies]
        model_companies: List[ICompanyData] = []
        base_year = self.projection_controls.BASE_YEAR

        for company_data in companies_data_dict:
            # company_data is a dict, not a dataframe
            try:
                # In this world (different from excel.py) we initialize projected_intensities and projected_targets
                # in a later step, after we know we have valid benchmark data
                company_id = company_data[ColumnsConfig.COMPANY_ID]

                # the ghg_s1s2 and ghg_s3 variables are values "as of" the financial data
                # TODO pull ghg_s1s2 and ghg_s3 from historic data as appropriate

                if not df_historic_data.empty:
                    df_historic_data = df_historic_data.sort_index(
                        level=df_historic_data.index.names
                    )
                    # FIXME: Is this the best place to finalize base_year_production, ghg_s1s2, and ghg_s3 data?
                    # Something tells me these parameters should be removed in favor of querying historical data directly
                    company_data[ColumnsConfig.BASE_YEAR_PRODUCTION] = (
                        df_historic_data.loc[
                            company_id, "Productions", "production"
                        ][base_year]
                    )
                    try:
                        company_data[ColumnsConfig.GHG_SCOPE12] = df_historic_data.loc[
                            company_id, "Emissions", "S1S2"
                        ].loc[base_year]
                    except KeyError:
                        if (
                            company_id,
                            "Emissions",
                            "S2",
                        ) not in df_historic_data.index:
                            logger.warning(
                                f"Scope 2 data missing from company with ID {company_id}; treating as zero"
                            )
                            try:
                                company_data[ColumnsConfig.GHG_SCOPE12] = (
                                    df_historic_data.loc[
                                        company_id, "Emissions", "S1"
                                    ].loc[base_year]
                                )
                                df_historic_data.loc[company_id, "Emissions", "S2"] = 0
                                df_historic_data.loc[
                                    company_id, "Emissions Intensities", "S2"
                                ] = 0
                                df_historic_data.loc[
                                    (company_id, "Emissions", "S2"), :
                                ] = (
                                    df_historic_data.loc[company_id, "Emissions", "S1"]
                                    * 0
                                )
                                df_historic_data.loc[
                                    (company_id, "Emissions Intensities", "S2"), :
                                ] = (
                                    df_historic_data.loc[
                                        company_id, "Emissions Intensities", "S1"
                                    ]
                                    * 0
                                )
                                df_historic_data = df_historic_data.sort_index(
                                    level=df_historic_data.index.names
                                )
                            except KeyError:
                                try:
                                    company_data[ColumnsConfig.GHG_SCOPE12] = (
                                        df_historic_data.loc[
                                            company_id, "Emissions", "S1S2S3"
                                        ].loc[base_year]
                                    )
                                    logger.warning(
                                        f"Using S1+S2+S3 as GHG_SCOPE12 because no Scope 1 or Scope 2 available for company with ID {company_id}"
                                    )
                                    # FIXME: we should not allocate these here, but rather in the benchmark alignment code
                                    if False:
                                        df_historic_data.loc[
                                            company_id, "Emissions", "S1S2"
                                        ] = df_historic_data.loc[
                                            company_id, "Emissions Intensities", "S1S2"
                                        ] = 0
                                        df_historic_data.loc[
                                            (company_id, "Emissions", "S1S2"), :
                                        ] = (
                                            df_historic_data.loc[
                                                company_id, "Emissions", "S1S2S3"
                                            ]
                                            * 0
                                        )
                                        df_historic_data.loc[
                                            (
                                                company_id,
                                                "Emissions Intensities",
                                                "S1S2",
                                            ),
                                            :,
                                        ] = (
                                            df_historic_data.loc[
                                                company_id,
                                                "Emissions Intensities",
                                                "S1S2S3",
                                            ]
                                            * 0
                                        )
                                except KeyError:
                                    logger.error(
                                        f"Company {company_id} snuck into finalization without any useable S1, S2, S1+S2, or S1+S2+S3 data"
                                    )
                                    company_data[ColumnsConfig.GHG_SCOPE12] = Q_(
                                        np.nan, "Mt CO2e"
                                    )
                        else:
                            # S1S2 as an emissions total upstream from S1+S2.  While normally done upstream, not done for newly created company_ids.
                            try:
                                company_data[ColumnsConfig.GHG_SCOPE12] = (
                                    df_historic_data.loc[
                                        company_id, "Emissions", "S1"
                                    ].loc[base_year]
                                    + df_historic_data.loc[
                                        company_id, "Emissions", "S2"
                                    ].loc[base_year]
                                )
                                df_historic_data.loc[
                                    company_id, "Emissions", "S1S2"
                                ] = df_historic_data.loc[
                                    company_id, "Emissions Intensities", "S1S2"
                                ] = 0
                                df_historic_data.loc[
                                    (company_id, "Emissions", "S1S2"), :
                                ] = (
                                    df_historic_data.loc[company_id, "Emissions", "S1"]
                                    + df_historic_data.loc[
                                        company_id, "Emissions", "S2"
                                    ]
                                )
                                df_historic_data.loc[
                                    (company_id, "Emissions Intensities", "S1S2"), :
                                ] = (
                                    df_historic_data.loc[
                                        company_id, "Emissions Intensities", "S1"
                                    ]
                                    + df_historic_data.loc[
                                        company_id, "Emissions Intensities", "S2"
                                    ]
                                )
                            except KeyError:
                                logger.error(
                                    f"Scope 1 data missing from Company with ID {company_id}; treating as zero"
                                )
                                company_data[ColumnsConfig.GHG_SCOPE12] = (
                                    df_historic_data.loc[
                                        company_id, "Emissions", "S2"
                                    ].loc[base_year]
                                )
                                df_historic_data.loc[company_id, "Emissions", "S1"] = 0
                                df_historic_data.loc[
                                    company_id, "Emissions Intensities", "S1"
                                ] = 0
                                df_historic_data.loc[
                                    (company_id, "Emissions", "S1"), :
                                ] = (
                                    df_historic_data.loc[company_id, "Emissions", "S2"]
                                    * 0
                                )
                                df_historic_data.loc[
                                    (company_id, "Emissions Intensities", "S1"), :
                                ] = (
                                    df_historic_data.loc[
                                        company_id, "Emissions Intensities", "S2"
                                    ]
                                    * 0
                                )
                                df_historic_data = df_historic_data.sort_index(
                                    level=df_historic_data.index.names
                                )
                    try:
                        company_data[ColumnsConfig.GHG_SCOPE3] = df_historic_data.loc[
                            company_id, "Emissions", "S3"
                        ].loc[base_year]
                    except KeyError:
                        # If there was no relevant historic S3 data, don't try to use it
                        pass
                    company_data[ColumnsConfig.HISTORIC_DATA] = dict(
                        self._convert_historic_data(
                            df_historic_data.loc[[company_id]].reset_index()
                        )
                    )
                else:
                    company_data[ColumnsConfig.HISTORIC_DATA] = None

                if company_id in df_target_data.index:
                    company_data[ColumnsConfig.TARGET_DATA] = [
                        dict(td)
                        for td in self._convert_target_data(
                            # don't let a single row of df_target_data become a pd.Series
                            df_target_data.loc[[company_id]].reset_index()
                        )
                    ]
                else:
                    company_data[ColumnsConfig.TARGET_DATA] = None

                # handling of missing market cap data is mainly done in _convert_from_template_company_data()
                # FIXME: needs currency units!
                if company_data[ColumnsConfig.COMPANY_MARKET_CAP] is pd.NA:
                    company_data[ColumnsConfig.COMPANY_MARKET_CAP] = np.nan

                model_companies.append(ICompanyData.model_validate(company_data))
            except ValidationError as err:
                logger.error(
                    f"{err}: (One of) the input(s) of company with ID {company_id} is invalid"
                )
                # breakpoint()
                raise
        return model_companies

    # Workaround for bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero
    def _np_sum(g):
        return np.sum(g.values)

    def _convert_target_data(self, target_data: pd.DataFrame) -> List[ITargetData]:
        """:param historic: historic production, emission and emission intensity data for a company
        :return: IHistoricData Pydantic object
        """
        target_data = target_data.rename(
            columns={
                "target_year": "target_end_year",
                "target_reduction_ambition": "target_reduction_pct",
            }
        )
        target_data = target_data.assign(
            netzero_year=target_data.netzero_year.astype("object").replace(
                {pd.NA: None}
            )
        )
        return [ITargetData(**td) for td in target_data.to_dict("records")]

    def _get_historic_data(
        self, company_ids: List[str], historic_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the historic data for list of companies
        :param company_ids: list of company ids
        :param historic_data: Dataframe Productions, Emissions, and Emissions Intensities mixed together
        :return: historic data with unit attributes added on a per-element basis
        """
        missing_ids = [
            company_id
            for company_id in company_ids
            if company_id not in historic_data.index
        ]
        if missing_ids:
            error_message = (
                f"Company ids missing in provided historic data: {missing_ids}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        for year in self.historic_years:
            historic_data[year] = (
                historic_data[year].map(str) + " " + historic_data["units"]
            )
        return historic_data.loc[company_ids]

    # In the following several methods, we implement SCOPE as STRING (used by Excel handlers)
    # so that the resulting scope dictionary can be used to pass values to named arguments
    def _convert_historic_data(self, historic: pd.DataFrame) -> IHistoricData:
        """:param historic: historic production, emission and emission intensity data for a company (already unitized)
        :return: IHistoricData Pydantic object
        """
        try:
            # Save ``company_id`` in case we need it for an error message
            company_id = historic.company_id.iloc[0]
            historic = historic.drop(columns="company_id")
            historic_t = asPintDataFrame(historic.set_index(["variable", "scope"]).T)
        except pint.errors.DimensionalityError as err:
            logger.error(
                f"Dimensionality error {err} in 'historic' DataFrame for company_id {company_id}:\n{historic}"
            )
            # breakpoint()
            raise

        # The conversion routines all use transposed data to preserve PintArray columns
        hd = IHistoricData(
            productions=self._convert_to_historic_productions(
                historic_t[VariablesConfig.PRODUCTIONS]
            ),
            emissions=self._convert_to_historic_emissions(
                historic_t[VariablesConfig.EMISSIONS]
            ),
            emissions_intensities=self._convert_to_historic_ei(
                historic_t[VariablesConfig.EMISSIONS_INTENSITIES]
            ),
        )
        return hd

    @classmethod
    def _squeeze_NA_to_nan(cls, ser):
        x = ser
        if isinstance(x, pd.Series):
            x = x.squeeze()
        if isinstance(x, pint.Quantity):
            if x.m is pd.NA:
                return PintType(x.u).na_value
        return x

    # Note that for the three following functions, we pd.Series.squeeze() the results because it's just one year / one company
    def _convert_to_historic_emissions(
        self, emissions_t: pd.DataFrame
    ) -> Optional[IHistoricEmissionsScopes]:
        """:param emissions: historic emissions data for a company
        :return: List of historic emissions per scope, or None if no data are provided
        """
        if emissions_t.empty:
            return None
        emissions_scopes: Dict[str, List[IEmissionRealization]] = dict.fromkeys(
            EScope.get_scopes(), []
        )
        for scope_name, emissions in emissions_t.items():
            if not emissions.empty:
                emissions_scopes[scope_name] = [
                    IEmissionRealization(
                        year=year,
                        value=TemplateProviderCompany._squeeze_NA_to_nan(
                            emissions.loc[year]
                        ),
                    )
                    for year in self.historic_years
                ]
        return IHistoricEmissionsScopes(**emissions_scopes)

    def _convert_to_historic_productions(
        self, productions_t: pd.DataFrame
    ) -> Optional[List[IProductionRealization]]:
        """:param productions: historic production data for a company
        :return: A list containing historic productions, or None if no data are provided
        """
        if productions_t.empty:
            return None
        return [
            IProductionRealization(
                year=year,
                value=TemplateProviderCompany._squeeze_NA_to_nan(
                    productions_t.loc[year]
                ),
            )
            for year in self.historic_years
        ]

    def _convert_to_historic_ei(
        self, intensities_t: pd.DataFrame
    ) -> Optional[IHistoricEIScopes]:
        """:param intensities: historic emission intensity data for a company
        :return: A list of historic emission intensities per scope, or None if no data are provided
        """
        if intensities_t.empty:
            return None
        intensity_scopes: Dict[str, List[IEIRealization]] = dict.fromkeys(
            EScope.get_scopes(), []
        )
        for scope_name, intensities in intensities_t.items():
            if not intensities.empty:
                intensity_scopes[scope_name] = [
                    IEIRealization(
                        year=year,
                        value=TemplateProviderCompany._squeeze_NA_to_nan(
                            intensities.loc[year]
                        ),
                    )
                    for year in self.historic_years
                ]
        return IHistoricEIScopes(**intensity_scopes)

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """:param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company (company_id is a column)

        FIXME: Callers want non-fundamental data here: base_year_production, ghg_s1s2, ghg_s3
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
        # company_ids_idx = pd.Index(company_ids)
        # df = self.df_fundamentals.loc[company_ids_idx]
        return df
