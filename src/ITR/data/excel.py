import logging
import warnings  # needed until apply behaves better with Pint quantities in arrays
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from ..configs import (
    ColumnsConfig,
    LoggingConfig,
    ProjectionControls,
    TabsConfig,
    VariablesConfig,
)
from ..data import Q_
from ..data.base_providers import (
    BaseCompanyDataProvider,
    BaseProviderIntensityBenchmark,
    BaseProviderProductionBenchmark,
)
from ..data.osc_units import (
    BenchmarkMetric,
    EI_Quantity,
    EmissionsQuantity,
    delta_degC_Quantity,
)
from ..interfaces import (
    EScope,
    IBenchmark,
    IBenchmarks,
    ICompanyData,
    ICompanyEIProjection,
    IEIBenchmarkScopes,
    IEIRealization,
    IEmissionRealization,
    IHistoricData,
    IHistoricEIScopes,
    IHistoricEmissionsScopes,
    IProductionBenchmarkScopes,
    IProductionRealization,
    IProjection,
    ITargetData,
    ProductionQuantity,
    UProjection,
)

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


# Excel spreadsheets don't have units elaborated, so we translate sectors to units
# FIXME: this is now out of data with our much better JSON-based benchmark data
sector_to_production_metric = {
    "Electricity Utilities": "GJ",
    "Steel": "Fe_ton",
    "Oil & Gas": "boe",
    "Autos": "pkm",
}
sector_to_intensity_metric = {
    "Electricity Utilities": "t CO2/MWh",
    "Steel": "t CO2/Fe_ton",
    "Oil & Gas": "kg CO2/boe",
    "Autos": "g CO2/pkm",
}

# TODO: Force validation for excel benchmarks

# Utils functions:


def convert_dimensionless_benchmark_excel_to_model(
    df_excel: dict, sheetname: str, column_name_sector: str, column_name_region: str
) -> IBenchmarks:
    """Converts excel into IBenchmarks
    :param df_excel: dictionary with a pd.DataFrame for each key representing a sheet of an Excel file
    :param sheetname: name of Excel file sheet to convert
    :param column_name_sector: name of sector
    :param column_name_region: name of region
    :return: IBenchmarks instance (list of IBenchmark)
    """
    try:
        df_sheet = df_excel[sheetname]
    except KeyError:
        logger.error(f"Sheet {sheetname} not in benchmark Excel file.")
        raise

    df_production = df_sheet.reset_index(drop=True).set_index(
        [column_name_sector, column_name_region, "benchmark_metric", "scope"]
    )

    result = []
    # FIXME: More pythonic to convert DF to dict and convert dict to Model
    for index, row in df_production.iterrows():
        bm = IBenchmark(
            sector=index[0],
            region=index[1],
            benchmark_metric=BenchmarkMetric("dimensionless"),
            projections_nounits=[
                UProjection(year=int(k), value=float(v)) for k, v in row.items()
            ],
        )
        result.append(bm)
    return IBenchmarks(benchmarks=result)


def convert_benchmarks_ei_excel_to_model(
    df_excel: pd.DataFrame,
    sheetname: str,
    column_name_sector: str,
    column_name_region: str,
    benchmark_temperature,
    benchmark_global_budget,
    is_AFOLU_included,
) -> IEIBenchmarkScopes:
    """Converts excel into IBenchmarks
    :param excal_path: file path to excel
    :return: IEIBenchmarkScopes instance
    """
    df_ei_bms = (
        df_excel[sheetname]
        .reset_index(drop=True)
        .set_index(
            [column_name_sector, column_name_region, "benchmark_metric", "scope"]
        )
    )
    bm_dict: Dict[str, Any] = {scope_name: [] for scope_name in EScope.get_scopes()}
    bm_dict["benchmark_temperature"] = benchmark_temperature
    bm_dict["benchmark_global_budget"] = benchmark_global_budget
    bm_dict["is_AFOLU_included"] = is_AFOLU_included
    # FIXME: More pythonic to convert DF to dict and convert dict to Model
    for index, row in df_ei_bms.iterrows():
        bm = IBenchmark(
            sector=index[0],
            region=index[1],
            benchmark_metric=index[2],
            projections=[
                IProjection(year=int(k), value=Q_(float(v), index[2]))
                for k, v in row.items()
            ],
        )
        bm_dict[index[3]].append(bm)
    for scope_name in EScope.get_scopes():
        bm_dict[scope_name] = IBenchmarks(benchmarks=bm_dict[scope_name])
    return IEIBenchmarkScopes(**bm_dict)


class ExcelProviderProductionBenchmark(BaseProviderProductionBenchmark):
    def __init__(
        self, excel_path: str, column_config: Type[ColumnsConfig] = ColumnsConfig
    ):
        """Overrices BaseProvider and provides an interfaces for excel the excel template
        :param excel_path: file path to excel
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        for sheetname, df in self.benchmark_excel.items():
            # This fills down the sector information for different regions
            self.benchmark_excel[sheetname] = df.ffill()
        self._convert_excel_to_model = convert_dimensionless_benchmark_excel_to_model
        production_bms = self._convert_excel_to_model(
            self.benchmark_excel,
            TabsConfig.PROJECTED_PRODUCTION,
            column_config.SECTOR,
            column_config.REGION,
        )
        super().__init__(
            IProductionBenchmarkScopes(AnyScope=production_bms), column_config
        )

    def _get_projected_production(self, scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """Interface from excel file and internally used DataFrame
        :param scope:
        :return:
        """
        # df = self.benchmark_excel[TabsConfig.PROJECTED_PRODUCTION].drop(columns='benchmark_metric')
        # df.loc[:, 'scope'] = df.scope.map(lambda x: EScope[x])
        # df.set_index([self.column_config.SECTOR, self.column_config.REGION, self.column_config.SCOPE], inplace=True)
        # df_partial_pp = df.add(1).cumprod(axis=1).astype('pint[dimensionless]')
        # return df_partial_pp
        return self._prod_df


class ExcelProviderIntensityBenchmark(BaseProviderIntensityBenchmark):
    def __init__(
        self,
        excel_path: str,
        benchmark_temperature: delta_degC_Quantity,
        benchmark_global_budget: EmissionsQuantity,
        is_AFOLU_included: bool,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
    ):
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        for sheetname, df in self.benchmark_excel.items():
            # This fills down the sector information for different regions
            self.benchmark_excel[sheetname] = df.ffill()
        self._convert_excel_to_model = convert_benchmarks_ei_excel_to_model
        ei_bm_scopes = self._convert_excel_to_model(
            self.benchmark_excel,
            TabsConfig.PROJECTED_EI,
            column_config.SECTOR,
            column_config.REGION,
            benchmark_temperature,
            benchmark_global_budget,
            is_AFOLU_included,
        )
        super().__init__(ei_bm_scopes, column_config)


# FIXME: Should we merge with TemplateProviderCompany and just use a different excel input method
# for this "simple" case?
class ExcelProviderCompany(BaseCompanyDataProvider):
    """Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param excel_path: A path to the Excel file with the company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    """

    def __init__(
        self,
        excel_path: str,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
    ):
        self.projection_controls = projection_controls
        self._companies = self._convert_from_excel_data(excel_path)
        self.historic_years: List[int] = []
        super().__init__(self._companies, column_config, projection_controls)

    def _check_company_data(self, company_tabs: dict) -> None:
        """Checks if the company data excel contains the data in the right format

        :return: None
        """
        required_tabs = {TabsConfig.FUNDAMENTAL, TabsConfig.PROJECTED_TARGET}
        optional_tabs = {TabsConfig.PROJECTED_EI, TabsConfig.HISTORIC_DATA}
        missing_tabs = (required_tabs | optional_tabs).difference(set(company_tabs))
        if missing_tabs.intersection(required_tabs):
            logger.error(f"Tabs {required_tabs} are required.")
            raise ValueError(f"Tabs {required_tabs} are required.")
        if optional_tabs.issubset(missing_tabs):
            logger.error(f"Either of the tabs {optional_tabs} is required.")
            raise ValueError(f"Either of the tabs {optional_tabs} is required.")

    def _convert_from_excel_data(self, excel_path: str) -> List[ICompanyData]:
        """Converts the Excel template to list of ICompanyDta objects. All dataprovider features will be inhereted from
        Base
        :param excel_path: file path to excel file
        :return: List of ICompanyData objects
        """
        company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_company_data(company_data)

        df_fundamentals = company_data[TabsConfig.FUNDAMENTAL].set_index(
            ColumnsConfig.COMPANY_ID
        )
        df_fundamentals[ColumnsConfig.PRODUCTION_METRIC] = df_fundamentals[
            ColumnsConfig.SECTOR
        ].map(sector_to_production_metric)
        company_ids = df_fundamentals.index.unique().get_level_values(level=0).tolist()
        # _get_projection creates S1S2 data from S1+S2.  _get_historic_data must do the same to keep up.
        df_targets = self._get_projection(
            company_ids,
            company_data[TabsConfig.PROJECTED_TARGET],
            df_fundamentals[ColumnsConfig.PRODUCTION_METRIC],
        )
        if TabsConfig.PROJECTED_EI in company_data:
            df_ei = self._get_projection(
                company_ids,
                company_data[TabsConfig.PROJECTED_EI],
                df_fundamentals[ColumnsConfig.PRODUCTION_METRIC],
            )
        else:
            df_ei = None
        if TabsConfig.HISTORIC_DATA in company_data:
            df_historic = company_data[TabsConfig.HISTORIC_DATA].set_index(
                ColumnsConfig.COMPANY_ID
            )
            # DON'T update historic data
            if False:
                df_prods = df_historic[df_historic.variable == "Productions"].drop(
                    columns=["variable", "scope"]
                )
                df_s1s2 = (
                    df_historic[df_historic.variable == "Emissions"]
                    .groupby(by=["company_id", "variable"])
                    .sum()
                    .droplevel("variable")
                )
                df_s1s2_ei = (
                    df_historic[df_historic.variable == "Emissions Intensities"]
                    .groupby(by=["company_id", "variable"])
                    .sum()
                    .droplevel("variable")
                )
                em_not_ei = df_s1s2.index.difference(df_s1s2_ei.index).intersection(
                    df_prods.index
                )
                ei_not_em = df_s1s2_ei.index.difference(df_s1s2.index).intersection(
                    df_prods.index
                )
                df_ei_from_em = df_s1s2.loc[em_not_ei].div(df_prods.loc[em_not_ei])
                df_em_from_ei = df_s1s2_ei.loc[ei_not_em].mul(df_prods.loc[ei_not_em])
                df_s1s2["scope"] = df_s1s2_ei["scope"] = df_ei_from_em[
                    "scope"
                ] = df_em_from_ei["scope"] = "S1S2"
                df_s1s2["variable"] = df_em_from_ei["variable"] = "Emissions"
                df_s1s2_ei["variable"] = df_ei_from_em["variable"] = (
                    "Emissions Intensities"
                )
                df_historic = pd.concat(
                    [df_historic, df_s1s2, df_s1s2_ei, df_ei_from_em, df_em_from_ei]
                )
            df_historic = df_historic.merge(
                df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].rename("units"),
                left_index=True,
                right_index=True,
            )
            df_historic.loc[df_historic.variable == "Emissions", "units"] = "t CO2"
            # If you think the following line of code is ugly, please answer https://stackoverflow.com/q/74555323/1291237
            df_historic.loc[
                df_historic.variable == "Emissions Intensities", "units"
            ] = df_historic.loc[
                df_historic.variable == "Emissions Intensities"
            ].units.map(lambda x: f"t CO2/({x})")
            df_historic = self._get_historic_data(company_ids, df_historic)
            company_data[TabsConfig.HISTORIC_DATA] = df_historic
        else:
            df_historic = None

        return self._company_df_to_model(
            df_fundamentals, df_targets, df_ei, df_historic
        )

    def _convert_series_to_projections(
        self, projections: pd.Series, ProjectionType: BaseModel
    ) -> List[IProjection]:
        """Converts a Pandas Series to a list of IProjection
        :param projections: Pandas Series with years as indices
        :return: List of IProjection objects
        """
        return [ProjectionType(year=y, value=v) for y, v in projections.items()]

    def _company_df_to_model(
        self,
        df_fundamentals: pd.DataFrame,
        df_targets: pd.DataFrame,
        df_ei: pd.DataFrame,
        df_historic: pd.DataFrame,
    ) -> List[ICompanyData]:
        """Transforms target Dataframe into list of IDataProviderTarget instances
        :param df_fundamentals: pandas Dataframe with fundamental data
        :param df_targets: pandas Dataframe with targets
        :param df_ei: pandas Dataframe with emission intensities
        :return: A list containing the ICompanyData objects
        """
        # set NaN to None since NaN is float instance
        df_fundamentals = df_fundamentals.where(
            pd.notnull(df_fundamentals), None
        ).replace({np.nan: None})

        companies_data_dict = df_fundamentals.to_dict(orient="index")
        model_companies: List[ICompanyData] = []
        for company_id, company_data in companies_data_dict.items():
            try:
                production_metric = sector_to_production_metric[
                    company_data[ColumnsConfig.SECTOR]
                ]
                intensity_metric = sector_to_intensity_metric[
                    company_data[ColumnsConfig.SECTOR]
                ]
                company_data[ColumnsConfig.PRODUCTION_METRIC] = production_metric
                company_data[ColumnsConfig.EMISSIONS_METRIC] = "t CO2"
                # pint automatically handles any unit conversions required

                v = df_fundamentals.loc[company_id][ColumnsConfig.GHG_SCOPE12]
                company_data[ColumnsConfig.GHG_SCOPE12] = Q_(
                    np.nan if v is None else v, "t CO2"
                )
                company_data[ColumnsConfig.BASE_YEAR_PRODUCTION] = (
                    company_data[ColumnsConfig.GHG_SCOPE12]
                    / df_ei.loc[company_id, :][self.projection_controls.BASE_YEAR]
                )
                v = df_fundamentals.loc[company_id][ColumnsConfig.GHG_SCOPE3]
                company_data[ColumnsConfig.GHG_SCOPE3] = Q_(
                    np.nan if v is None else v, "t CO2"
                )
                company_data[ColumnsConfig.PROJECTED_TARGETS] = {
                    "S1S2": {
                        "projections": self._convert_series_to_projections(
                            df_targets.loc[company_id, :], ICompanyEIProjection
                        ),
                        "ei_metric": intensity_metric,
                    }
                }
                company_data[ColumnsConfig.PROJECTED_EI] = {
                    "S1S2": {
                        "projections": self._convert_series_to_projections(
                            df_ei.loc[company_id, :], ICompanyEIProjection
                        ),
                        "ei_metric": intensity_metric,
                    }
                }

                fundamental_metrics = [
                    ColumnsConfig.COMPANY_MARKET_CAP,
                    ColumnsConfig.COMPANY_REVENUE,
                    ColumnsConfig.COMPANY_ENTERPRISE_VALUE,
                    ColumnsConfig.COMPANY_TOTAL_ASSETS,
                    ColumnsConfig.COMPANY_CASH_EQUIVALENTS,
                ]
                for col in fundamental_metrics:
                    company_data[col] = Q_(
                        company_data[col], company_data[ColumnsConfig.COMPANY_CURRENCY]
                    )
                company_data[ColumnsConfig.COMPANY_EV_PLUS_CASH] = (
                    company_data[ColumnsConfig.COMPANY_ENTERPRISE_VALUE]
                    + company_data[ColumnsConfig.COMPANY_CASH_EQUIVALENTS]
                )
                if df_historic is not None:
                    company_data[TabsConfig.HISTORIC_DATA] = dict(
                        self._convert_historic_data(df_historic.loc[company_id, :])
                    )
                else:
                    company_data[TabsConfig.HISTORIC_DATA] = None

                # Put the index back into the dictionary so model builds correctly
                company_data[ColumnsConfig.COMPANY_ID] = company_id
                model_companies.append(ICompanyData.model_validate(company_data))
            except ValidationError as e:
                logger.warning(
                    f"EX {e}: (one of) the input(s) of company %s is invalid and will be skipped"
                    % company_data[ColumnsConfig.COMPANY_NAME]
                )
        return model_companies

    # Workaround for bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero
    def _np_sum(g):
        return np.sum(g.values)

    # FIXME: Need to deal with S3 emissions as well
    def _get_projection(
        self,
        company_ids: List[str],
        projections: pd.DataFrame,
        production_metric: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get the projected emission intensities for list of companies
        :param company_ids: list of company ids
        :param projections: Dataframe with listed projections per company
        :param production_metric: Dataframe with production_metric per company
        :return: series of projected emission intensities
        """
        projections = projections.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        missing_companies = [
            company_id
            for company_id in company_ids
            if company_id not in projections.index
        ]
        if missing_companies:
            error_message = f"Missing target or trajectory projections for companies with ID: {missing_companies}"
            logger.error(error_message)
            raise ValueError(error_message)

        projections = projections.loc[
            company_ids,
            range(
                self.projection_controls.BASE_YEAR,
                self.projection_controls.TARGET_YEAR + 1,
            ),
        ]
        # The following creates projections for S1+S2 from data giving only S1 and S2; Need to get historic_data to see that...
        # Due to bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero workaround below:
        projected_emissions_s1s2 = projections.groupby(level=0, sort=False).agg(
            ExcelProviderCompany._np_sum
        )  # add scope 1 and 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            projected_emissions_s1s2 = projected_emissions_s1s2.apply(
                lambda x: x.astype(f"pint[t CO2/({production_metric[x.name]})]"), axis=1
            )

        return projected_emissions_s1s2

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
        return [ITargetData(**td) for td in target_data.to_dict("records")]

    def _get_historic_data(
        self, company_ids: List[str], historic_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the historic data for list of companies
        :param company_ids: list of company ids
        :param historic_data: Dataframe Productions, Emissions, and Emissions Intensities mixed together
        :return: historic data with unit attributes added on a per-element basis
        """
        self.historic_years = [
            column for column in historic_data.columns if isinstance(column, int)
        ]

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
            historic_data[year] = historic_data.apply(
                lambda x: f"{x[year]} {x.units}", axis=1
            )
        return historic_data.loc[company_ids]

    # In the following several methods, we implement SCOPE as STRING (used by Excel handlers)
    # so that the resulting scope dictionary can be used to pass values to named arguments
    def _convert_historic_data(self, historic: pd.DataFrame) -> IHistoricData:
        """:param historic: historic production, emission and emission intensity data for a company (already unitized)
        :return: IHistoricData Pydantic object
        """
        productions = historic.loc[
            historic[ColumnsConfig.VARIABLE] == VariablesConfig.PRODUCTIONS
        ]
        emissions = historic.loc[
            historic[ColumnsConfig.VARIABLE] == VariablesConfig.EMISSIONS
        ]
        emissions_intensities = historic.loc[
            historic[ColumnsConfig.VARIABLE] == VariablesConfig.EMISSIONS_INTENSITIES
        ]
        hd = IHistoricData(
            productions=self._convert_to_historic_productions(productions),
            emissions=self._convert_to_historic_emissions(emissions),
            emissions_intensities=self._convert_to_historic_ei(emissions_intensities),
        )
        return hd

    # Note that for the three following functions, we pd.Series.squeeze() the results because it's just one year / one company
    def _convert_to_historic_emissions(
        self, emissions: pd.DataFrame
    ) -> Optional[IHistoricEmissionsScopes]:
        """:param emissions: historic emissions data for a company
        :return: List of historic emissions per scope, or None if no data are provided
        """
        if emissions.empty:
            return None

        emissions_scopes = {}
        for scope_name in EScope.get_scopes():
            results = emissions.loc[emissions[ColumnsConfig.SCOPE] == scope_name]
            emissions_scopes[scope_name] = (
                []
                if results.empty
                else [
                    IEmissionRealization(
                        year=year, value=EmissionsQuantity(results[year].squeeze())
                    )
                    for year in self.historic_years
                ]
            )
        return IHistoricEmissionsScopes(**emissions_scopes)

    def _convert_to_historic_productions(
        self, productions: pd.DataFrame
    ) -> Optional[List[IProductionRealization]]:
        """:param productions: historic production data for a company
        :return: A list containing historic productions, or None if no data are provided
        """
        if productions.empty:
            return []
        return [
            IProductionRealization(
                year=year, value=ProductionQuantity(productions[year].squeeze())
            )
            for year in self.historic_years
        ]

    def _convert_to_historic_ei(
        self, intensities: pd.DataFrame
    ) -> Optional[IHistoricEIScopes]:
        """:param intensities: historic emission intensity data for a company
        :return: A list of historic emission intensities per scope, or None if no data are provided
        """
        if intensities.empty:
            return None

        intensities = intensities.copy()
        intensity_scopes = {}

        for scope_name in EScope.get_scopes():
            results = intensities.loc[intensities[ColumnsConfig.SCOPE] == scope_name]
            intensity_scopes[scope_name] = (
                []
                if results.empty
                else [
                    IEIRealization(
                        year=year, value=EI_Quantity(results[year].squeeze())
                    )
                    for year in self.historic_years
                ]
            )
        return IHistoricEIScopes(**intensity_scopes)
