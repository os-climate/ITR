import warnings # needed until apply behaves better with Pint quantities in arrays
from typing import Type, List, Optional
import pandas as pd
import numpy as np
from pint import Quantity
import pint
import logging

from pydantic import ValidationError
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, VariablesConfig, TabsConfig, LoggingConfig
from ITR.interfaces import BaseModel, ICompanyData, ICompanyEIProjection, EScope, IEIBenchmarkScopes, \
    IProductionBenchmarkScopes, IBenchmark, IBenchmarks, IHistoricEmissionsScopes, \
    IProductionRealization, IHistoricEIScopes, IHistoricData, IEmissionRealization, IEIRealization, IProjection

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)

ureg = pint.get_application_registry()
Q_ = ureg.Quantity


# Excel spreadsheets don't have units elaborated, so we translate sectors to units
sector_to_production_metric = {'Electricity Utilities': 'GJ', 'Steel': 'Fe_ton'}
sector_to_intensity_metric = {'Electricity Utilities': 't CO2/MWh', 'Steel': 't CO2/Fe_ton'}

# TODO: Force validation for excel benchmarks

# Utils functions:

def convert_dimensionless_benchmark_excel_to_model(df_excel: dict, sheetname: str, column_name_region: str,
                                                   column_name_sector: str) -> IBenchmarks:
    """
    Converts excel into IBenchmarks
    :param df_excel: dictionary with a pd.DataFrame for each key representing a sheet of an Excel file
    :param sheetname: name of Excel file sheet to convert
    :param column_name_region: name of region
    :param column_name_sector: name of sector
    :return: IBenchmarks instance (list of IBenchmark)
    """
    try:
        df_sheet = df_excel[sheetname]
    except KeyError:
        logger.error(f"Sheet {sheetname} not in benchmark Excel file.")
        raise

    df_ei_bms = df_sheet.reset_index().drop(columns=['index']).set_index(
        [column_name_region, column_name_sector])

    result = []
    for index, row in df_ei_bms.iterrows():
        bm = IBenchmark(region=index[0], sector=index[1], benchmark_metric={'units': 'dimensionless'},
                        projections=[IProjection(year=int(k), value=Q_(v, ureg('dimensionless'))) for k, v in row.items()])
        result.append(bm)
    return IBenchmarks(benchmarks=result)


def convert_intensity_benchmark_excel_to_model(df_excel: pd.DataFrame, sheetname: str, column_name_region: str,
                                               column_name_sector: str) -> IBenchmarks:
    """
    Converts excel into IBenchmarks
    :param excal_path: file path to excel
    :return: IBenchmarks instance (list of IBenchmark)
    """
    df_ei_bms = df_excel[sheetname].reset_index().drop(columns=['index']).set_index(
        [column_name_region, column_name_sector])
    result = []
    for index, row in df_ei_bms.iterrows():
        intensity_units = sector_to_intensity_metric[index[1]]
        bm = IBenchmark(region=index[0], sector=index[1], benchmark_metric={'units':intensity_units},
                        projections=[IProjection(year=int(k), value=Q_(v, ureg(intensity_units))) for k, v in row.items()])
        result.append(bm)
    return IBenchmarks(benchmarks=result)


class ExcelProviderProductionBenchmark(BaseProviderProductionBenchmark):
    def __init__(self, excel_path: str, column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        """
        Overrices BaseProvider and provides an interfaces for excel the excel template
        :param excel_path: file path to excel
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
        """
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._convert_excel_to_model = convert_dimensionless_benchmark_excel_to_model
        production_bms = self._convert_excel_to_model(self.benchmark_excel, TabsConfig.PROJECTED_PRODUCTION,
                                                      column_config.REGION, column_config.SECTOR)
        super().__init__(
            IProductionBenchmarkScopes(benchmark_metric={'units': 'dimensionless'}, S1S2=production_bms), column_config,
            tempscore_config)

    def _get_projected_production(self, scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        interface from excel file and internally used DataFrame
        :param scope:
        :return:
        """
        return self.benchmark_excel[TabsConfig.PROJECTED_PRODUCTION].set_index(
            [self.column_config.REGION, self.column_config.SECTOR])


class ExcelProviderIntensityBenchmark(BaseProviderIntensityBenchmark):
    def __init__(self, excel_path: str, benchmark_temperature: Quantity['delta_degC'],
                 benchmark_global_budget: Quantity['CO2'], is_AFOLU_included: bool,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._convert_excel_to_model = convert_intensity_benchmark_excel_to_model
        EI_benchmarks = self._convert_excel_to_model(self.benchmark_excel, TabsConfig.PROJECTED_EI,
                                                     column_config.REGION, column_config.SECTOR)
        # TODO: Fix units for Steel
        super().__init__(
            IEIBenchmarkScopes(benchmark_metric={'units':'t CO2/MWh'}, S1S2=EI_benchmarks,
                               benchmark_temperature=benchmark_temperature,
                               benchmark_global_budget=benchmark_global_budget,
                               is_AFOLU_included=is_AFOLU_included),
            column_config,
            tempscore_config)


class ExcelProviderCompany(BaseCompanyDataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param excel_path: A path to the Excel file with the company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
    """

    def __init__(self, excel_path: str, column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        self._companies = self._convert_from_excel_data(excel_path)
        self.historic_years = None
        super().__init__(self._companies, column_config, tempscore_config)

    def _check_company_data(self, company_tabs: dict) -> None:
        """
        Checks if the company data excel contains the data in the right format

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
        """
        Converts the Excel template to list of ICompanyDta objects. All dataprovider features will be inhereted from
        Base
        :param excel_path: file path to excel file
        :return: List of ICompanyData objects
        """
        company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_company_data(company_data)

        df_fundamentals = company_data[TabsConfig.FUNDAMENTAL].set_index(ColumnsConfig.COMPANY_ID, drop=False)
        df_fundamentals[ColumnsConfig.PRODUCTION_METRIC] = df_fundamentals[ColumnsConfig.SECTOR].map(sector_to_production_metric)
        company_ids = list(df_fundamentals[ColumnsConfig.COMPANY_ID].unique())
        df_targets = self._get_projection(company_ids, company_data[TabsConfig.PROJECTED_TARGET], df_fundamentals[ColumnsConfig.PRODUCTION_METRIC])
        if TabsConfig.PROJECTED_EI in company_data:
            df_ei = self._get_projection(company_ids, company_data[TabsConfig.PROJECTED_EI], df_fundamentals[ColumnsConfig.PRODUCTION_METRIC])
        else:
            df_ei = None
        if TabsConfig.HISTORIC_DATA in company_data:
            df_historic = company_data[TabsConfig.HISTORIC_DATA].set_index(ColumnsConfig.COMPANY_ID, drop=False)
            df_historic = df_historic.merge(df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].rename('units'), left_index=True, right_index=True)
            df_historic.loc[df_historic.variable == 'Emissions', 'units'] = 't CO2'
            df_historic.loc[df_historic.variable == 'Emission Intensities', 'units'] = 't CO2/' + df_historic.loc[df_historic.variable == 'Emission Intensities', 'units']
            df_historic = self._get_historic_data(company_ids, df_historic)
        else:
            df_historic = None
        return self._company_df_to_model(df_fundamentals, df_targets, df_ei, df_historic)

    def _convert_series_to_projections(self, projections: pd.Series, ProjectionType: BaseModel) -> [IProjection]:
        """
        Converts a Pandas Series to a list of IProjection
        :param projections: Pandas Series with years as indices
        :return: List of IProjection objects
        """
        return [ProjectionType(year=y, value=v) for y, v in projections.items()]

    def _company_df_to_model(self, df_fundamentals: pd.DataFrame, df_targets: pd.DataFrame, df_ei: pd.DataFrame,
                             df_historic: pd.DataFrame) -> List[ICompanyData]:

        """
        transforms target Dataframe into list of IDataProviderTarget instances

        :param df_fundamentals: pandas Dataframe with fundamental data
        :param df_targets: pandas Dataframe with targets
        :param df_ei: pandas Dataframe with emission intensities
        :return: A list containing the ICompanyData objects
        """
        # set NaN to None since NaN is float instance
        df_fundamentals = df_fundamentals.where(pd.notnull(df_fundamentals), None).replace({np.nan: None})

        companies_data_dict = df_fundamentals.to_dict(orient="records")
        model_companies: List[ICompanyData] = []
        for company_data in companies_data_dict:
            try:
                company_id = company_data[ColumnsConfig.COMPANY_ID]
                production_metric = sector_to_production_metric[company_data[ColumnsConfig.SECTOR]]
                intensity_metric = sector_to_intensity_metric[company_data[ColumnsConfig.SECTOR]]
                company_data[ColumnsConfig.PRODUCTION_METRIC] = {'units': production_metric}
                company_data[ColumnsConfig.EMISSIONS_METRIC] = {'units': 't CO2'}
                # pint automatically handles any unit conversions required

                v = df_fundamentals[df_fundamentals[ColumnsConfig.COMPANY_ID]==company_id][ColumnsConfig.GHG_SCOPE12].squeeze()
                company_data[ColumnsConfig.GHG_SCOPE12] = Q_(v or np.nan, 't CO2')
                company_data[ColumnsConfig.BASE_YEAR_PRODUCTION] = \
                    company_data[ColumnsConfig.GHG_SCOPE12] / df_ei.loc[company_id, :][TemperatureScoreConfig.CONTROLS_CONFIG.base_year]
                v = df_fundamentals[df_fundamentals[ColumnsConfig.COMPANY_ID]==company_id][ColumnsConfig.GHG_SCOPE3].squeeze()
                company_data[ColumnsConfig.GHG_SCOPE3] = Q_(v or np.nan, 't CO2')
                company_data[ColumnsConfig.PROJECTED_TARGETS] = {'S1S2': {
                    'projections': self._convert_series_to_projections (df_targets.loc[company_id, :], ICompanyEIProjection),
                    'ei_metric': {'units': intensity_metric}}}
                company_data[ColumnsConfig.PROJECTED_EI] = {'S1S2': {
                    'projections': self._convert_series_to_projections (df_ei.loc[company_id, :], ICompanyEIProjection),
                    'ei_metric': {'units': intensity_metric}}}

                if df_historic is not None:
                    company_data[TabsConfig.HISTORIC_DATA] = self._convert_historic_data(
                        df_historic.loc[company_data[ColumnsConfig.COMPANY_ID], :]).dict()
                else:
                    company_data[TabsConfig.HISTORIC_DATA] = None

                model_companies.append(ICompanyData.parse_obj(company_data))
            except ValidationError as e:
                logger.warning(
                    f"EX {e}: (one of) the input(s) of company %s is invalid and will be skipped" % company_data[
                        ColumnsConfig.COMPANY_NAME])
        return model_companies
    
    # Workaround for bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero 
    def _np_sum(g):
        return np.sum(g.values)

    def _get_projection(self, company_ids: List[str], projections: pd.DataFrame, production_metric: pd.DataFrame) \
            -> pd.DataFrame:
        """
        get the projected emission intensities for list of companies
        :param company_ids: list of company ids
        :param projections: Dataframe with listed projections per company
        :param production_metric: Dataframe with production_metric per company
        :return: series of projected emission intensities
        """
        projections = projections.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        missing_companies = [company_id for company_id in company_ids if company_id not in projections.index]
        if missing_companies:
            error_message = f"Missing target or trajectory projections for companies with ID: {missing_companies}"
            logger.error(error_message)
            raise ValueError(error_message)

        projections = projections.loc[company_ids, range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                         TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)]
        # Due to bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero workaround below:
        projected_emissions_s1s2 = projections.groupby(level=0, sort=False).agg(ExcelProviderCompany._np_sum)  # add scope 1 and 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            projected_emissions_s1s2 = projected_emissions_s1s2.apply(lambda x: x.astype(f'pint[t CO2/({production_metric[x.name]})]'), axis=1)

        return projected_emissions_s1s2

    def _get_historic_data(self, company_ids: List[str], historic_data: pd.DataFrame) -> pd.DataFrame:
        """
        get the historic data for list of companies
        :param company_ids: list of company ids
        :param historic_data: Dataframe Productions, Emissions, and Emission Intensities mixed together
        :return: historic data with unit attributes added to yearly data on a per-element basis
        """
        self.historic_years = [column for column in historic_data.columns if type(column) == int]

        missing_ids = [company_id for company_id in company_ids if company_id not in historic_data.index]
        if missing_ids:
            error_message = f"Company ids missing in provided historic data: {missing_ids}"
            logger.error(error_message)
            raise ValueError(error_message)

        for year in self.historic_years:
            historic_data[year] = historic_data[year].map(str) + " " + historic_data['units']
        return historic_data.loc[company_ids]

    def _convert_historic_data(self, historic: pd.DataFrame) -> IHistoricData:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :return: IHistoricData Pydantic object
        """
        productions = historic.loc[historic[ColumnsConfig.VARIABLE] == VariablesConfig.PRODUCTIONS]
        emissions = historic.loc[historic[ColumnsConfig.VARIABLE] == VariablesConfig.EMISSIONS]
        emissions_intensities = historic.loc[historic[ColumnsConfig.VARIABLE] == VariablesConfig.EMISSIONS_INTENSITIES]
        return IHistoricData(
            productions=self._convert_to_historic_productions(productions),
            emissions=self._convert_to_historic_emissions(emissions),
            emissionss_intensities=self._convert_to_historic_ei(emissions_intensities)
        )

    # Note that for the three following functions, we pd.Series.squeeze() the results because it's just one year / one company
    def _convert_to_historic_emissions(self, emissions: pd.DataFrame) -> Optional[IHistoricEmissionsScopes]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :param convert_unit: whether or not to convert the units of measure
        :return: List of historic emissions per scope, or None if no data are provided
        """
        if emissions.empty:
            return None

        emissions_scopes = {}
        for scope in EScope.get_scopes():
            results = emissions.loc[emissions[ColumnsConfig.SCOPE] == scope]
            emissions_scopes[scope] = [] \
                if results.empty \
                else [IEmissionRealization(year=year, value=Q_(*results[year].squeeze().split(' ', 1))) for year in self.historic_years]
        return IHistoricEmissionsScopes(**emissions_scopes)

    def _convert_to_historic_productions(self, productions: pd.DataFrame) \
            -> Optional[List[IProductionRealization]]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :return: A list containing historic productions, or None if no data are provided
        """
        if productions.empty:
            return None

        production_realizations = \
            [IProductionRealization(year=year, value=Q_(*productions[year].squeeze().split(' ', 1))) for year in self.historic_years]
        return production_realizations

    def _convert_to_historic_ei(self, intensities: pd.DataFrame) \
            -> Optional[IHistoricEIScopes]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :return: A list of historic emission intensities per scope, or None if no data are provided
        """
        if intensities.empty:
            return None

        intensities = intensities.copy()
        intensity_scopes = {}

        for scope in EScope.get_scopes():
            results = intensities.loc[intensities[ColumnsConfig.SCOPE] == scope]
            intensity_scopes[scope] = [] \
                if results.empty \
                else [IEIRealization(year=year, value=Q_(*results[year].squeeze().split(' ', 1))) for year in self.historic_years]
        return IHistoricEIScopes(**intensity_scopes)
