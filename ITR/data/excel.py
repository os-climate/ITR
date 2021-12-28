from typing import Type, List, Union, Optional
import pandas as pd
import numpy as np

from pint import Quantity
import pint
import pint_pandas
ureg = pint.get_application_registry()
Q_ = ureg.Quantity

from pydantic import ValidationError
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, SectorsConfig, VariablesConfig, TabsConfig
from ITR.interfaces import ICompanyData, ICompanyProjection, EScope, IEmissionIntensityBenchmarkScopes, \
    IProductionBenchmarkScopes, IBenchmark, IBenchmarks, IBenchmarkProjection, IHistoricEmissionsScopes, \
    IProductionRealization, IHistoricEIScopes, IHistoricData, IEmissionRealization, IEIRealization

import logging

from ITR.interfaces import ICompanyProjections
import inspect

# TODO: Force validation for excel benchmarks

# Utils functions:

def convert_benchmark_excel_to_model(df_excel: pd.DataFrame, sheetname: str, column_name_region: str,
                                     column_name_sector: str, cell_unit: str) -> IBenchmarks:
    """
    Converts excel into IBenchmarks
    :param excal_path: file path to excel
    :return: IBenchmarks instance (list of IBenchmark)
    """
    df_ei_bms = df_excel[sheetname].reset_index().drop(columns=['index']).set_index(
        [column_name_region, column_name_sector])
    result = []
    for index, row in df_ei_bms.iterrows():
        bm = IBenchmark(region=index[0], sector=index[1],
                        projections=[IBenchmarkProjection(year=int(k), value=Q_(v, cell_unit)) for k, v in row.items()])
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
        self._check_sector_data()
        self._convert_excel_to_model = convert_benchmark_excel_to_model
        production_bms = self._convert_excel_to_model(self.benchmark_excel, TabsConfig.PROJECTED_PRODUCTION,
                                                      column_config.REGION, column_config.SECTOR, 'Mt CO2')
        super().__init__(
            IProductionBenchmarkScopes(S1S2=production_bms), column_config,
            tempscore_config)

    def _check_sector_data(self) -> None:
        """
        Checks if the sector data excel contains the data in the right format

        :return: None
        """
        assert pd.Series([TabsConfig.PROJECTED_PRODUCTION, TabsConfig.PROJECTED_EI]).isin(
            self.benchmark_excel.keys()).all(), "some tabs are missing in the sector data excel"

    def _get_projected_production(self, scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        interface from excel file and internally used DataFrame
        :param scope:
        :return:
        """
        return self.benchmark_excel[TabsConfig.PROJECTED_PRODUCTION].reset_index().set_index(
            [self.column_config.REGION, self.column_config.SECTOR])


class ExcelProviderIntensityBenchmark(BaseProviderIntensityBenchmark):
    def __init__(self, excel_path: str, benchmark_temperature: Quantity['degC'],
                 benchmark_global_budget: Quantity['CO2'], is_AFOLU_included: bool,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self._convert_excel_to_model = convert_benchmark_excel_to_model
        EI_benchmarks = self._convert_excel_to_model(self.benchmark_excel, TabsConfig.PROJECTED_EI,
                                                     column_config.REGION, column_config.SECTOR, 't CO2/MWh')
        super().__init__(
            IEmissionIntensityBenchmarkScopes(S1S2=EI_benchmarks, benchmark_temperature=benchmark_temperature,
                                              benchmark_global_budget=benchmark_global_budget,
                                              is_AFOLU_included=is_AFOLU_included), column_config,
            tempscore_config)

    def _check_sector_data(self) -> None:
        """
        Checks if the sector data excel contains the data in the right format
        :return: None
        """
        assert pd.Series([TabsConfig.PROJECTED_PRODUCTION, TabsConfig.PROJECTED_EI]).isin(
            self.benchmark_excel.keys()).all(), "some tabs are missing in the sector data excel"


class ExcelProviderCompany(BaseCompanyDataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param excel_path: A path to the Excel file with the company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
    """

    def __init__(self, excel_path: str, column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__(None, column_config, tempscore_config)
        self._companies = self._convert_excel_data_to_ICompanyData(excel_path)

    def _check_company_data(self, df: pd.DataFrame) -> None:
        """
        Checks if the company data excel contains the data in the right format

        :return: None
        """
        required_tabs = [TabsConfig.FUNDAMENTAL, TabsConfig.PROJECTED_TARGET]
        optional_tabs = [TabsConfig.PROJECTED_EI, TabsConfig.HISTORIC_DATA]
        missing_tabs = [tab for tab in required_tabs + optional_tabs if tab not in df.keys()]
        assert not any(tab in missing_tabs for tab in required_tabs), f"Tabs {required_tabs} are required."
        assert not all(tab in missing_tabs for tab in optional_tabs), f"Either of the tabs {optional_tabs} is required."

    def _convert_from_excel_data(self, excel_path: str) -> List[ICompanyData]:
        """
        Converts the Excel template to list of ICompanyDta objects. All dataprovider features will be inhereted from
        Base
        :param excel_path: file path to excel file
        :return: List of ICompanyData objects
        """
        df_company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_company_data(df_company_data)

        df_fundamentals = df_company_data[TabsConfig.FUNDAMENTAL]
        company_ids = df_fundamentals[self.column_config.COMPANY_ID].unique()
        df_targets = self._get_projection(company_ids, df_company_data[TabsConfig.PROJECTED_TARGET], 'pint[Mt CO2]')
        if TabsConfig.PROJECTED_EI in df_company_data.keys():
            df_ei = self._get_projection(company_ids, df_company_data[TabsConfig.PROJECTED_EI], 'pint[t CO2/MWh]')
        else:
            df_ei = None
        if TabsConfig.HISTORIC_DATA in df_company_data.keys():
            df_historic = self._get_historic_data(company_ids, df_company_data[TabsConfig.HISTORIC_DATA], 'pint[t CO2/MWh]'
        else:
            df_historic = None
        return self._company_df_to_model(df_fundamentals, df_targets, df_ei, df_historic)

    def _convert_series_to_projections(self, projections: pd.Series, convert_unit: bool = False) -> List[
        ICompanyProjection]:
        """
        Converts a Pandas Series in a list of ICompanyProjections
        :param projections: Pandas Series with years as indices
        :param convert_unit: Boolean if series values needs conversion for unit of measure
        :return: List of ICompanyProjection objects
        """
        projections = projections * self.ENERGY_UNIT_CONVERSION_FACTOR if convert_unit else projections
        return [ICompanyProjection(year=y, value=v) for y, v in projections.items()]

    def _company_df_to_model(self, df_fundamentals: pd.DataFrame, df_targets: pd.DataFrame, df_ei: pd.DataFrame,
                             df_historic: pd.DataFrame) -> List[ICompanyData]:

        """
        transforms target Dataframe into list of IDataProviderTarget instances

        :param df_fundamentals: pandas Dataframe with fundamental data
        :param df_targets: pandas Dataframe with targets
        :param df_ei: pandas Dataframe with emission intensities
        :return: A list containing the ICompanyData objects
        """
        logger = logging.getLogger(__name__)
        # set NaN to None since NaN is float instance
        df_fundamentals = df_fundamentals.where(pd.notnull(df_fundamentals), None).replace({np.nan: None})

        companies_data_dict = df_fundamentals.to_dict(orient="records")
        model_companies: List[ICompanyData] = []
        for company_data in companies_data_dict:
            # company_data is a dict, not a dataframe
            try:
                # convert_unit_of_measure = company_data[ColumnsConfig.SECTOR] in self.CORRECTION_SECTORS
                # company_targets = self._convert_series_to_projections(
                #     df_targets.loc[company_data[ColumnsConfig.COMPANY_ID], :], convert_unit_of_measure)
                # company_ei = self._convert_series_to_projections(
                #     df_ei.loc[company_data[ColumnsConfig.COMPANY_ID], :],
                #     convert_unit_of_measure)

                # company_data.update({ColumnsConfig.PROJECTED_TARGETS: {'S1S2': {'projections': df_targets}}})
                # company_data.update({ColumnsConfig.PROJECTED_EI: {'S1S2': {'projections': df_ei}}})

                company_id = company_data[self.column_config.COMPANY_ID]
                # pint automatically handles any unit conversions required
                ghg_s1s2 = df_fundamentals[df_fundamentals[self.column_config.COMPANY_ID]==company_id][self.column_config.GHG_SCOPE12].squeeze()
                if ghg_s1s2 is None:
                    ghg_s1s2 = 1
                company_data[self.column_config.GHG_SCOPE12] = Q_(ghg_s1s2, ureg('t CO2'))
                ghg_s3 = df_fundamentals[df_fundamentals[self.column_config.COMPANY_ID]==company_id][self.column_config.GHG_SCOPE3].squeeze()
                if ghg_s3 is None:
                    ghg_s3 = 1
                company_data[self.column_config.GHG_SCOPE3] = Q_(ghg_s3, ureg('t CO2'))
                company_data[self.column_config.PROJECTED_TARGETS] = {'S1S2': {'projections': self._convert_series_to_projections (df_targets.loc[company_id, :])}}
                company_data[self.column_config.PROJECTED_EI] = {'S1S2': {'projections': self._convert_series_to_projections (df_ei.loc[company_id, :])}}

                if df_historic is not None:
                    company_data[TabsConfig.HISTORIC_DATA] = df_historic.loc[company_data[ColumnsConfig.COMPANY_ID], :]

                # The call to parse_obj essentially says "I put it all together manually, please validate that it's correct",
                # as opposed to using constructors to build the object validly in the first place.
                model_companies.append(ICompanyData.parse_obj(company_data))
            except ValidationError as e:
                print(__name__, e)
                logger.warning(
                    "(one of) the input(s) of company %s is invalid and will be skipped" % company_data[
                        ColumnsConfig.COMPANY_NAME])
                pass
        return model_companies
    
    # Workaround for bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero 
    def _np_sum(g):
        return np.sum(g.values)

    def _get_projection(self, company_ids: List[str], projections: pd.DataFrame, astype: str) -> pd.DataFrame:
        """
        get the projected emissions for list of companies
        :param company_ids: list of company ids
        :param projections: Dataframe with listed projections per company
        :return: series of projected emissions
        """

        projections = projections.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        assert all(company_id in projections.index for company_id in company_ids), \
            f"company ids missing in provided projections"

        projections = projections.loc[company_ids, :]
        projections = projections.loc[:, range(self.temp_config.CONTROLS_CONFIG.base_year,
                                               self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        # Due to bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero workaround below:
        projected_emissions_s1s2 = projections.groupby(level=0, sort=False).agg(ExcelProviderCompany._np_sum)  # add scope 1 and 2
        # print("about to convert in _get_projection")
        for col in projected_emissions_s1s2.columns:
            projected_emissions_s1s2[col] = projected_emissions_s1s2[col].astype(astype)
        # print(f"projected_emissions_s1s2.loc[{astype}] = {projected_emissions_s1s2.iloc[0:7, 0:7]}")

        return projected_emissions_s1s2

    def _get_historic_data(self, company_ids: List[str], historic_data: pd.DataFrame) -> pd.DataFrame:
        historic_data = historic_data.reset_index().drop(columns=['index']).set_index(ColumnsConfig.COMPANY_ID)
        self.historic_years = [column for column in historic_data.columns if type(column) == int]

        missing_ids = [company_id for company_id in company_ids if company_id not in historic_data.index]
        assert not missing_ids, f"Company ids missing in provided historic data: {missing_ids}"

        return historic_data.loc[company_ids, :]

    def _convert_historic_data(self, historic: pd.DataFrame, convert_unit: bool) -> IHistoricData:
        productions = historic.loc[historic[ColumnsConfig.VARIABLE] == VariablesConfig.PRODUCTIONS]
        emissions = historic.loc[historic[ColumnsConfig.VARIABLE] == VariablesConfig.EMISSIONS]
        emission_intensities = historic.loc[historic[ColumnsConfig.VARIABLE] == VariablesConfig.EMISSION_INTENSITIES]
        return IHistoricData(
            productions=self._convert_to_historic_productions(productions, convert_unit),
            emissions=self._convert_to_historic_emissions(emissions),
            emission_intensities=self._convert_to_historic_emission_intensities(emission_intensities, convert_unit)
        )

    def _convert_to_historic_emissions(self, emissions: pd.DataFrame) -> Optional[IHistoricEmissionsScopes]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :param convert_unit: whether or not to convert the units of measure
        :return: List of historic emissions per scope, or None if no data are provided
        """
        if emissions.empty:
            return None

        emission_scopes = {}
        for scope in EScope.get_scopes():
            results = emissions.loc[emissions[ColumnsConfig.SCOPE] == scope]
            emission_scopes[scope] = [] \
                if results.empty \
                else [IEmissionRealization(year=year, value=results[year]) for year in self.historic_years]
        return IHistoricEmissionsScopes(**emission_scopes)

    def _convert_to_historic_productions(self, productions: pd.DataFrame, convert_unit: bool) \
            -> Optional[List[IProductionRealization]]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :param convert_unit: whether or not to convert the units of measure
        :return: A list containing historic productions, or None if no data are provided
        """
        if productions.empty:
            return None

        if convert_unit:
            converted = productions[self.historic_years] * self.ENERGY_UNIT_CONVERSION_FACTOR
            production_realizations = \
                [IProductionRealization(year=year, value=converted[year]) for year in self.historic_years]
        else:
            production_realizations = \
                [IProductionRealization(year=year, value=productions[year]) for year in self.historic_years]
        return production_realizations

    def _convert_to_historic_emission_intensities(self, intensities: pd.DataFrame, convert_unit: bool) \
            -> Optional[IHistoricEIScopes]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :param convert_unit: whether or not to convert the units of measure
        :return: A list of historic emission intensities per scope, or None if no data are provided
        """
        if intensities.empty:
            return None

        intensities = intensities.copy()
        if convert_unit:
            intensities[self.historic_years] *= self.ENERGY_UNIT_CONVERSION_FACTOR

        intensity_scopes = {}
        for scope in EScope.get_scopes():
            results = intensities.loc[intensities[ColumnsConfig.SCOPE] == scope]
            intensity_scopes[scope] = [] \
                if results.empty \
                else [IEIRealization(year=year, value=results[year]) for year in self.historic_years]
        return IHistoricEIScopes(**intensity_scopes)
