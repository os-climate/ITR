from typing import Type, List
import pandas as pd
import numpy as np

from pint import Quantity
# from pint_pandas import PintArray

import pint
import pint_pandas
ureg = pint.get_application_registry()
Q_ = ureg.Quantity
# PA_ = pint_pandas.PintArray

from pydantic import ValidationError
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, SectorsConfig
from ITR.interfaces import ICompanyData, ICompanyProjection, \
    EScope, IEmissionIntensityBenchmarkScopes, \
    IProductionBenchmarkScopes, IBenchmark, IBenchmarks, IBenchmarkProjection
import logging

from ITR.interfaces import ICompanyProjections, ICompanyProjections
import inspect

# Excel spreadsheets don't have units elaborated, so we translate sectors to units
sector_to_production_metric = { 'Electricity Utilities':'MWh', 'Steel':'Fe_ton' }

# TODO: Force validation for excel benchmarks

# Utils functions:

def convert_dimensionless_benchmark_excel_to_model(df_excel: pd.DataFrame, sheetname: str, column_name_region: str,
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
        bm = IBenchmark(region=index[0], sector=index[1], benchmark_metric={'units':'dimensionless'},
                        projections=[IBenchmarkProjection(year=int(k), value=v) for k, v in row.items()])
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
        intensity_units = f't CO2/({sector_to_production_metric[index[1]]})'
        bm = IBenchmark(region=index[0], sector=index[1], benchmark_metric={'units':intensity_units},
                        projections=[IBenchmarkProjection(year=int(k), value=v) for k, v in row.items()])
        result.append(bm)
    return IBenchmarks(benchmarks=result)


class TabsConfig:
    FUNDAMENTAL = "fundamental_data"
    PROJECTED_EI = "projected_ei_in_Wh"
    PROJECTED_PRODUCTION = "projected_production"
    PROJECTED_TARGET = "projected_target"


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
        self._convert_excel_to_model = convert_dimensionless_benchmark_excel_to_model
        production_bms = self._convert_excel_to_model(self.benchmark_excel, TabsConfig.PROJECTED_PRODUCTION,
                                                      column_config.REGION, column_config.SECTOR)
        super().__init__(
            IProductionBenchmarkScopes(benchmark_metric={'units':'dimensionless'}, S1S2=production_bms), column_config,
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
    def __init__(self, excel_path: str, benchmark_temperature: Quantity['delta_degC'],
                 benchmark_global_budget: Quantity['CO2'], is_AFOLU_included: bool,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self._convert_excel_to_model = convert_intensity_benchmark_excel_to_model
        EI_benchmarks = self._convert_excel_to_model(self.benchmark_excel, TabsConfig.PROJECTED_EI,
                                                     column_config.REGION, column_config.SECTOR)
        super().__init__(
            IEmissionIntensityBenchmarkScopes(benchmark_metric={'units':'t CO2/MWh'}, S1S2=EI_benchmarks,
                                              benchmark_temperature=benchmark_temperature,
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
        assert pd.Series([TabsConfig.FUNDAMENTAL, TabsConfig.PROJECTED_TARGET, TabsConfig.PROJECTED_EI]).isin(
            df.keys()).all(), "some tabs are missing in the company data excel"

    def _convert_excel_data_to_ICompanyData(self, excel_path: str) -> List[ICompanyData]:
        """
        Converts the Excel template to list of ICompanyDta objects. All dataprovider features will be inhereted from
        Base
        :param excel_path: file path to excel file
        :return: List of ICompanyData objects
        """
        df_company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_company_data(df_company_data)

        df_fundamentals = df_company_data[TabsConfig.FUNDAMENTAL].set_index(self.column_config.COMPANY_ID, drop=False)
        company_ids = df_fundamentals[self.column_config.COMPANY_ID].unique()
        df_fundamentals[self.column_config.PRODUCTION_METRIC] = df_fundamentals[self.column_config.SECTOR].map(sector_to_production_metric)
        df_targets = self._get_projection(company_ids, df_company_data[TabsConfig.PROJECTED_TARGET], df_fundamentals[self.column_config.PRODUCTION_METRIC])
        df_ei = self._get_projection(company_ids, df_company_data[TabsConfig.PROJECTED_EI], df_fundamentals[self.column_config.PRODUCTION_METRIC])
        return self._company_df_to_model(df_fundamentals, df_targets, df_ei)

    def _convert_series_to_ICompanyProjections(self, projections: pd.Series) -> List[
        ICompanyProjection]:
        """
        Converts a Pandas Series in a list of ICompanyProjections
        :param projections: Pandas Series with years as indices
        :return: List of ICompanyEIProjection objects
        """
        return [ICompanyProjection(year=y, value=v) for y, v in projections.items()]

    def _company_df_to_model(self, df_fundamentals: pd.DataFrame, df_targets: pd.DataFrame, df_trajectories: pd.DataFrame) -> \
            List[ICompanyData]:
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
                company_id = company_data[self.column_config.COMPANY_ID]
                units = sector_to_production_metric[company_data[self.column_config.SECTOR]]
                company_data[self.column_config.PRODUCTION_METRIC] = {'units': units}
                # pint automatically handles any unit conversions required
                v = df_fundamentals[df_fundamentals[self.column_config.COMPANY_ID]==company_id][self.column_config.GHG_SCOPE12].squeeze()
                if v:
                    company_data[self.column_config.GHG_SCOPE12] = {'year': 2019, 'value': v}
                v = df_fundamentals[df_fundamentals[self.column_config.COMPANY_ID]==company_id][self.column_config.GHG_SCOPE3].squeeze()
                if v:
                    company_data[self.column_config.GHG_SCOPE3] = {'year': 2019, 'value': v}
                company_data[self.column_config.PROJECTED_TARGETS] = {'S1S2': {'units': units,
                                                                               'projections': self._convert_series_to_ICompanyProjections (df_targets.loc[company_id, :].apply(lambda x: x.m))}}
                company_data[self.column_config.PROJECTED_TRAJECTORIES] = {'S1S2': {'units': units,
                                                                                    'projections': self._convert_series_to_ICompanyProjections (df_trajectories.loc[company_id, :].apply(lambda x: x.m))}}
                # The call to parse_obj essentially says "I put it all together manually, please validate that it's correct",
                # as opposed to using constructors to build the object validly in the first place.
                model_companies.append(ICompanyData.parse_obj(company_data))
            except ValidationError as e:
                logger.warning(
                    f"EX {e}: (one of) the input(s) of company %s is invalid and will be skipped" % company_data[
                        self.column_config.COMPANY_NAME])
                pass
        return model_companies
    
    # Workaround for bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero 
    def _np_sum(g):
        return np.sum(g.values)

    def _get_projection(self, company_ids: List[str], projections: pd.DataFrame, production_metric: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected emission intensities for list of companies
        :param company_ids: list of company ids
        :param projections: Dataframe with listed projections per company
        :param production_metric: Dataframe with production_metric per company
        :return: series of projected emission intensities
        """

        projections = projections.reset_index().set_index(self.column_config.COMPANY_ID)

        assert all(company_id in projections.index for company_id in company_ids), \
            f"company ids missing in provided projections"

        projections = projections.loc[company_ids, range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                         self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        # Due to bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero workaround below:
        projected_ei_s1s2 = projections.groupby(level=0, sort=False).agg(ExcelProviderCompany._np_sum)  # add scope 1 and 2
        projected_ei_s1s2 = projected_ei_s1s2.apply(lambda x: x.astype(f'pint[t CO2/({production_metric[x.name]})]'), axis=1)
        return projected_ei_s1s2
