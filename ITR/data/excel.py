from typing import Type, List
import pandas as pd
import numpy as np
from pydantic import ValidationError
from ITR.data.data_providers import ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.data.base_providers import BaseCompanyDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, SectorsConfig
from ITR.interfaces import ICompanyData, ICompanyProjection
import logging


class TabsConfig:
    FUNDAMENTAL = "fundamental_data"
    PROJECTED_EI = "projected_ei_in_Wh"
    PROJECTED_PRODUCTION = "projected_production"
    PROJECTED_TARGET = "projected_target"


class ExcelProviderProductionBenchmark(ProductionBenchmarkDataProvider):
    def __init__(self, excel_path: str, column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__()
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self.temp_config = tempscore_config
        self.column_config = column_config

    def _check_sector_data(self) -> None:
        """
        Checks if the sector data excel contains the data in the right format

        :return: None
        """
        assert pd.Series([TabsConfig.PROJECTED_PRODUCTION, TabsConfig.PROJECTED_EI]).isin(
            self.benchmark_excel.keys()).all(), "some tabs are missing in the sector data excel"

    def get_company_projected_production(self, ghg_scope12: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for list of companies in ghg_scope12
        :param ghg_scope12: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID,ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: DataFrame of projected productions for [base_year - base_year + 50]
        """
        benchmark_production_projections = self.get_benchmark_projections(ghg_scope12)
        return benchmark_production_projections.add(1).cumprod(axis=1).mul(
            ghg_scope12[self.column_config.GHG_SCOPE12].values, axis=0)

    def get_benchmark_projections(self, company_sector_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with production benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        benchmark_projection = self.benchmark_excel[TabsConfig.PROJECTED_PRODUCTION]
        sectors = company_sector_region_info[self.column_config.SECTOR]
        regions = company_sector_region_info[self.column_config.REGION]
        benchmark_regions = regions.copy()
        mask = benchmark_regions.isin(benchmark_projection[self.column_config.REGION])
        benchmark_regions.loc[~mask] = "Global"
        benchmark_projection = benchmark_projection.reset_index().set_index(
            [self.column_config.SECTOR, self.column_config.REGION])

        benchmark_projection = benchmark_projection.loc[list(zip(sectors, benchmark_regions)),
                                                        range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                              self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index

        return benchmark_projection


class ExcelProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(self, excel_path: str, benchmark_temperature: float,
                 benchmark_global_budget: float, is_AFOLU_included: bool,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__(benchmark_temperature, benchmark_global_budget, is_AFOLU_included)
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self.temp_config = tempscore_config
        self.column_config = column_config

    def get_SDA_intensity_benchmarks(self, company_info_at_base_year: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_info_at_base_year: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.BASE_EI ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        intensity_benchmarks = self._get_intensity_benchmarks(company_info_at_base_year)
        decarbonization_paths = self._get_decarbonizations_paths(intensity_benchmarks)
        last_ei = intensity_benchmarks[self.temp_config.CONTROLS_CONFIG.target_end_year]
        ei_base = company_info_at_base_year[self.column_config.BASE_EI]

        return decarbonization_paths.mul((ei_base - last_ei), axis=0).add(last_ei, axis=0)

    def _get_decarbonizations_paths(self, intensity_benchmarks: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        Returns a DataFrame with the projected decarbonization paths for the supplied companies in intensity_benchmarks.
        :param: A DataFrame with company and intensity benchmarks per calendar year per row
        :return: A pd.DataFrame with company and decarbonisation path s per calendar year per row
        """
        return intensity_benchmarks.apply(lambda row: self._get_decarbonization(row), axis=1)

    def _get_decarbonization(self, intensity_benchmark_row: pd.Series) -> pd.Series:
        """
        Overrides subclass method
        returns a Series with the decarbonization path for a benchmark.
        :param: A Series with company and intensity benchmarks per calendar year per row
        :return: A pd.Series with company and decarbonisation path s per calendar year per row
        """
        first_ei = intensity_benchmark_row[self.temp_config.CONTROLS_CONFIG.base_year]
        last_ei = intensity_benchmark_row[self.temp_config.CONTROLS_CONFIG.target_end_year]
        return intensity_benchmark_row.apply(lambda x: (x - last_ei) / (first_ei - last_ei))

    def _get_intensity_benchmarks(self, company_sector_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        benchmark_projection = self.benchmark_excel[TabsConfig.PROJECTED_EI]
        sectors = company_sector_region_info[self.column_config.SECTOR]
        regions = company_sector_region_info[self.column_config.REGION]
        benchmark_regions = regions.copy()
        mask = benchmark_regions.isin(benchmark_projection[self.column_config.REGION])
        benchmark_regions.loc[~mask] = "Global"
        benchmark_projection = benchmark_projection.reset_index().set_index(
            [self.column_config.SECTOR, self.column_config.REGION])

        benchmark_projection = benchmark_projection.loc[list(zip(sectors, benchmark_regions)),
                                                        range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                              self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index

        return benchmark_projection

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
        self.ENERGY_UNIT_CONVERSION_FACTOR = 3.6
        self.CORRECTION_SECTORS = [SectorsConfig.ELECTRICITY]
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

        df_fundamentals = df_company_data[TabsConfig.FUNDAMENTAL]
        company_ids = df_fundamentals[self.column_config.COMPANY_ID].unique()
        df_targets = self._get_projection(company_ids, df_company_data[TabsConfig.PROJECTED_TARGET])
        df_ei = self._get_projection(company_ids, df_company_data[TabsConfig.PROJECTED_EI])
        return self._company_df_to_model(df_fundamentals, df_targets, df_ei)

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

    def _company_df_to_model(self, df_fundamentals: pd.DataFrame, df_targets: pd.DataFrame, df_ei: pd.DataFrame) -> \
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
            try:
                convert_unit_of_measure = company_data[self.column_config.SECTOR] in self.CORRECTION_SECTORS
                company_targets = self._convert_series_to_projections(
                    df_targets.loc[company_data[self.column_config.COMPANY_ID], :], convert_unit_of_measure)
                company_ei = self._convert_series_to_projections(
                    df_ei.loc[company_data[self.column_config.COMPANY_ID], :],
                    convert_unit_of_measure)

                company_data.update({self.column_config.PROJECTED_TARGETS: {'S1S2': {'projections': company_targets}}})
                company_data.update({self.column_config.PROJECTED_EI: {'S1S2': {'projections': company_ei}}})

                model_companies.append(ICompanyData.parse_obj(company_data))

            except ValidationError as e:
                logger.warning(
                    "(one of) the input(s) of company %s is invalid and will be skipped" % company_data[
                        self.column_config.COMPANY_NAME])
                pass
        return model_companies

    def _get_projection(self, company_ids: List[str], projections: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected emissions for list of companies
        :param company_ids: list of company ids
        :param projections: Dataframe with listed projections per company
        :return: series of projected emissions
        """

        projections = projections.reset_index().set_index(self.column_config.COMPANY_ID)

        assert all(company_id in projections.index for company_id in company_ids), \
            f"company ids missing in provided projections"

        projections = projections.loc[company_ids, :]
        projections = projections.loc[:, range(self.temp_config.CONTROLS_CONFIG.base_year,
                                               self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]

        # Due to bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero workaround below:
        projections = projections.fillna(np.inf)
        projected_emissions_s1s2 = projections.groupby(level=0, sort=False).sum()  # add scope 1 and 2
        projected_emissions_s1s2 = projected_emissions_s1s2.replace(np.inf, np.nan)

        return projected_emissions_s1s2
