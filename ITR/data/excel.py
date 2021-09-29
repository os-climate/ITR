from typing import Type, List
from pydantic import ValidationError
import logging

import pandas as pd
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, IntensityBenchmarkDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, SectorsConfig
from ITR.interfaces import ICompanyData


class TabsConfig:
    FUNDAMENTAL = "fundamental_data"
    PROJECTED_EI = "projected_ei_in_Wh"
    PROJECTED_PRODUCTION = "projected_production"
    PROJECTED_TARGET = "projected_target"

class ExcelProviderProductionBenchmark(ProductionBenchmarkDataProvider):
    def __init__(self, excel_path: str, config: Type[ColumnsConfig] = ColumnsConfig):
        super().__init__()
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self.c = config  # TODO polish config

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
            ghg_scope12[ColumnsConfig.GHG_SCOPE12].values, axis=0)

    def get_benchmark_projections(self, company_secor_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        return self.get_benchmark_projections_from_sheet(company_secor_region_info, TabsConfig.PROJECTED_PRODUCTION)

    def get_benchmark_projections_from_sheet(self, company_secor_region_info: pd.DataFrame,
                                             feature: str) -> pd.DataFrame:
        """
        get the sector emissions for companies in the subset_company_data frame
        If there is no data for the sector, then it will be replaced by the global value
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :param feature: name of the projected feature
        :return: series of projected emissions for the sector
        """
        benchmark_projection = self.benchmark_excel[feature]
        sectors = company_secor_region_info[ColumnsConfig.SECTOR]
        regions = company_secor_region_info[ColumnsConfig.REGION]
        regions.loc[~regions.isin(benchmark_projection[ColumnsConfig.REGION])] = "Global"
        benchmark_projection = benchmark_projection.reset_index().set_index(
            [ColumnsConfig.SECTOR, ColumnsConfig.REGION])

        benchmark_projection = benchmark_projection.loc[list(zip(sectors, regions)),
                                                        range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                              TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index

        return benchmark_projection


class ExcelProviderIntensistyBenchmark(IntensityBenchmarkDataProvider):
    def __init__(self, excel_path: str, benchmark_temperature: float,
                 benchmark_global_budget: float, AFOLU_included: bool, config: Type[ColumnsConfig] = ColumnsConfig):
        super().__init__(benchmark_temperature, benchmark_global_budget, AFOLU_included)
        self.benchmark_excel = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_sector_data()
        self.c = config

    def get_SDA_intensity_benchmarks(self, company_info_at_base_year: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_info_at_base_year: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        intensity_benchmarks = self._get_intensity_benchmarks(company_info_at_base_year)
        decarbonization_paths = self._get_decarbonizations_paths(intensity_benchmarks)
        last_ei = intensity_benchmarks[TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year]
        ei_base = company_info_at_base_year[ColumnsConfig.BASE_EI]

        return decarbonization_paths.mul((ei_base - last_ei), axis=0).add(last_ei, axis=0)

    def _get_decarbonizations_paths(self, intensity_benchmarks: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A pd.DataFrame with company and decarbonisation path s per calendar year per row
        """
        return intensity_benchmarks.apply(lambda row: self._get_decarbonization(row), axis=1)

    def _get_decarbonization(self, intensity_benchmark_row: pd.Series) -> pd.Series:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A pd.Series with company and decarbonisation path s per calendar year per row
        """
        first_ei = intensity_benchmark_row[TemperatureScoreConfig.CONTROLS_CONFIG.base_year]
        last_ei = intensity_benchmark_row[TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year]
        return intensity_benchmark_row.apply(lambda x: (x - last_ei) / (first_ei - last_ei))

    def _get_intensity_benchmarks(self, company_secor_region_info: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        return self.get_benchmark_projections(company_secor_region_info, TabsConfig.PROJECTED_EI)

    def _check_sector_data(self) -> None:
        """
        Checks if the sector data excel contains the data in the right format
        :return: None
        """
        assert pd.Series([TabsConfig.PROJECTED_PRODUCTION, TabsConfig.PROJECTED_EI]).isin(
            self.benchmark_excel.keys()).all(), "some tabs are missing in the sector data excel"

    def get_benchmark_projections(self, company_secor_region_info: pd.DataFrame, feature: str) -> pd.DataFrame:
        """

        :param company_secor_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :param feature: name of the projected feature corresponding to the sheet name in the excel
        :return: series of projected emissions for the sector
        """
        benchmark_projection = self.benchmark_excel[feature]
        sectors = company_secor_region_info[ColumnsConfig.SECTOR]
        regions = company_secor_region_info[ColumnsConfig.REGION]
        regions.loc[~regions.isin(benchmark_projection[ColumnsConfig.REGION])] = "Global"
        benchmark_projection = benchmark_projection.reset_index().set_index(
            [ColumnsConfig.SECTOR, ColumnsConfig.REGION])

        benchmark_projection = benchmark_projection.loc[list(zip(sectors, regions)),
                                                        range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                              TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index

        return benchmark_projection


class ExcelProviderCompany(CompanyDataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!

    :param config: A dictionary containing a "path" field that leads to the path of the CSV file
    """

    def __init__(self, excel_path: str, config: Type[ColumnsConfig] = ColumnsConfig):
        super().__init__()
        self.company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_company_data()
        self.c = config
        self.ENERGY_UNIT_CONVERSION_FACTOR = 3.6

    def _check_company_data(self) -> None:
        """
        Checks if the company data excel contains the data in the right format

        :return: None
        """
        assert pd.Series([TabsConfig.FUNDAMENTAL, TabsConfig.PROJECTED_TARGET, TabsConfig.PROJECTED_EI]).isin(
            self.company_data.keys()).all(), "some tabs are missing in the company data excel"

    def get_company_data(self, company_ids: List[str]) -> List[ICompanyData]:
        """
        Get all relevant data for a list of company ids (ISIN). This method should return a list of IDataProviderCompany
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        data_company = self.company_data[TabsConfig.FUNDAMENTAL]

        assert pd.Series(company_ids).isin(data_company.loc[:, ColumnsConfig.COMPANY_ID]).all(), \
            "some of the company ids are not included in the fundamental data"

        data_company = data_company.loc[data_company.loc[:, ColumnsConfig.COMPANY_ID].isin(company_ids), :]
        companies = data_company.to_dict(orient="records")
        model_companies: List[ICompanyData] = [ICompanyData.parse_obj(company) for company in companies]

        return model_companies

    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        get the value of a variable of a list of companies
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        company_data = self.company_data[TabsConfig.FUNDAMENTAL]
        company_data = company_data.reset_index().set_index(ColumnsConfig.COMPANY_ID)
        return company_data.loc[company_ids, variable_name]

    def _get_projection(self, company_ids: List[str], feature: str) -> pd.DataFrame:
        """
        get the projected emissions for list of companies
        :param company_ids: list of company ids
        :param feature: name of the projected feature
        :return: series of projected emissions
        """
        projected_emissions = self.company_data[feature]
        projected_emissions = projected_emissions.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        assert all(company_id in projected_emissions.index for company_id in company_ids), \
            f"company ids missing in {feature}"

        projected_emissions = projected_emissions.loc[company_ids, :]
        projected_emissions = self._unit_of_measure_correction(company_ids, projected_emissions)

        projected_emissions = projected_emissions.loc[:, range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                               TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)]
        projected_emissions_s1s2 = projected_emissions.groupby(level=0, sort=False).sum()  # add scope 1 and 2

        return projected_emissions_s1s2

    def _unit_of_measure_correction(self, company_ids: List[str], projected_emission: pd.DataFrame) -> pd.DataFrame:
        """
        :param company_ids: list of company ids
        :param projected_emission: series of projected emissions
        :return: series of projected emissions corrected for unit of measure
        """
        projected_emission.loc[self.get_value(company_ids, ColumnsConfig.SECTOR).isin(SectorsConfig.CORRECTION_SECTORS),
                               range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                     TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)] *= \
            self.ENERGY_UNIT_CONVERSION_FACTOR
        return projected_emission

    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: list of company ids
        :return: DataFrame with projected targets per company extracted from the excel
        """
        return self._get_projection(company_ids, TabsConfig.PROJECTED_TARGET)

    def get_company_projected_intensities(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: list of company ids
        :return: DataFrame with projected intensities per company extracted from the excel
        """
        return self._get_projection(company_ids, TabsConfig.PROJECTED_EI)

    def get_company_intensity_and_production_at_base_year(self, company_ids: List[str]) -> pd.DataFrame:
        """
        overrides subclass method
        :param company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        """

        df_company_data = pd.DataFrame.from_records([c.dict() for c in self.get_company_data(company_ids)])
        company_info = df_company_data[[
            ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.GHG_SCOPE12]].set_index(
            ColumnsConfig.COMPANY_ID)
        ei_at_base = self._get_company_intensity_at_year(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                         company_ids).rename(ColumnsConfig.BASE_EI)
        return company_info.merge(ei_at_base, left_index=True, right_index=True)
