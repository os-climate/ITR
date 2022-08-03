import warnings  # needed until apply behaves better with Pint quantities in arrays
from typing import Type, List, Optional
import pandas as pd
import numpy as np
import logging
import pint
from pydantic import ValidationError

from ITR.data.base_providers import BaseCompanyDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, VariablesConfig, TabsConfig, SectorsConfig, LoggingConfig
from ITR.interfaces import ICompanyData, EScope, \
    IHistoricEmissionsScopes, \
    IProductionRealization, IHistoricEIScopes, IHistoricData, ITargetData, IEmissionRealization, IEIRealization, \
    IProjection, ProjectionControls
from ITR.utils import get_project_root

ureg = pint.get_application_registry()
Q_ = ureg.Quantity

pkg_root = get_project_root()
df_country_regions = pd.read_csv(f"{pkg_root}/data/input/country_region_info.csv")

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


def ITR_country_to_region(country):
    if len(country)==2:
        regions = df_country_regions[df_country_regions.alpha_2==country].region_ar6_10
    elif len(country)==3:
        regions = df_country_regions[df_country_regions.alpha_3==country].region_ar6_10
    else:
        if country in df_country_regions.name:
            regions = df_country_regions[df_country_regions.name==country].region_ar6_10
        elif country in df_country_regions.common_name:
            regions = df_country_regions[df_country_regions.common_name==country].region_ar6_10
        else:
            raise ValueError(f"country {country} not found")
    region = regions.squeeze()
    if region in ['North America', 'Europe']:
        return region
    if 'Asia' in region:
        return 'Asia'
    return 'Global'


class TemplateProviderCompany(BaseCompanyDataProvider):
    """
    Data provider skeleton for CSV files. This class serves primarily for testing purposes only!
    As of Feb 2022, we are testing!!

    :param excel_path: A path to the Excel file with the company data
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
    """

    def __init__(self, excel_path: str,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig,
                 projection_controls: Type[ProjectionControls] = ProjectionControls):
        self._companies = self._convert_from_template_company_data(excel_path)
        super().__init__(self._companies, column_config, tempscore_config, projection_controls)

    def _convert_from_template_company_data(self, excel_path: str) -> List[ICompanyData]:
        """
        Converts the Excel template to list of ICompanyData objects. All dataprovider features will be inhereted from
        Base
        :param excel_path: file path to excel file
        :return: List of ICompanyData objects
        """

        def _fixup_name(x):
            prefix, _, suffix = x.partition('_')
            suffix = suffix.replace('ghg_', '')
            if suffix != 'production':
                suffix = suffix.upper()
            return f"{suffix}-{prefix}"

        df_company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)

        input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA
        try:
            df = df_company_data[input_data_sheet]
        except KeyError as e:
            logger.error(f"Tab {input_data_sheet} is required in input Excel file.")
            raise

        df['exposure'].fillna('presumed_equity', inplace=True)
        # TODO: Fix market_cap column naming inconsistency
        df.rename(
            columns={'revenue': 'company_revenue', 'market_cap': 'company_market_cap',
                     'ev': 'company_enterprise_value', 'evic': 'company_ev_plus_cash',
                     'assets': 'company_total_assets'}, inplace=True)
        df.loc[df.region.isnull(), 'region'] = df.apply(lambda x: ITR_country_to_region(x.country), axis=1)

        df_fundamentals = df.set_index(ColumnsConfig.COMPANY_ID, drop=False).convert_dtypes()
        # GH https://github.com/pandas-dev/pandas/issues/46044
        df_fundamentals.company_id = df_fundamentals.company_id.astype('object')

        # testing if all data is in the same currency
        if len(df_fundamentals[ColumnsConfig.TEMPLATE_CURRENCY].unique()) != 1:
            error_message = f"All data should be in the same currency."
            logger.error(error_message)
            raise ValueError(error_message)

        # are there empty sectors?
        comp_with_missing_sectors = df_fundamentals[ColumnsConfig.COMPANY_ID][df_fundamentals[ColumnsConfig.SECTOR].isnull()].to_list()
        if comp_with_missing_sectors:
            error_message = f"For {comp_with_missing_sectors} companies the sector column is empty."
            logger.error(error_message)
            raise ValueError(error_message)

        # testing if only valid sectors are provided
        sectors_from_df = df_fundamentals[ColumnsConfig.SECTOR].unique()
        configured_sectors = SectorsConfig.get_configured_sectors()
        not_configured_sectors = [sec for sec in sectors_from_df if sec not in configured_sectors]
        if not_configured_sectors:
            error_message = f"Sector {not_configured_sectors} is not covered by the ITR tool currently."
            logger.error(error_message)
            raise ValueError(error_message)

        # The nightmare of naming columns 20xx_metric instead of metric_20xx...and potentially dealing with data from 1990s...
        historic_columns = [col for col in df_fundamentals.columns if col[:1].isdigit()]
        historic_scopes = ['S1', 'S2', 'S3', 'S1S2', 'S1S2S3', 'production']
        df_historic = df_fundamentals[['company_id'] + historic_columns].dropna(axis=1, how='all')
        df_fundamentals = df_fundamentals[df_fundamentals.columns.difference(historic_columns, sort=False)]

        # Checking if there are not many missing market cap
        missing_cap_ids = df_fundamentals[ColumnsConfig.COMPANY_ID][df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP].isnull()].to_list()
        # For the missing Market Cap we should use the ratio below to get dummy market cap:
        #   (Avg for the Sector (Market Cap / Revenues) + Avg for the Sector (Market Cap / Assets)) 2
        df_fundamentals['MCap_to_Reven']=df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP]/df_fundamentals[ColumnsConfig.COMPANY_REVENUE] # new temp column with ratio
        df_fundamentals['MCap_to_Assets']=df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP]/df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS] # new temp column with ratio
        df_fundamentals['AVG_MCap_to_Reven'] = df_fundamentals.groupby(ColumnsConfig.SECTOR)['MCap_to_Reven'].transform('mean')
        df_fundamentals['AVG_MCap_to_Assets'] = df_fundamentals.groupby(ColumnsConfig.SECTOR)['MCap_to_Assets'].transform('mean')
        df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP] = df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP].fillna(0.5*(df_fundamentals[ColumnsConfig.COMPANY_REVENUE] * df_fundamentals['AVG_MCap_to_Reven']+df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS] * df_fundamentals['AVG_MCap_to_Assets']))
        df_fundamentals.drop(['MCap_to_Reven','MCap_to_Assets','AVG_MCap_to_Reven','AVG_MCap_to_Assets'], axis=1, inplace=True) # deleting temporary columns
        
        if missing_cap_ids:
            logger.warning(f"Missing market capitalisation values are estimated for companies with ID: "
                           f"{missing_cap_ids}.")

        # df_fundamentals now ready for conversion to list of models

        df_historic = df_historic.rename(columns={col: _fixup_name(col) for col in historic_columns})
        df = pd.wide_to_long(df_historic, historic_scopes, i='company_id', j='year', sep='-',
                             suffix='\d+').reset_index()
        df2 = (df.pivot(index='company_id', columns='year', values=historic_scopes)
               .stack(level=0)
               .reset_index()
               .rename(columns={'level_1': ColumnsConfig.SCOPE})
               .set_index('company_id'))
        df2.loc[df2[ColumnsConfig.SCOPE] == 'production', ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS
        df2.loc[df2[ColumnsConfig.SCOPE] != 'production', ColumnsConfig.VARIABLE] = VariablesConfig.EMISSIONS
        df3 = df2.reset_index().set_index(['company_id', 'variable', 'scope'])
        df3 = pd.concat([df3.xs(VariablesConfig.PRODUCTIONS, level=1, drop_level=False)
            .apply(
            lambda x: x.map(lambda y: Q_(y if y is not pd.NA else np.nan,
                                         df_fundamentals.loc[df_fundamentals.company_id == x.name[0],
                                                             'production_metric'].squeeze())), axis=1),
            df3.xs(VariablesConfig.EMISSIONS, level=1, drop_level=False)
            .apply(lambda x: x.map(
                lambda y: Q_(y if y is not pd.NA else np.nan,
                             df_fundamentals.loc[df_fundamentals.company_id == x.name[0],
                                                 'emissions_metric'].squeeze())), axis=1)])
        df4 = df3.xs(VariablesConfig.EMISSIONS, level=1) / df3.xs((VariablesConfig.PRODUCTIONS, 'production'),
                                                                  level=[1, 2])
        df4['variable'] = VariablesConfig.EMISSIONS_INTENSITIES
        df4 = df4.reset_index().set_index(['company_id', 'variable', 'scope'])
        df5 = pd.concat([df3, df4])
        df_historic_data = df5
        # df_historic now ready for conversion to model for each company
        self.historic_years = [column for column in df_historic_data.columns if type(column) == int]

        test_target_sheet = TabsConfig.TEMPLATE_TARGET_DATA
        try:
            df_target_data = df_company_data[test_target_sheet].set_index('company_id').convert_dtypes()
        except KeyError:
            logger.error(f"Tab {test_target_sheet} is required in input Excel file.")
            raise

        df_target_data = self._validate_target_data(df_target_data)

        # company_id, netzero_year, target_type, target_scope, target_start_year, target_base_year, target_base_year_qty, target_base_year_unit, target_year, target_reduction_ambition
        # df_target_data now ready for conversion to model for each company
        return self._company_df_to_model(df_fundamentals, df_target_data, df_historic_data)

    def _validate_target_data(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs checks on the supplied target data. Some values are put in to make the tool function.
        :param target_data:
        :return:
        """
        # TODO: need to fix Pydantic definition or data to allow optional int.  In the mean time...
        c_ids_without_start_year = list(target_data[target_data['target_start_year'].isna()].index)
        if c_ids_without_start_year:
            target_data.loc[target_data.target_start_year.isna(), 'target_start_year'] = 2021
            logger.warning(f"Missing target start year set to 2021 for companies with ID: {c_ids_without_start_year}")

        c_ids_invalid_netzero_year = list(target_data[target_data['netzero_year'] > 2050].index)
        if c_ids_invalid_netzero_year:
            error_message = f"Invalid net-zero target years (>2050) are entered for companies with ID: " \
                            f"{c_ids_without_netzero_year}"
            logger.error(error_message)
            raise ValueError(error_message)
        target_data.loc[target_data.netzero_year.isna(), 'netzero_year'] = 2050

        c_ids_with_increase_target = list(target_data[target_data['target_reduction_ambition'] < 0].index)
        if c_ids_with_increase_target:
            error_message = f"Negative target reduction ambition is invalid and entered for companies with ID: " \
                            f"{c_ids_with_increase_target}"
            logger.error(error_message)
            raise ValueError(error_message)

        return target_data

    def _convert_series_to_IProjections(self, projections: pd.Series) -> [IProjection]:
        """
        Converts a Pandas Series to a list of IProjection
        :param projections: Pandas Series with years as indices
        :return: List of IProjection objects
        """
        return [IProjection(year=y, value=v) for y, v in projections.items()]

    def _company_df_to_model(self, df_fundamentals: pd.DataFrame,
                             df_target_data: pd.DataFrame,
                             df_historic_data: pd.DataFrame) -> List[ICompanyData]:

        """
        transforms target Dataframe into list of ICompanyData instances.
        We don't necessarily have enough info to do target projections at this stage.

        :param df_fundamentals: pandas Dataframe with fundamental data
        :param df_target_data: pandas Dataframe with target data
        :param df_historic_data: pandas Dataframe with historic emissions, intensity, and production information
        :return: A list containing the ICompanyData objects
        """
        companies_data_dict = df_fundamentals.to_dict(orient="records")
        model_companies: List[ICompanyData] = []
        for company_data in companies_data_dict:
            # company_data is a dict, not a dataframe
            try:
                # In this world (different from excel.py) we initialize projected_intensities and projected_targets
                # in a later step, after we know we have valid benchmark data
                company_id = company_data[ColumnsConfig.COMPANY_ID]

                # the ghg_s1s2 and ghg_s3 variables are values "as of" the financial data
                # TODO pull ghg_s1s2 and ghg_s3 from historic data as appropriate

                if df_historic_data is not None:
                    company_data[ColumnsConfig.HISTORIC_DATA] = self._convert_historic_data(
                        df_historic_data.loc[[company_data[ColumnsConfig.COMPANY_ID]]].reset_index()).dict()
                else:
                    company_data[ColumnsConfig.HISTORIC_DATA] = None

                if df_target_data is not None and company_id in df_target_data.index:
                    company_data[ColumnsConfig.TARGET_DATA] = [td.dict() for td in self._convert_target_data(
                        df_target_data.loc[[company_data[ColumnsConfig.COMPANY_ID]]].reset_index())]
                else:
                    company_data[ColumnsConfig.TARGET_DATA] = None

                if company_data[ColumnsConfig.PRODUCTION_METRIC]:
                    company_data[ColumnsConfig.PRODUCTION_METRIC] = {
                        'units': company_data[ColumnsConfig.PRODUCTION_METRIC]}
                if company_data[ColumnsConfig.EMISSIONS_METRIC]:
                    company_data[ColumnsConfig.EMISSIONS_METRIC] = {
                        'units': company_data[ColumnsConfig.EMISSIONS_METRIC]}

                # handling of missing market cap data is mainly done in _convert_from_template_company_data()
                if company_data[ColumnsConfig.COMPANY_MARKET_CAP] is pd.NA:
                    company_data[ColumnsConfig.COMPANY_MARKET_CAP] = np.nan

                model_companies.append(ICompanyData.parse_obj(company_data))
            except ValidationError:
                logger.error(f"(One of) the input(s) of company {company_data['company_name']} is invalid")
                raise
        return model_companies

    # Workaround for bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero 
    def _np_sum(g):
        return np.sum(g.values)

    def _get_projection(self, company_ids: List[str], projections: pd.DataFrame,
                        production_metric: pd.DataFrame) -> pd.DataFrame:
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
        projected_ei_s1s2 = projections.groupby(level=0, sort=False).agg(
            TemplateProviderCompany._np_sum)  # add scope 1 and 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            projected_ei_s1s2 = projected_ei_s1s2.apply(
                lambda x: x.astype(f'pint[??t CO2/({production_metric[x.name]})]'), axis=1)

        return projected_ei_s1s2

    def _convert_target_data(self, target_data: pd.DataFrame) -> List[ITargetData]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :return: IHistoricData Pydantic object
        """
        target_data = target_data.rename(columns={'target_year': 'target_end_year', 'target_reduction_ambition': 'target_reduction_pct',})
        return [ITargetData(**td) for td in target_data.to_dict('records')]

    def _get_historic_data(self, company_ids: List[str], historic_data: pd.DataFrame) -> pd.DataFrame:
        """
        get the historic data for list of companies
        :param company_ids: list of company ids
        :param historic_data: Dataframe Productions, Emissions, and Emission Intensities mixed together
        :return: historic data with unit attributes added on a per-element basis
        """
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
        :param historic: historic production, emission and emission intensity data for a company (already unitized)
        :return: IHistoricData Pydantic object
        """
        historic.set_index('variable', drop=False, inplace=True)
        productions = historic.loc[[VariablesConfig.PRODUCTIONS]]
        emissions = historic.loc[[VariablesConfig.EMISSIONS]]
        emissions_intensities = historic.loc[[VariablesConfig.EMISSIONS_INTENSITIES]]
        hd = IHistoricData(productions=self._convert_to_historic_productions(productions),
                           emissions=self._convert_to_historic_emissions(emissions),
                           emissions_intensities=self._convert_to_historic_ei(emissions_intensities))
        return hd

    # Note that for the three following functions, we pd.Series.squeeze() the results because it's just one year / one company
    def _convert_to_historic_emissions(self, emissions: pd.DataFrame) -> Optional[IHistoricEmissionsScopes]:
        """
        :param emissions: historic emissions data for a company
        :return: List of historic emissions per scope, or None if no data are provided
        """
        if emissions.empty:
            return None

        emissions_scopes = {}
        for scope in EScope.get_scopes():
            results = emissions.loc[emissions[ColumnsConfig.SCOPE] == scope]
            emissions_scopes[scope] = [] \
                if results.empty \
                else [IEmissionRealization(year=year, value=results[year].squeeze()) for year in self.historic_years]
        return IHistoricEmissionsScopes(**emissions_scopes)

    def _convert_to_historic_productions(self, productions: pd.DataFrame) -> Optional[List[IProductionRealization]]:
        """
        :param productions: historic production data for a company
        :return: A list containing historic productions, or None if no data are provided
        """
        if productions.empty:
            return None

        return [IProductionRealization(year=year, value=productions[year].squeeze()) for year in self.historic_years]

    def _convert_to_historic_ei(self, intensities: pd.DataFrame) -> Optional[IHistoricEIScopes]:
        """
        :param intensities: historic emission intensity data for a company
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
                else [IEIRealization(year=year, value=results[year].squeeze()) for year in self.historic_years]
        return IHistoricEIScopes(**intensity_scopes)
