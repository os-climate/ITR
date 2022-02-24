import warnings # needed until apply behaves better with Pint quantities in arrays
from typing import Type, List, Optional
import pandas as pd
import numpy as np


import pint
ureg = pint.get_application_registry()
Q_ = ureg.Quantity

from pydantic import ValidationError
from ITR.data.base_providers import BaseCompanyDataProvider, BaseProviderProductionBenchmark, \
    BaseProviderIntensityBenchmark, EITargetProjector
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, VariablesConfig, TabsConfig
from ITR.interfaces import ICompanyData, EScope, \
    IHistoricEmissionsScopes, \
    IProductionRealization, IHistoricEIScopes, IHistoricData, ITargetData, IEmissionRealization, IEIRealization, IProjection

import logging

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
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        self._companies = self._convert_from_template_company_data(excel_path)
        # self.historic_years = None
        super().__init__(self._companies, column_config, tempscore_config)

    def _calculate_target_projections(self,
                                      production_bm: BaseProviderProductionBenchmark,
                                      EI_bm: BaseProviderIntensityBenchmark):
        """
        We cannot calculate target projections until after we have loaded benchmark data.
        
        :param Production_bm: A Production Benchmark (multi-sector, single-scope, 2020-2050)
        :param EI_bm: An Emissions Intensity Benchmark (multi-sector, single-scope, 2020-2050)
        """
        for c in self._companies:
            if c.projected_targets is not None:
                continue
            elif c.target_data is None:
                print(f"no target data for {c.company_name}")
                continue
            else:
                base_year_production = next((p.value for p in c.historic_data.productions if p.year == self.temp_config.CONTROLS_CONFIG.base_year), None)
                company_sector_region_info = pd.DataFrame({
                    self.column_config.COMPANY_ID: c.company_id,
                    # self.column_config.GHG_SCOPE12 is incorrect in production_bm.get_company_projected_production.
                    # Should be production value at base_year as defined in temp_config.CONTROLS_CONFIG
                    # Do not confuse this base year metric with any target base year.
                    # Historic data is given in terms of its own EMISSIONS_METRIC and PRODUCTION_METRIC
                    # TODO: don't use c.production_metric; rather, grovel through c to address appropriately using PRODUCTION_METRIC text string.
                    self.column_config.GHG_SCOPE12: base_year_production.to(c.production_metric.units).magnitude,
                    self.column_config.SECTOR: c.sector,
                    self.column_config.REGION: c.region
                }, index=[0])
                bm_production_data = (production_bm.get_company_projected_production(company_sector_region_info)
                                      # We transpose the data so that we get a pd.Series that will accept the pint units as a whole (not element-by-element)
                                      .iloc[0].T
                                      .astype(f'pint[{str(base_year_production.units)}]'))
                c.projected_targets = EITargetProjector().project_ei_targets(c.target_data, c.historic_data, bm_production_data)
    
    def _check_company_data(self, df: pd.DataFrame) -> None:
        """
        Checks if the company data excel contains the data in the right format

        :return: None
        """
        required_tabs = [TabsConfig.TEMPLATE_INPUT_DATA, TabsConfig.TEMPLATE_TARGET_DATA]
        missing_tabs = [tab for tab in required_tabs if tab not in df.keys()]
        assert not any(tab in missing_tabs for tab in required_tabs), f"Tabs {required_tabs} are required."

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
            if suffix!='production':
                suffix = suffix.upper()
            return f"{suffix}-{prefix}"
        
        df_company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)
        self._check_company_data(df_company_data)

        input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA
        if "Test input data" in df_company_data:
            input_data_sheet = "Test input data"
        df_fundamentals = df_company_data[input_data_sheet].set_index(ColumnsConfig.COMPANY_ID, drop=False).convert_dtypes()
        # GH https://github.com/pandas-dev/pandas/issues/46044
        df_fundamentals.company_id = df_fundamentals.company_id.astype('object')
        
        company_ids = df_fundamentals[ColumnsConfig.COMPANY_ID].unique()
        # The nightmare of naming columns 20xx_metric instead of metric_20xx...and potentially dealing with data from 1990s...
        historic_columns = [col for col in df_fundamentals.columns if col[:1].isdigit()]
        historic_scopes = ['S1', 'S2', 'S3', 'S1S2', 'S1S2S3', 'production']
        df_historic = df_fundamentals[['company_id'] + historic_columns].dropna(axis=1,how='all')
        df_fundamentals = df_fundamentals[df_fundamentals.columns.difference(historic_columns, sort=False)]
        # df_fundamentals now ready for conversion to list of models
        
        df_historic = df_historic.rename(columns={col:_fixup_name(col) for col in historic_columns})
        df = pd.wide_to_long(df_historic, historic_scopes, i='company_id', j='year', sep='-', suffix='\d+').reset_index()
        df2 = (df.pivot(index='company_id', columns='year', values=historic_scopes)
               .stack(level=0)
               .reset_index()
               .rename(columns={'level_1':ColumnsConfig.SCOPE})
               .set_index('company_id'))
        df2.loc[df2[ColumnsConfig.SCOPE]=='production', ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS
        df2.loc[df2[ColumnsConfig.SCOPE]!='production', ColumnsConfig.VARIABLE] = VariablesConfig.EMISSIONS
        df3 = df2.reset_index().set_index(['company_id', 'variable', 'scope'])
        df3 = pd.concat([df3.xs(VariablesConfig.PRODUCTIONS,level=1,drop_level=False)
                         .apply(lambda x: x.map(lambda y: Q_(y, df_fundamentals.loc[df_fundamentals.company_id==x.name[0],
                                                                                    'production_metric'].squeeze())), axis=1),
                         df3.xs(VariablesConfig.EMISSIONS,level=1,drop_level=False)
                         .apply(lambda x: x.map(lambda y: Q_(y, df_fundamentals.loc[df_fundamentals.company_id==x.name[0],
                                                                                    'emissions_metric'].squeeze())), axis=1)])
        df4 = df3.xs(VariablesConfig.EMISSIONS,level=1) / df3.xs((VariablesConfig.PRODUCTIONS,'production'),level=[1,2])
        df4['variable'] = VariablesConfig.EMISSIONS_INTENSITIES
        df4 = df4.reset_index().set_index(['company_id', 'variable', 'scope'])
        df5 = pd.concat([df3, df4])
        df_historic_data = df5
        # df_historic now ready for conversion to model for each company
        self.historic_years = [column for column in df_historic_data.columns if type(column) == int]

        input_target_sheet = TabsConfig.TEMPLATE_TARGET_DATA
        if "Test target data" in df_company_data:
            input_target_sheet = "Test target data"
        df_target_data = df_company_data[input_target_sheet].set_index('company_id').convert_dtypes()
        
        # TODO: need to fix Pydantic definition or data to allow optional int.  In the mean time...
        df_target_data.loc[df_target_data.target_start_year.isna(), 'target_start_year'] = 2020
        df_target_data.loc[df_target_data.netzero_year.isna(), 'netzero_year'] = 2050
        
        # company_id, netzero_year, target_type, target_scope, target_start_year, target_base_year, target_base_year_qty, target_base_year_unit, target_year, target_reduction_ambition
        # df_target_data now ready for conversion to model for each company
        return self._company_df_to_model(df_fundamentals, df_target_data, df_historic_data)

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
        logger = logging.getLogger(__name__)
        # set NaN to None since NaN is float instance
        df_fundamentals = df_fundamentals.where(pd.notnull(df_fundamentals), None).replace({np.nan: None})

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
                
                # v = df_fundamentals[df_fundamentals[ColumnsConfig.COMPANY_ID]==company_id][ColumnsConfig.GHG_SCOPE12].squeeze()
                # company_data[ColumnsConfig.GHG_SCOPE12] = Q_(v or np.nan, ureg(units))
                # v = df_fundamentals[df_fundamentals[ColumnsConfig.COMPANY_ID]==company_id][ColumnsConfig.GHG_SCOPE3].squeeze()
                # company_data[ColumnsConfig.GHG_SCOPE3] = Q_(v or np.nan, ureg(units))

                # df.loc[[index]] is like df.loc[index, :] except it always returns a DataFrame and not a Series when there's only one row
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
                    company_data[ColumnsConfig.PRODUCTION_METRIC] = { 'units': company_data[ColumnsConfig.PRODUCTION_METRIC]}
                if company_data[ColumnsConfig.EMISSIONS_METRIC]:
                    company_data[ColumnsConfig.EMISSIONS_METRIC] = { 'units': company_data[ColumnsConfig.EMISSIONS_METRIC]}

                model_companies.append(ICompanyData.parse_obj(company_data))
            except ValidationError as e:
                logger.warning(
                    f"EX {e}: (one of) the input(s) of company %s is invalid and will be skipped" % company_data[
                        ColumnsConfig.COMPANY_NAME])
                continue
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

        projections = projections.reset_index().set_index(ColumnsConfig.COMPANY_ID)

        assert all(company_id in projections.index for company_id in company_ids), \
            f"company ids missing in provided projections"

        projections = projections.loc[company_ids, range(TemperatureScoreConfig.CONTROLS_CONFIG.base_year,
                                                         TemperatureScoreConfig.CONTROLS_CONFIG.target_end_year + 1)]
        # Due to bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero workaround below:
        projected_emissions_s1s2 = projections.groupby(level=0, sort=False).agg(ExcelProviderCompany._np_sum)  # add scope 1 and 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            projected_emissions_s1s2 = projected_emissions_s1s2.apply(lambda x: x.astype(f'pint[??t CO2/({production_metric[x.name]})]'), axis=1)

        return projected_emissions_s1s2

# class ITargetData(PintModel):
#     netzero_year: int
#     target_type: Union[Literal['intensity'],Literal['absolute'],Literal['other']]
#     target_scope: EScope
#     start_year: Optional[int]
#     base_year: int
#     end_year: int
    
#     target_base_qty: float
#     target_base_unit: str
#     target_reduction_pct: float

    def _convert_target_data(self, target_data: pd.DataFrame) -> List[ITargetData]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :return: IHistoricData Pydantic object
        """
        target_data = target_data.rename(columns={'target_base_year':'base_year',
                                                  'target_start_year':'start_year',
                                                  'target_year':'end_year',
                                                  'target_reduction_ambition':'target_reduction_pct',
                                                  'target_base_year_qty':'target_base_qty',
                                                  'target_base_year_unit':'target_base_unit'})
        return [ITargetData(**td) for td in target_data.to_dict('records')]

    def _get_historic_data(self, company_ids: List[str], historic_data: pd.DataFrame) -> pd.DataFrame:
        """
        get the historic data for list of companies
        :param company_ids: list of company ids
        :param historic_data: Dataframe Productions, Emissions, and Emission Intensities mixed together
        :return: historic data with unit attributes added on a per-element basis
        """
        # We don't need this reset/set index dance because we set the index to COMPANY_ID to get units sorted
        # historic_data = historic_data.reset_index().drop(columns=['index']).set_index(ColumnsConfig.COMPANY_ID)
        
        missing_ids = [company_id for company_id in company_ids if company_id not in historic_data.index]
        assert not missing_ids, f"Company ids missing in provided historic data: {missing_ids}"

        # There has got to be a better way to do this...
        historic_data = (
            historic_data.loc[company_ids, :]
            .apply(lambda x: pd.Series({col:x[col] for col in x.index if type(col)!=int}
                                       | {y:f"{x[y]} {x['units']}" for y in self.historic_years},
                                       index=x.index),
                   axis=1)
        )
        return historic_data

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
                else [IEmissionRealization(year=year, value=results[year].squeeze()) for year in self.historic_years]
        return IHistoricEmissionsScopes(**emissions_scopes)

    def _convert_to_historic_productions(self, productions: pd.DataFrame) \
            -> Optional[List[IProductionRealization]]:
        """
        :param historic: historic production, emission and emission intensity data for a company
        :return: A list containing historic productions, or None if no data are provided
        """
        if productions.empty:
            return None

        try:
            production_realizations = \
                [IProductionRealization(year=year, value=productions[year].squeeze()) for year in self.historic_years]
        except TypeError as e:
            print(e)
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
            try:
                intensity_scopes[scope] = [] \
                    if results.empty \
                    else [IEIRealization(year=year, value=results[year].squeeze()) for year in self.historic_years]
            except TypeError as e:
                print(e)
        return IHistoricEIScopes(**intensity_scopes)
