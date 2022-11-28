import warnings  # needed until apply behaves better with Pint quantities in arrays
from typing import Type, List, Optional
import pandas as pd
import numpy as np

from pydantic import ValidationError

import ITR
from ITR.data.osc_units import ureg, Q_, PA_, ProductionQuantity, EmissionsQuantity, EI_Quantity
import pint

from ITR.data.base_providers import BaseCompanyDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, VariablesConfig, TabsConfig, SectorsConfig, ProjectionControls
from ITR.interfaces import ICompanyData, EScope, IHistoricEmissionsScopes, IProductionRealization, \
    IHistoricEIScopes, IHistoricData, ITargetData, IEmissionRealization, IEIRealization, \
    IProjection
from ITR.utils import get_project_root
from ITR.logger import logger

pkg_root = get_project_root()
df_country_regions = pd.read_csv(f"{pkg_root}/data/input/country_region_info.csv")


def ITR_country_to_region(country:str) -> str:
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


# FIXME: circa line 480 of pint_array is code for master_scalar that shows how to decide if a thing has units

def _estimated_value(y: pd.Series) -> pint.Quantity: 
    """
    Parameters
    ----------
    y : a pd.Series that arrives via a pd.GroupBy operation.
        The elements of the series are all data (or np.nan) matching a metric/sub-metric.
        This function 

    Returns
    -------
    A Quantity which could be either the first or only element from the pd.Series,
    or an estimate of correct answer.
        The estimateion is based on the mean of the non-null entries.
        It could be changed to output the last (most recent) value (if the inputs arrive sorted)
    """

    try:
        if isinstance(y, pd.DataFrame):
            # Something went wrong with the GroupBy operation and we got a pd.DataFrame
            # insted of being called column-by-column.
            logger.error("Cannot estimate value of whole DataFrame; Something went wrong with GroupBy operation")
            breakpoint()
            raise ValueError
        # This relies on the fact that we can now see Quantity(np.nan, ...) for both float and ufloat magnitudes
        x = PA_._from_sequence(y)
        xq = x.quantity
        xm = xq.m
        x = y[~ITR.isnan(PA_._from_sequence(y).quantity.m)]
    except TypeError:
        logger.error(f"type_error({y}) returning {y.values}[0]")
        breakpoint()
        x = PA_._from_sequence(y)
        xq = x.quantity
        xm = xq.m
        return y.iloc[0]
    if len(x) == 0:
        # If all inputs are NaN, return the first NaN
        return y.iloc[0]
    if len(x) == 1:
        # If there's only one non-NaN input, return that one
        return x.iloc[0]
    if isinstance(x.values[0], pint.Quantity):
        values = x.values
        units = values[0].u
        assert all([v.u==units for v in values])
        if ITR.HAS_UNCERTAINTIES:
            wavg = ITR.umean(values)
        else:
            wavg = np.mean(values)
        est = Q_(wavg, units)
    else:
        logger.error(f"non-qty: _estimated_values called on non-Quantity {x.values[0]};;;")
        est = x.mean()
    return est

# FIXME: Should make this work with a pure PintArray
def prioritize_submetric(x: pd.Series) -> pint.Quantity:
    """
    Parameters
    ----------
    x : pd.Series
        The first column of the pd.Series is the list of submetrics in priority order
        All subsequent columns are observations to be selected from.

    Returns
    -------
    y : The first non-NaN quantity using the given priority order.
    """
    if isinstance(x.iloc[0], str):
        # Nothing to prioritize
        return x
    y = x.copy()
    for c in range(1,len(x)):
        if not isinstance(x.iloc[c], np.ndarray):
            # If we don't have a list to prioritize, don't try
            continue
        for p in range(0, len(x.iloc[0])):
            if not ITR.isnan(x.iloc[c][p]):
                # Replace array to be prioritized with best choice
                y.iloc[c] = x.iloc[c][p]
                break
        if isinstance(y.iloc[c], np.ndarray):
            # If we made it to the end without replacing, arbitrarily pick the first value
            y.iloc[c] = y.iloc[c][0]
    return y


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
}
s3_category_dict = { v.lower():k for k, v in s3_category_rdict.items() }

# FIXME: Should we change this to derive from ExcelProviderCompany?
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

        template_version = 1

        def _fixup_name(x):
            prefix, _, suffix = x.partition('_')
            suffix = suffix.replace('ghg_', '')
            if suffix != 'production':
                suffix = suffix.upper()
            return f"{suffix}-{prefix}"
        
        df_company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)

        if TabsConfig.TEMPLATE_INPUT_DATA_V2 and TabsConfig.TEMPLATE_INPUT_DATA_V2 in df_company_data:
            template_version = 2
            input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA_V2
        else:
            input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA
        try:
            df = df_company_data[input_data_sheet]
        except KeyError as e:
            logger.error(f"Tab {input_data_sheet} is required in input Excel file.")
            raise KeyError
        if template_version==2:
            esg_data_sheet = TabsConfig.TEMPLATE_ESG_DATA_V2
            try:
                df_esg = df_company_data[esg_data_sheet].drop(columns='company_lei').copy() # .iloc[0:45]
                df_esg.loc[df_esg.submetric.map(lambda x: type(x)!=str), 'submetric'] = ''
            except KeyError as e:
                logger.error(f"Tab {esg_data_sheet} is required in input Excel file.")
                raise KeyError
            df_esg.loc[:, [ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID]] = (
                df_esg.loc[:, [ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID]].fillna(method='ffill')
                )

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

        # Checking missing company ids
        missing_company_ids = df_fundamentals[ColumnsConfig.COMPANY_ID].isnull().any()
        if missing_company_ids:
            error_message = "Missing company ids"
            logger.error(error_message)
            raise ValueError(error_message)

        # ignore company data that does not come with emissions and/or production metrics
        if template_version==1:
            missing_esg_metrics_df = df_fundamentals[ColumnsConfig.COMPANY_ID][
                df_fundamentals[ColumnsConfig.EMISSIONS_METRIC].isnull() | df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].isnull()]
        else:
            missing_production_idx = df_fundamentals.index.difference(df_esg[df_esg.metric.eq('production')].company_id.unique())
            missing_esg_idx = df_fundamentals.index.difference(df_esg[df_esg.metric.str.upper().isin(['S1', 'S1S2', 'S3', 'S1S2S3'])].company_id.unique())
            missing_esg_metrics_df = df_fundamentals.loc[missing_production_idx.union(missing_esg_idx)]
            
        if len(missing_esg_metrics_df)>0:
            logger.warning(f"Missing ESG metrics for companies with ID (will be ignored): "
                           f"{missing_esg_metrics_df.index}.")
            df_fundamentals = df_fundamentals[~df_fundamentals.index.isin(missing_esg_metrics_df.index)]

        # Checking if there are not many missing market cap
        missing_cap_ids = df_fundamentals[ColumnsConfig.COMPANY_ID][df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP].isnull()].to_list()
        # For the missing Market Cap we should use the ratio below to get dummy market cap:
        #   (Avg for the Sector (Market Cap / Revenues) + Avg for the Sector (Market Cap / Assets)) 2
        df_fundamentals['MCap_to_Reven']=df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP]/df_fundamentals[ColumnsConfig.COMPANY_REVENUE] # new temp column with ratio
        df_fundamentals['MCap_to_Assets']=df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP]/df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS] # new temp column with ratio
        df_fundamentals['AVG_MCap_to_Reven'] = df_fundamentals.groupby(ColumnsConfig.SECTOR)['MCap_to_Reven'].transform('mean')
        df_fundamentals['AVG_MCap_to_Assets'] = df_fundamentals.groupby(ColumnsConfig.SECTOR)['MCap_to_Assets'].transform('mean')
        # FIXME: Add uncertainty here!
        df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP] = df_fundamentals[ColumnsConfig.COMPANY_MARKET_CAP].fillna(0.5*(df_fundamentals[ColumnsConfig.COMPANY_REVENUE] * df_fundamentals['AVG_MCap_to_Reven']+df_fundamentals[ColumnsConfig.COMPANY_TOTAL_ASSETS] * df_fundamentals['AVG_MCap_to_Assets']))
        df_fundamentals.drop(['MCap_to_Reven','MCap_to_Assets','AVG_MCap_to_Reven','AVG_MCap_to_Assets'], axis=1, inplace=True) # deleting temporary columns
        
        if missing_cap_ids:
            logger.warning(f"Missing market capitalisation values are estimated for companies with ID: "
                           f"{missing_cap_ids}.")

        if template_version==1:
            # The nightmare of naming columns 20xx_metric instead of metric_20xx...and potentially dealing with data from 1990s...
            historic_columns = [col for col in df_fundamentals.columns if col[:1].isdigit()]
            historic_scopes = ['S1', 'S2', 'S3', 'S1S2', 'S1S2S3', 'production']
            df_historic = df_fundamentals[['company_id'] + historic_columns].dropna(axis=1, how='all')
            df_fundamentals = df_fundamentals[df_fundamentals.columns.difference(historic_columns, sort=False)]

            # df_fundamentals now ready for conversion to list of models

            df_historic = df_historic.rename(columns={col: _fixup_name(col) for col in historic_columns})
            df = pd.wide_to_long(df_historic, historic_scopes, i='company_id', j='year', sep='-',
                                 suffix='\d+').reset_index()
            df2 = (df.pivot(index='company_id', columns='year', values=historic_scopes)
                   .stack(level=0, dropna=False)
                   .reset_index()
                   .rename(columns={'level_1': ColumnsConfig.SCOPE})
                   .set_index('company_id'))
            df2.loc[df2[ColumnsConfig.SCOPE] == 'production', ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS
            df2.loc[df2[ColumnsConfig.SCOPE] != 'production', ColumnsConfig.VARIABLE] = VariablesConfig.EMISSIONS
            df3 = df2.reset_index().set_index(['company_id', 'variable', 'scope'])
            df3 = pd.concat([df3.xs(VariablesConfig.PRODUCTIONS, level=1, drop_level=False)
                .apply(
                lambda x: x.map(lambda y: Q_(float(y) if y is not pd.NA else np.nan,
                                             df_fundamentals.loc[df_fundamentals.company_id == x.name[0],
                                                                 'production_metric'].squeeze())), axis=1),
                df3.xs(VariablesConfig.EMISSIONS, level=1, drop_level=False)
                .apply(lambda x: x.map(
                    lambda y: Q_(float(y) if y is not pd.NA else np.nan,
                                 df_fundamentals.loc[df_fundamentals.company_id == x.name[0],
                                                     'emissions_metric'].squeeze())), axis=1)])
            df4 = df3.xs(VariablesConfig.EMISSIONS, level=1) / df3.xs((VariablesConfig.PRODUCTIONS, 'production'),
                                                                      level=[1, 2])
            df4['variable'] = VariablesConfig.EMISSIONS_INTENSITIES
            df4 = df4.reset_index().set_index(['company_id', 'variable', 'scope'])
            df5 = pd.concat([df3, df4])
            df_historic_data = df5
        else:
            # df_esg = df_esg.iloc[0:45]
            # We are already much tidier, so don't need the wide_to_long conversion.
            df_esg_hasunits = df_esg.unit.notna()
            df_esg_nounits = df_esg[~df_esg_hasunits]
            df_esg = df_esg[df_esg_hasunits]
            esg_year_columns = df_esg.columns[df_esg.columns.get_loc(2016):]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for col in esg_year_columns:
                    qty_col = df_esg.apply(lambda x: Q_(np.nan if pd.isna(x[col]) else float(x[col]), x['unit']), axis=1)
                    df_esg[col] = df_esg[col].astype('object')
                    df_esg.loc[df_esg.index, col] = qty_col
            prod_mask = df_esg.metric == 'production'
            prod_metrics = df_esg[prod_mask].groupby(by=['company_id'])['unit'].agg(lambda x: x.values[0])
            df_fundamentals.loc[prod_metrics.index, ColumnsConfig.PRODUCTION_METRIC] = prod_metrics
            em_metrics = df_esg[~prod_mask]
            em_unit_ambig = em_metrics.groupby(by=['company_id', 'metric']).count()
            em_unit_ambig = em_unit_ambig[em_unit_ambig.unit>1]
            if len(em_unit_ambig)>0:
                em_unit_ambig = em_unit_ambig.reset_index('metric').drop(columns='unit')
                for id in em_unit_ambig.index.unique():
                    logger.warning(f"Company {id} uses multiple units describing scopes {[s for s in em_unit_ambig.loc[[id]]['metric']]}")
                logger.warning(f"The ITR Tool will choose one and covert all to that")
            else:
                em_metrics.metrics = 'emissions'
                em_unit_ambig = em_metrics.groupby(by=['company_id', 'metric']).count()
                em_unit_ambig = em_unit_ambig[em_unit_ambig.unit>1]
                if len(em_unit_ambig)>0:
                    em_unit_ambig = em_unit_ambig.droplevel('metric')
                    for id in em_unit_ambig.index.unique():
                        logger.warning(f"Company {id} uses multiple units describing different scopes {[s for s in em_unit_ambig.loc[[id]]['unit']]}")
                    logger.warning(f"The ITR Tool will choose one and covert all to that")

            em_units = em_metrics.groupby(by=['company_id'], group_keys=True).first()
            df_fundamentals.loc[em_units.index, ColumnsConfig.EMISSIONS_METRIC] = em_units.unit

            grouped_prod = (
                df_esg[df_esg.metric.isin(['production'])].drop(columns=['unit', 'report_date'])
                # first collect things together down to sub-metric category
                .fillna(np.nan)
                .groupby(by=[ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID, 'metric', 'submetric'],
                         dropna=False)[esg_year_columns]
                # then estimate values for each submetric (many/most of which will be NaN)
                .agg(_estimated_value)
                .reset_index(level='submetric')
                )

            # Now prioritize the submetrics we want: the best non-NaN values for each company in each column
            grouped_prod.submetric = pd.Categorical(grouped_prod['submetric'], ordered=True, categories=['equity', '', 'gross', 'net', 'full'])
            best_prod = (
                grouped_prod.sort_values('submetric')
                .groupby(by=[ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID, 'metric'])
                .agg(lambda x:x)
                .apply(prioritize_submetric, axis=1)
            )
            best_prod = best_prod.drop(columns='submetric')
            best_prod[ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS

            s3_lookup_index = df_esg[df_esg.metric.str.lower().eq('s3') & df_esg.submetric.str.lower().isin(s3_category_dict)].index
            df_esg.loc[s3_lookup_index, 'submetric'] = df_esg.loc[s3_lookup_index].submetric.str.lower().map(s3_category_dict)
            grouped_em = (
                df_esg.loc[em_metrics.index].drop(columns=['unit', 'report_date'])
                .assign(metric=df_esg.metric.str.upper())
                # first collect things together down to sub-metric category
                .fillna(np.nan)
                .groupby(by=[ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID, 'metric', 'submetric'],
                         dropna=False)[esg_year_columns]
                # then estimate values for each submetric (many/most of which will be NaN)
                .agg(_estimated_value)
                .reset_index(level='submetric')
                )

            # Now prioritize the submetrics we want: the best non-NaN values for each company in each column
            grouped_em.submetric = pd.Categorical(grouped_em['submetric'], ordered=True, categories=['', 'all', 'combined', 'total', 'location', 'market'])
            best_em = (
                grouped_em.sort_values('submetric')
                .groupby(by=[ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID, 'metric'])
                .agg(lambda x:x)
                .apply(prioritize_submetric, axis=1)
                .drop(columns='submetric')
            )
            em_all_nan = best_em.apply(lambda x: x.map(lambda y: ITR.isnan(y.m)).all(), axis=1)
            missing_em = best_em[em_all_nan]
            if len(missing_em):
                logger.warning(f"Emissions data missing for {missing_em.index}") 
                best_em = best_em[~em_all_nan].copy()
            best_em[ColumnsConfig.VARIABLE]=VariablesConfig.EMISSIONS
            df3 = pd.concat([best_prod, best_em]).reset_index(level='metric').rename(columns={'metric':'scope'}).set_index([ColumnsConfig.VARIABLE, 'scope'], append=True)
            # XS is how we match labels in indexes.  Here 'variable' is level=2, (company_name=0, company_id=1)
            # By knocking out 'production', we don't get production / production in the calculations, only emissions (all scopes in data) / production
            df4 = df3.xs(VariablesConfig.EMISSIONS, level=2) / df3.xs((VariablesConfig.PRODUCTIONS, 'production'), level=[2,3])
            df4['variable'] = VariablesConfig.EMISSIONS_INTENSITIES
            df4 = df4.reset_index().set_index(['company_name', 'company_id', 'variable', 'scope'])
            df5 = pd.concat([df3, df4]).droplevel(level='company_name')
            df_historic_data = df5
            
        # df_historic now ready for conversion to model for each company
        self.historic_years = [column for column in df_historic_data.columns if type(column) == int]

        def get_scoped_df(df, mask, names=None):
            return df.loc[mask[mask].index].set_index(names)

        df = df_historic_data.reset_index()
        index_names = ['company_id', 'variable']
        mask_nonprod = df.variable.ne('Productions')
        mask_s1 = df[mask_nonprod].scope.eq('S1')
        mask_s2 = df[mask_nonprod].scope.eq('S2')
        mask_s1s2 = df[mask_nonprod].scope.eq('S1S2')
        mask_s3 = df[mask_nonprod].scope.eq('S3')
        mask_s1s2s3 = df[mask_nonprod].scope.eq('S1S2S3')

        tmp_s1s2 = df.loc[mask_s1s2[mask_s1s2].index, (df.columns[3:])].apply(lambda x: all([np.isnan(y.m) for y in x]), axis=1)
        idx_s1s2 = df.loc[tmp_s1s2[tmp_s1s2].index].index
        df_s1 = get_scoped_df(df, mask_s1, index_names)
        df_s2 = get_scoped_df(df, mask_s2, index_names)
        midx_s1s2 = pd.MultiIndex.from_tuples(zip(df.loc[idx_s1s2].company_id, df.loc[idx_s1s2].variable),
                                              names=index_names)
        df.loc[idx_s1s2, self.historic_years] = (df_s1.loc[midx_s1s2, self.historic_years]
                                                 .add(df_s2.loc[midx_s1s2, self.historic_years])
                                                 .set_axis(idx_s1s2))
        
        tmp_s1s2s3 = df.loc[mask_s1s2s3[mask_s1s2s3].index, (df.columns[3:])].apply(lambda x: all([np.isnan(y.m) for y in x]), axis=1)
        idx_s1s2s3 = df.loc[tmp_s1s2s3[tmp_s1s2s3].index].index
        df_s1s2 = get_scoped_df(df, mask_s1s2, index_names)
        df_s3 = get_scoped_df(df, mask_s3, index_names)
        midx_s1s2s3 = pd.MultiIndex.from_tuples(zip(df.loc[idx_s1s2s3].company_id, df.loc[idx_s1s2s3].variable),
                                                names=index_names)
        df.loc[idx_s1s2s3, self.historic_years] = (df_s1s2.loc[midx_s1s2s3, self.historic_years]
                                                   .add(df_s3.loc[midx_s1s2s3, self.historic_years])
                                                   .set_axis(idx_s1s2s3))

        df_historic_data = df.set_index(['company_id', 'variable', 'scope'])
        
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

        c_ids_invalid_netzero_year = list(target_data[target_data['netzero_year'] > ProjectionControls.TARGET_YEAR].index)
        if c_ids_invalid_netzero_year:
            error_message = f"Invalid net-zero target years (>{ProjectionControls.TARGET_YEAR}) are entered for companies with ID: " \
                            f"{c_ids_invalid_netzero_year}"
            logger.error(error_message)
            raise ValueError(error_message)
        target_data.loc[target_data.netzero_year.isna(), 'netzero_year'] = ProjectionControls.TARGET_YEAR

        c_ids_with_increase_target = list(target_data[target_data['target_reduction_ambition'] < 0].index)
        if c_ids_with_increase_target:
            error_message = f"Negative target reduction ambition is invalid and entered for companies with ID: " \
                            f"{c_ids_with_increase_target}"
            logger.error(error_message)
            raise ValueError(error_message)

        return target_data

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
                    # FIXME: Is this the best place to finalize base_year_production, ghg_s1s2, and ghg_s3 data?
                    # Something tells me these parameters should be removed in favor of querying historical data directly
                    if not ColumnsConfig.BASE_YEAR_PRODUCTION in company_data:
                        company_data[ColumnsConfig.BASE_YEAR_PRODUCTION] = df_historic_data.loc[
                            company_data[ColumnsConfig.COMPANY_ID], 'Productions', 'production'][
                                TemperatureScoreConfig.CONTROLS_CONFIG.base_year]
                    if not ColumnsConfig.GHG_SCOPE12 in company_data:
                        company_data[ColumnsConfig.GHG_SCOPE12] = df_historic_data.loc[
                            company_data[ColumnsConfig.COMPANY_ID], 'Emissions', 'S1S2'][
                                TemperatureScoreConfig.CONTROLS_CONFIG.base_year]
                    try:
                        if not ColumnsConfig.GHG_SCOPE3 in company_data:
                            company_data[ColumnsConfig.GHG_SCOPE3] = df_historic_data.loc[
                                company_data[ColumnsConfig.COMPANY_ID], 'Emissions', 'S3'][
                                    TemperatureScoreConfig.CONTROLS_CONFIG.base_year]
                    except KeyError:
                        # If there was no relevant historic data, don't try to use it
                        pass
                    company_data[ColumnsConfig.HISTORIC_DATA] = self._convert_historic_data(
                        df_historic_data.loc[[company_data[ColumnsConfig.COMPANY_ID]]].reset_index()).dict()
                else:
                    company_data[ColumnsConfig.HISTORIC_DATA] = None

                if df_target_data is not None and company_id in df_target_data.index:
                    company_data[ColumnsConfig.TARGET_DATA] = [td.dict() for td in self._convert_target_data(
                        df_target_data.loc[[company_data[ColumnsConfig.COMPANY_ID]]].reset_index())]
                else:
                    company_data[ColumnsConfig.TARGET_DATA] = None

                # handling of missing market cap data is mainly done in _convert_from_template_company_data()
                if company_data[ColumnsConfig.COMPANY_MARKET_CAP] is pd.NA:
                    company_data[ColumnsConfig.COMPANY_MARKET_CAP] = np.nan

                model_companies.append(ICompanyData.parse_obj(company_data))
            except ValidationError:
                logger.error(f"(One of) the input(s) of company with ID {company_id} is invalid")
                raise
        return model_companies

    # Workaround for bug (https://github.com/pandas-dev/pandas/issues/20824) in Pandas where NaN are treated as zero 
    def _np_sum(g):
        return np.sum(g.values)

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
        :param historic_data: Dataframe Productions, Emissions, and Emissions Intensities mixed together
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

    # In the following several methods, we implement SCOPE as STRING (used by Excel handlers)
    # so that the resulting scope dictionary can be used to pass values to named arguments
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
        for scope_name in EScope.get_scopes():
            results = emissions.loc[emissions[ColumnsConfig.SCOPE] == scope_name]
            emissions_scopes[scope_name] = [] \
                if results.empty \
                else [IEmissionRealization(year=year, value=EmissionsQuantity(results[year].squeeze())) for year in self.historic_years]
        return IHistoricEmissionsScopes(**emissions_scopes)

    def _convert_to_historic_productions(self, productions: pd.DataFrame) -> Optional[List[IProductionRealization]]:
        """
        :param productions: historic production data for a company
        :return: A list containing historic productions, or None if no data are provided
        """
        if productions.empty:
            return None
        return [IProductionRealization(year=year, value=ProductionQuantity(productions[year].squeeze())) for year in self.historic_years]

    def _convert_to_historic_ei(self, intensities: pd.DataFrame) -> Optional[IHistoricEIScopes]:
        """
        :param intensities: historic emission intensity data for a company
        :return: A list of historic emission intensities per scope, or None if no data are provided
        """
        if intensities.empty:
            return None

        intensities = intensities.copy()
        intensity_scopes = {}

        for scope_name in EScope.get_scopes():
            results = intensities.loc[intensities[ColumnsConfig.SCOPE] == scope_name]
            intensity_scopes[scope_name] = [] \
                if results.empty \
                else [IEIRealization(year=year, value=EI_Quantity(results[year].squeeze())) for year in self.historic_years]
        return IHistoricEIScopes(**intensity_scopes)
