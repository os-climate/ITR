import warnings  # needed until apply behaves better with Pint quantities in arrays
from typing import Type, List, Optional
import pandas as pd
import numpy as np

from pydantic import ValidationError

import ITR
from ITR.data.osc_units import ureg, Q_, PA_, ProductionQuantity, EmissionsQuantity, EI_Quantity, asPintSeries, asPintDataFrame, ProductionMetric, EmissionsMetric
import pint

from ITR.data.base_providers import BaseCompanyDataProvider
from ITR.configs import ColumnsConfig, TemperatureScoreConfig, VariablesConfig, TabsConfig, SectorsConfig, ProjectionControls, LoggingConfig

import logging
logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)

from ITR.interfaces import ICompanyData, EScope, IHistoricEmissionsScopes, IProductionRealization, \
    IHistoricEIScopes, IHistoricData, ITargetData, IEmissionRealization, IEIRealization, \
    IProjection
from ITR.utils import get_project_root

pkg_root = get_project_root()
df_country_regions = pd.read_csv(f"{pkg_root}/data/input/country_region_info.csv")

def ITR_country_to_region(country:str) -> str:
    if not isinstance(country, str):
        if np.isnan(country):
            return 'Global'
        raise ValueError(f"ITR_country_to_region received invalid valud {country}")
    if len(country)==2:
        if country=='UK':
            country='GB'
        regions = df_country_regions[df_country_regions.alpha_2==country].region_ar6_10
    elif len(country)==3:
        regions = df_country_regions[df_country_regions.alpha_3==country].region_ar6_10
    else:
        if country in df_country_regions.name.values:
            regions = df_country_regions[df_country_regions.name==country].region_ar6_10
        elif country in df_country_regions.common_name.values:
            regions = df_country_regions[df_country_regions.common_name==country].region_ar6_10
        elif country == 'Great Britain':
            return 'Europe'
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
            raise ValueError
        # This relies on the fact that we can now see Quantity(np.nan, ...) for both float and ufloat magnitudes
        # remove NaNs, which mess with mean estimation
        x = y[~ITR.isnan(PA_._from_sequence(y).quantity.m)]
    except TypeError:
        # FIXME: can get here if we have garbage units in one row (such as 't/MWh') that don't match another ('t CO2e/MWh')
        # Need to deal with that...
        logger.error(f"type_error({y}), so returning {y.values}[0]")
        return y.iloc[0]
    if len(x) == 0:
        # If all inputs are NaN, return the first NaN
        return y.iloc[0]
    if len(x) == 1:
        # If there's only one non-NaN input, return that one
        return x.iloc[0]
    if isinstance(x.values[0], pint.Quantity):
        # Let PintArray do all the work of harmonizing compatible units
        x = PA_._from_sequence(x)
        if ITR.HAS_UNCERTAINTIES:
            wavg = ITR.umean(x.quantity.m)
        else:
            wavg = np.mean(x.quantity.m)
        est = Q_(wavg, x.quantity.u)
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
        The index of the series is (SECTOR, COMPANY_ID, SCOPE)
        The first element of the pd.Series is the list of submetrics in priority order; it is a categorical type
        All subsequent elements are observations to be selected from.

    Returns
    -------
    y : The first non-NaN quantity using the given priority order.
    """
    if isinstance(x.submetric, float) or isinstance(x.submetric, str):
        # Nothing to prioritize
        return x
    y = x.copy()
    if x.name[0] == 'Electricity Utiltiies':
        for p in range(0, len(x.submetric)):
            if x.submetric[p] == '3':
                pinned_priority = p
                break
    elif x.name[0] in ['Autos', 'Oil & Gas']:
        for p in range(0, len(x.submetric)):
            if x.submetric[p] == '11':
                pinned_priority = p
                break
    else: 
        pinned_priority = None

    for c in range(1,len(x)):
        if not isinstance(x.iloc[c], np.ndarray):
            # If we don't have a list to prioritize, don't try
            continue
        if pinned_priority:
            if not ITR.isnan(x.iloc[c][pinned_priority].m):
                # Replace array to be prioritized with pinned choice as best
                y.iloc[c] = x.iloc[c][pinned_priority]
                continue
        for p in range(0, len(x.submetric)):
            if not ITR.isnan(x.iloc[c][p].m):
                # Replace array to be prioritized with best choice
                y.iloc[c] = x.iloc[c][p]
                break
        if isinstance(y.iloc[c], np.ndarray):
            # If we made it to the end without replacing, arbitrarily pick the first (nan) value
            assert ITR.isnan(y.iloc[c][0].m)
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
    """

    def __init__(self, excel_path: str,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 projection_controls: Type[ProjectionControls] = ProjectionControls):
        self.template_v2_start_year = None
        self.projection_controls = projection_controls
        # The initial population of companies' data
        self._companies = self._init_from_template_company_data(excel_path)
        super().__init__(self._companies, column_config, projection_controls)
        # The perfection of historic ESG data
        self._companies = self._convert_from_template_company_data()

    # When rows of data are expressed in terms of scope intensities, solve for the implied production 
    def _solve_intensities(self, df_fundamentals: pd.DataFrame, df_esg: pd.DataFrame) -> pd.DataFrame:
        # We have organized all emissions and productions data given, and inferred intensities from that.
        # But we may yet have been giving emissions and emissions intensities without production numbers,
        # or emissions intensities and production numbers without absolute emissions numbers.
        # We calculate those and append to the result.
        df_esg_has_intensity = df_esg[df_esg.metric.str.contains('intensity')]
        df_esg_has_intensity = df_esg_has_intensity.assign(scope=lambda x: list(map(lambda y: y[0], x['metric'].str.split(' '))))
        missing_production_idx = df_fundamentals.index.difference(df_esg[df_esg.metric.eq('production')].company_id.unique())
        df_esg_missing_production = df_esg[df_esg['company_id'].isin(missing_production_idx)]
        start_year_loc = df_esg.columns.get_loc(self.template_v2_start_year)
        if len(df_esg_missing_production):
            df_intensities = df_esg_has_intensity.rename(columns={'metric':'intensity_metric', 'scope':'metric'})
            df_intensities = df_intensities.reset_index(drop=False).set_index(['company_id', 'metric', 'report_date'])
            df_emissions = df_esg_missing_production.set_index(['company_id', 'metric', 'report_date'])
            common_idx = df_emissions.index.intersection(df_intensities.index)
            df_emissions = df_emissions.loc[common_idx]
            df_intensities = df_intensities.loc[common_idx]
            df_emissions.loc[:, 'index'] = df_intensities.loc[:, 'index']
            df_intensities = df_intensities.set_index('index', append=True).iloc[:, (start_year_loc-2):]
            df_emissions = df_emissions.set_index('index', append=True).iloc[:, (start_year_loc-3):]
            df_intensities_t = asPintDataFrame(df_intensities.T)
            df_emissions_t = asPintDataFrame(df_emissions.T)
            df_productions_t = df_emissions_t.divide(df_intensities_t, axis=1)
            # FIXME: need to reconcile what happens if multiple scopes all yield the same production metrics
            df_productions_t = df_productions_t.apply(lambda x: x.astype(f"pint[{ureg(str(x.dtype)[5:-1]).to_reduced_units().u}]")).dropna(axis=1, how='all')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # pint units don't like being twisted from columns to rows, but it's ok
                df_productions = df_productions_t.T.droplevel(['company_id', 'metric', 'report_date'])
            production_idx = df_productions.index
            df_esg.loc[production_idx] = pd.concat([df_esg.loc[production_idx].iloc[:, 0:start_year_loc], df_productions], axis=1)
            df_esg.loc[production_idx, 'metric'] = 'production'
            # Set production metrics we just derived (if necessary)
            for idx in production_idx:
                if pd.isna(df_fundamentals.loc[df_esg.loc[idx].company_id, ColumnsConfig.PRODUCTION_METRIC]):
                    df_fundamentals.loc[df_esg.loc[idx].company_id, ColumnsConfig.PRODUCTION_METRIC] = str(df_esg.loc[idx].iloc[-1].u)
        df_esg_has_intensity = df_esg[df_esg.metric.str.contains('intensity')]
        df_esg_has_intensity = df_esg_has_intensity.assign(scope=lambda x: list(map(lambda y: y[0], x['metric'].str.split(' '))))
        if len(df_esg_has_intensity):
            # Convert EI to Emissions using production
            # Save index...we'll need it later to bring values back to df_esg
            # Hide all our values except those we need to use/join with later
            df1 = df_esg_has_intensity.reset_index().set_index(['company_id', 'report_date', 'scope', 'index']).iloc[:, (start_year_loc-2):]
            df2 = df_esg[df_esg.metric.eq('production')].set_index(['company_id', 'report_date'])
            df3 = df1.multiply(df2.iloc[:,(start_year_loc-2):].loc[df1.index.droplevel(['scope', 'index'])])
            if ITR.HAS_UNCERTAINTIES:
                df4 = df3.astype('pint[t CO2e]') # .drop_duplicates() # When we have uncertainties, multiple observations influence the observed error term
                # Also https://github.com/pandas-dev/pandas/issues/12693
            else:
                df4 = df3.astype('pint[t CO2e]').drop_duplicates()
            df5 = df4.droplevel(['company_id', 'report_date']).swaplevel() # .sort_index()
            df_esg.loc[df5.index.get_level_values('index'), 'metric'] = df5.index.get_level_values('scope')
            df5.index = df5.index.droplevel('scope')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Change from emissions intensity metrics to emissions metrics; we know this will trigger a warning
                df_esg.loc[df5.index, df_esg.columns[start_year_loc:].to_list()] = df5[df_esg.columns[start_year_loc:].to_list()]
            # Set emissions metrics we just derived (if necessary), both in df_fundamentals (for targets, globally) and df_esg (for projections etc)
            for idx in df5.index:
                if pd.isna(df_fundamentals.loc[df_esg.loc[idx].company_id, ColumnsConfig.EMISSIONS_METRIC]):
                    df_fundamentals.loc[df_esg.loc[idx].company_id, ColumnsConfig.EMISSIONS_METRIC] = str(df_esg.loc[idx].iloc[-1].u)
                df_esg.loc[idx, 'unit'] = str(df_esg.loc[idx].iloc[-1].u)
        return df_esg

    def _init_from_template_company_data(self, excel_path: str):
        """
        Converts first sheet of Excel template to list of minimal ICompanyData objects (fundamental data, but no ESG data).
        All dataprovider features will be inhereted from Base.
        :param excel_path: file path to excel file
        """

        self.template_version = 1

        df_company_data = pd.read_excel(excel_path, sheet_name=None, skiprows=0)

        if TabsConfig.TEMPLATE_INPUT_DATA_V2 and TabsConfig.TEMPLATE_INPUT_DATA_V2 in df_company_data:
            self.template_version = 2
            input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA_V2
        else:
            input_data_sheet = TabsConfig.TEMPLATE_INPUT_DATA
        try:
            df = df_company_data[input_data_sheet]
        except KeyError as e:
            logger.error(f"Tab {input_data_sheet} is required in input Excel file.")
            raise KeyError
        if self.template_version==2:
            esg_data_sheet = TabsConfig.TEMPLATE_ESG_DATA_V2
            try:
                df_esg = df_company_data[esg_data_sheet].drop(columns='company_lei').copy() # .iloc[0:45]
            except KeyError as e:
                logger.error(f"Tab {esg_data_sheet} is required in input Excel file.")
                raise KeyError
            if 'base_year' in df_esg.columns:
                self.template_v2_start_year = df_esg.columns[df_esg.columns.get_loc('base_year')+1]
            else:
                self.template_v2_start_year = df_esg.columns[df_esg.columns.map(lambda col: isinstance(col, int))][0]
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
                    category=FutureWarning,
                    message=(
                        ".*will attempt to set the values inplace instead of always setting a new array. "
                        "To retain the old behavior, use either.*"
                    ),
                )

                df_esg.loc[:, [ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID, ColumnsConfig.TEMPLATE_REPORT_DATE]] = (
                    df_esg[[ColumnsConfig.COMPANY_NAME, ColumnsConfig.COMPANY_ID, ColumnsConfig.TEMPLATE_REPORT_DATE]].fillna(method='ffill')
                )

        # NA in exposure is how we drop rows we want to ignore
        df = df[df.exposure.notna()]

        # TODO: Fix market_cap column naming inconsistency
        df.rename(
            columns={'revenue': 'company_revenue', 'market_cap': 'company_market_cap',
                     'ev': 'company_enterprise_value', 'evic': 'company_ev_plus_cash',
                     'assets': 'company_total_assets'}, inplace=True)
        df.loc[df.region.isnull(), 'region'] = df.country.map(ITR_country_to_region)

        df_fundamentals = df.set_index(ColumnsConfig.COMPANY_ID, drop=False).convert_dtypes()

        # testing if all data is in the same currency
        if ColumnsConfig.TEMPLATE_FX_QUOTE in df_fundamentals.columns:
            fx_quote = df_fundamentals[ColumnsConfig.TEMPLATE_FX_QUOTE].notna()
            if len(df_fundamentals.loc[~fx_quote, ColumnsConfig.TEMPLATE_CURRENCY].unique()) > 1 \
               or len(df_fundamentals.loc[fx_quote, ColumnsConfig.TEMPLATE_FX_QUOTE].unique()) > 1 \
               or (fx_quote.any() and (~fx_quote).any() and not \
                   (df_fundamentals.loc[~fx_quote, ColumnsConfig.TEMPLATE_CURRENCY].iloc[0] == \
                    df_fundamentals.loc[fx_quote, ColumnsConfig.TEMPLATE_FX_QUOTE].iloc[0])):
                error_message = f"All data should be in the same currency."
                logger.error(error_message)
                raise ValueError(error_message)
            elif fx_quote.any():
                fundamental_metrics = ['company_market_cap', 'company_revenue', 'company_enterprise_value', 'company_ev_plus_cash', 'company_total_assets']
                col_num = df_fundamentals.columns.get_loc('report_date')
                missing_fundamental_metrics = [fm for fm in fundamental_metrics if fm not in df_fundamentals.columns[col_num+1:]]
                if len(missing_fundamental_metrics)>0:
                    raise KeyError(f"Expected fundamental metrics {missing_fundamental_metrics}")
                for col in fundamental_metrics:
                    df_fundamentals[col] = df_fundamentals[col].astype('Float64')
                    df_fundamentals[f"{col}_base"] = df_fundamentals[col]
                    df_fundamentals.loc[fx_quote, col] = df_fundamentals.loc[fx_quote, 'fx_rate'] * df_fundamentals.loc[fx_quote, f"{col}_base"]
                # create context for currency conversions
                # df_fundamentals defines 'report_date', 'currency', 'fx_quote', and 'fx_rate'
                # our base currency is USD, but european reports may be denominated in EUR.  We crosswalk from report_base to pint_base currency.

                def convert_prefix_to_scalar(x):
                    try:
                        prefix, unit, suffix = ureg.parse_unit_name(x)[0]
                    except IndexError:
                        raise f"Currency {x} is not in the registry; please update registry or remove data row"
                    return unit, ureg._prefixes[prefix].value if prefix else 1.0

                fx_df = df_fundamentals.loc[fx_quote, ['report_date', 'currency', 'fx_quote', 'fx_rate']].drop_duplicates().set_index('report_date')
                fx_df = fx_df.assign(currency_tuple=lambda x: x['currency'].map(convert_prefix_to_scalar),
                                     fx_quote_tuple=lambda x: x['fx_quote'].map(convert_prefix_to_scalar))
                self.fx = pint.Context('FX')
                # FIXME: These simple rules don't take into account different conversion rates at different time periods.
                fx_df.apply(lambda x: self.fx.redefine(f"{x.currency_tuple[0]} = {x.fx_rate * x.fx_quote_tuple[1] / x.currency_tuple[1]} {x.fx_quote_tuple[0]}")
                            if x.currency != 'USD' else self.fx.redefine(f"{x.fx_quote_tuple[0]} = {x.currency_tuple[1]/(x.fx_rate * x.fx_quote_tuple[1])} {x.currency_tuple[0]}"),
                            axis=1)
                ureg.add_context(self.fx)
                ureg.enable_contexts('FX')
            else:
                self.fx = None
        else:
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
        df_fundamentals.drop(columns=['MCap_to_Reven','MCap_to_Assets','AVG_MCap_to_Reven','AVG_MCap_to_Assets'], inplace=True) # deleting temporary columns
        
        if missing_cap_ids:
            logger.warning(f"Missing market capitalisation values are estimated for companies with ID: "
                           f"{missing_cap_ids}.")

        test_target_sheet = TabsConfig.TEMPLATE_TARGET_DATA
        try:
            self.df_target_data = df_company_data[test_target_sheet].set_index('company_id').convert_dtypes()
        except KeyError:
            logger.error(f"Tab {test_target_sheet} is required in input Excel file.")
            raise

        if self.template_version==1:
            # ignore company data that does not come with emissions and/or production metrics
            missing_esg_metrics_df = df_fundamentals[ColumnsConfig.COMPANY_ID][
                df_fundamentals[ColumnsConfig.EMISSIONS_METRIC].isnull() | df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].isnull()]
            if len(missing_esg_metrics_df)>0:
                logger.warning(f"Missing ESG metrics for companies with ID (will be ignored): "
                               f"{missing_esg_metrics_df.index}.")
                df_fundamentals = df_fundamentals[~df_fundamentals.index.isin(missing_esg_metrics_df.index)]
            # Template V1 does not use df_esg
            # self.df_esg = None
        else:
            # df_esg = df_esg.iloc[0:45]

            # Disable rows we do not yet handle
            df_esg = df_esg[~ df_esg.metric.isin(['generation', 'consumption'])]
            if ColumnsConfig.BASE_YEAR in df_esg.columns:
                df_esg = df_esg[df_esg.base_year.str.lower().ne('x')]
            # FIXME: Should we move more df_esg work up here?
            self.df_esg = df_esg
        self.df_fundamentals = df_fundamentals
        # We don't want to process historic and target data yet
        return self._company_df_to_model(df_fundamentals, pd.DataFrame(), pd.DataFrame())

    def _convert_from_template_company_data(self) -> List[ICompanyData]:
        """
        Converts ESG sheet of Excel template to flesh out ICompanyData objects.
        :return: List of ICompanyData objects
        """
        df_fundamentals = self.df_fundamentals
        df_target_data = self.df_target_data
        if self.template_version > 1:
            df_esg = self.df_esg

        def _fixup_name(x):
            prefix, _, suffix = x.partition('_')
            suffix = suffix.replace('ghg_', '')
            if suffix != 'production':
                suffix = suffix.upper()
            return f"{suffix}-{prefix}"
        
        if self.template_version==1:
            # The nightmare of naming columns 20xx_metric instead of metric_20xx...and potentially dealing with data from 1990s...
            historic_columns = [col for col in df_fundamentals.columns if col[:1].isdigit()]
            historic_scopes = ['S1', 'S2', 'S3', 'S1S2', 'S1S2S3', 'production']
            df_historic = df_fundamentals[['company_id'] + historic_columns].dropna(axis=1, how='all')
            df_fundamentals = df_fundamentals[df_fundamentals.columns.difference(historic_columns, sort=False)]

            df_historic = df_historic.rename(columns={col: _fixup_name(col) for col in historic_columns})
            df = pd.wide_to_long(df_historic, historic_scopes, i='company_id', j='year', sep='-',
                                 suffix=r'\d+').reset_index()
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
            # We are already much tidier, so don't need the wide_to_long conversion.
            esg_year_columns = df_esg.columns[df_esg.columns.get_loc(self.template_v2_start_year):]
            df_esg_hasunits = df_esg.unit.notna()
            df_esg_badunits = df_esg[df_esg_hasunits].unit.map(lambda x: x not in ureg)
            badunits_idx = df_esg_badunits[df_esg_badunits].index
            if df_esg_badunits.any():
                logger.error(f"The following row of data contain units that are not in the registry and will be removed from analysis\n{df_esg.loc[badunits_idx]}")
                df_esg_hasunits.loc[badunits_idx] = False
            df_esg = df_esg[df_esg_hasunits]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_col = df_esg['unit']
                for col in esg_year_columns:
                    m_col = df_esg[col].map(lambda x: np.nan if pd.isna(x) else float(x))
                    df_esg[col] = m_col.combine(u_col, lambda m, u: Q_(m, u))
            prod_mask = df_esg.metric == 'production'
            prod_metrics = df_esg[prod_mask].groupby(by=['company_id'])['unit'].agg(lambda x: x.values[0])
            # We update the metrics we were told with the metrics we are given
            df_fundamentals.loc[prod_metrics.index, ColumnsConfig.PRODUCTION_METRIC] = prod_metrics
            em_metrics = df_esg[df_esg.metric.str.upper().isin(['S1', 'S2', 'S3', 'S1S2', 'S1S3', 'S1S2S3'])]
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
            # We update the metrics we were told with the metrics we are given
            df_fundamentals.loc[em_units.index, ColumnsConfig.EMISSIONS_METRIC] = em_units.unit

            # We solve while we still have valid report_date data.  After we group reports together to find the "best"
            # the report_date becomes meaningless (and is dropped by _solve_intensities)
            df_esg = self._solve_intensities(df_fundamentals, df_esg)

            # After this point we can gripe if missing emissions and/or production metrics
            missing_esg_metrics_df = df_fundamentals[ColumnsConfig.COMPANY_ID][
                df_fundamentals[ColumnsConfig.EMISSIONS_METRIC].isnull() | df_fundamentals[ColumnsConfig.PRODUCTION_METRIC].isnull()]
            if len(missing_esg_metrics_df)>0:
                logger.warning(f"Missing ESG metrics for companies with ID (will be ignored): "
                               f"{missing_esg_metrics_df.index}.")
                df_fundamentals = df_fundamentals[~df_fundamentals.index.isin(missing_esg_metrics_df.index)]
                df_esg = df_esg[~df_esg.company_id.isin(missing_esg_metrics_df.index)]

            # Recalculate if any of the above dropped rows from df_esg
            em_metrics = df_esg[df_esg.metric.str.upper().isin(['S1', 'S2', 'S3', 'S1S2', 'S1S3', 'S1S2S3'])]

            # Validate that all our em_metrics are, in fact, some kind of emissions quanity
            em_invalid = df_esg.loc[em_metrics.index].unit.map(lambda x: not isinstance(x, str) or not ureg(x).is_compatible_with('t CO2'))
            em_invalid_idx = em_invalid[em_invalid].index
            if len(em_invalid_idx)>0:
                logger.error(f"The following rows of data do not have proper emissions data (can be converted to t CO2e) and will be dropped from the analysis\n{df_esg.loc[em_invalid_idx]}")
                df_esg = df_esg.loc[df_esg.index.difference(em_invalid_idx)]
                em_metrics = em_metrics.loc[em_metrics.index.difference(em_invalid_idx)]

            # We don't need units here anymore--they've been translated/transported everywhere we need them
            df_esg.drop(columns='unit', inplace=True)

            grouped_prod = (
                df_esg[df_esg.metric.isin(['production'])]
                .assign(sector=df_esg['company_id'].map(lambda x: df_fundamentals.loc[x].sector))
                # first collect things together down to sub-metric category
                .fillna(np.nan)
                .groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, 'metric', 'submetric'],
                         dropna=False)[esg_year_columns]
                # then estimate values for each submetric (many/most of which will be NaN)
                .agg(_estimated_value)
                .reset_index(level='submetric')
                )

            # Now prioritize the submetrics we want: the best non-NaN values for each company in each column
            # For categoricals, fisrt listed is least and sorts to first in ascending order
            grouped_prod.submetric = pd.Categorical(grouped_prod['submetric'], ordered=True, categories=['operated', 'own', 'generation', 'equity', '', 'gross', 'net', 'full'])
            best_prod = (
                grouped_prod.sort_values('submetric')
                .groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, 'metric'])
                .agg(lambda x:x)
                .apply(prioritize_submetric, axis=1)
            )
            best_prod = best_prod.drop(columns='submetric')
            best_prod[ColumnsConfig.VARIABLE] = VariablesConfig.PRODUCTIONS

            # convert "nice" word descriptions of S3 emissions to category numbers
            s3_idx = df_esg.metric.str.upper().eq('S3')
            s3_dict_matches = df_esg[s3_idx].submetric.str.lower().isin(s3_category_dict)
            s3_dict_idx = s3_dict_matches[s3_dict_matches].index
            df_esg.loc[s3_dict_idx, 'submetric'] = df_esg.loc[s3_dict_idx].submetric.str.lower().map(s3_category_dict)

            # We group, in order to prioritize, emissions according to boundary-like and/or category submetrics.
            grouped_em = (
                df_esg.loc[em_metrics.index]
                .assign(metric=df_esg.loc[em_metrics.index].metric.str.upper())
                .assign(submetric=df_esg.loc[em_metrics.index].submetric.map(lambda x: '' if pd.isna(x) else str(x)))
                .assign(sector=df_esg['company_id'].map(lambda x: df_fundamentals.loc[x].sector))
                # first collect things together down to sub-metric category
                .fillna(np.nan)
                .groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, 'metric', 'submetric'],
                         dropna=False)[esg_year_columns]
                # then estimate values for each submetric (many/most of which will be NaN)
                .agg(_estimated_value)
                .reset_index(level='submetric')
                )

            # Now prioritize the submetrics we want: the best non-NaN values for each company in each column
            grouped_non_s3 = grouped_em.loc[grouped_em.index.get_level_values('metric') != 'S3'].copy()
            grouped_non_s3.submetric = pd.Categorical(grouped_non_s3['submetric'], ordered=True,
                                                      categories=['generation', '', 'all', 'combined', 'total', 'location', 'market'])
            best_em = (
                grouped_non_s3.sort_values('submetric')
                .groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, 'metric'])
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

            # We still need to group and prioritize S3 emissions, according to the benchmark requirements
            grouped_s3 = grouped_em.loc[grouped_em.index.get_level_values('metric') == 'S3'].copy()
            grouped_s3.submetric = pd.Categorical(grouped_s3['submetric'], ordered=True,
                                                  categories=['', 'all', 'combined', 'total', '3', '11'])
            best_s3 = (
                grouped_s3.sort_values('submetric')
                .groupby(by=[ColumnsConfig.SECTOR, ColumnsConfig.COMPANY_ID, 'metric'])
                .agg(lambda x:x)
                .apply(prioritize_submetric, axis=1)
                .drop(columns='submetric')
            )
            s3_all_nan = best_s3.apply(lambda x: x.map(lambda y: ITR.isnan(y.m)).all(), axis=1)
            missing_s3 = best_s3[s3_all_nan]
            if len(missing_s3):
                logger.warning(f"Scope 3 Emissions data missing for {missing_s3.index}")
                # We cannot fill in missing data here, because we don't yet know what benchmark(s) will in use
                best_s3 = best_s3[~s3_all_nan].copy()
            best_s3[ColumnsConfig.VARIABLE]=VariablesConfig.EMISSIONS
            df3 = (pd.concat([best_prod, best_em, best_s3])
                   .reset_index(level='metric')
                   .rename(columns={'metric':'scope'})
                   .set_index([ColumnsConfig.VARIABLE, 'scope'], append=True)
                   .droplevel('sector'))
            # XS is how we match labels in indexes.  Here 'variable' is level=1, (company_id=0, scope/production=2)
            # By knocking out 'production', we don't get production / production in the calculations, only emissions (all scopes in data) / production
            df4 = df3.xs(VariablesConfig.EMISSIONS, level=1) / df3.xs((VariablesConfig.PRODUCTIONS, 'production'), level=[1,2])
            df4['variable'] = VariablesConfig.EMISSIONS_INTENSITIES
            df4 = df4.reset_index().set_index(['company_id', 'variable', 'scope'])
            df5 = pd.concat([df3, df4])

            df_historic_data = df5
            
        # df_target_data now ready for conversion to model for each company
        df_target_data = self._validate_target_data(df_target_data)

        # df_historic now ready for conversion to model for each company
        self.historic_years = [column for column in df_historic_data.columns if type(column) == int]

        def get_scoped_df(df, scope, names):
            mask = df.scope.eq(scope)
            return df.loc[mask[mask].index].set_index(names)

        def fill_blank_or_missing_rows(df, scope_a, scope_b, scope_ab, index_names, historic_years):
            # Translate from long format, where each scope is on its own line, to common index
            df_a = get_scoped_df(df, scope_a, index_names)
            df_b = get_scoped_df(df, scope_b, index_names)
            df_ab = get_scoped_df(df, scope_ab, index_names).set_index('scope', append=True)
            # This adds rows of SCOPE_AB data that could be created by adding SCOPE_A and SCOPE_B rows
            new_ab_idx = df_a.index.intersection(df_b.index)
            new_ab = df_a.loc[new_ab_idx, historic_years]+df_b.loc[new_ab_idx, historic_years]
            new_ab.insert(0, 'scope', scope_ab)
            new_ab.set_index('scope', append=True, inplace=True)
            df_ab[df_ab.applymap(lambda x: ITR.isnan(x.m))] = new_ab
            # DF_AB has gaps filled, but not whole new rows that did not exist before
            # Drop rows in NEW_AB already covered by DF_AB and consolidate
            new_ab.drop(index=df_ab.index, inplace=True, errors='ignore')
            # Now update original dataframe with our new data
            df.set_index(index_names+['scope'], inplace=True)
            df.update(df_ab)
            df = pd.concat([df, new_ab]).reset_index()
            return df

        df = df_historic_data.reset_index()
        index_names = ['company_id', 'variable']
        df = fill_blank_or_missing_rows(df, 'S1', 'S2', 'S1S2', index_names, self.historic_years)
        df = fill_blank_or_missing_rows(df, 'S1S2', 'S3', 'S1S2S3', index_names, self.historic_years)
        df_historic_data = df.set_index(['company_id', 'variable', 'scope'])
        # We might run `fill_blank_or_missing_rows` again if we get newly estimated S3 data from an as-yet unknown benchmark
        
        # Drop from our companies list the companies dropped in df_fundamentals
        self._companies = [c for c in self._companies if c.company_id in df_fundamentals.index]
        for company in self._companies:
            row = df_fundamentals.loc[company.company_id]
            company.emissions_metric = EmissionsMetric(row[ColumnsConfig.EMISSIONS_METRIC])
            company.production_metric = ProductionMetric(row[ColumnsConfig.PRODUCTION_METRIC])
        # And keep df_fundamentals in sync
        self.df_fundamentals = df_fundamentals

        # company_id, netzero_year, target_type, target_scope, target_start_year, target_base_year, target_base_year_qty, target_base_year_unit, target_year, target_reduction_ambition
        return self._company_df_to_model(None, df_target_data, df_historic_data)

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

        c_ids_with_nonnumeric_target = list(target_data[target_data['target_reduction_ambition'].map(lambda x: isinstance(x, str))].index)
        if c_ids_with_nonnumeric_target:
            error_message = f"Non-numeric target reduction ambition is invalid; please fix companies with ID: " \
                            f"{c_ids_with_nonnumeric_target}"
            logger.error(error_message)
            raise ValueError(error_message)
        c_ids_with_increase_target = list(target_data[target_data['target_reduction_ambition'] < 0].index)
        if c_ids_with_increase_target:
            error_message = f"Negative target reduction ambition is invalid and entered for companies with ID: " \
                            f"{c_ids_with_increase_target}"
            logger.error(error_message)
            raise ValueError(error_message)

        # https://stackoverflow.com/a/74193599/1291237
        with warnings.catch_warnings():
            # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
            # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
            # See also: https://stackoverflow.com/q/74057367/859591
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=(
                    ".*will attempt to set the values inplace instead of always setting a new array. "
                    "To retain the old behavior, use either.*"
                ),
            )

            target_data.loc[:, 'target_scope'] = (
                target_data.target_scope.str.replace(r'\bs([123])', r'S\1', regex=True)
                .str.strip()
                .replace(r' ?\+ ?', '+', regex=True)
            )

        return target_data

    def _company_df_to_model(self, df_fundamentals: pd.DataFrame,
                             df_target_data: pd.DataFrame,
                             df_historic_data: pd.DataFrame) -> List[ICompanyData]:

        """
        transforms target Dataframe into list of ICompanyData instances.
        We don't necessarily have enough info to do target projections at this stage.

        :param df_fundamentals: pandas Dataframe with fundamental data; if None, use self._companies
        :param df_target_data: pandas Dataframe with target data; could be empty if we are partially initialized
        :param df_historic_data: pandas Dataframe with historic emissions, intensity, and production information; could be empty
        :return: A list containing the ICompanyData objects
        """
        if df_fundamentals is not None:
            companies_data_dict = df_fundamentals.to_dict(orient="records")
        else:
            companies_data_dict = [c.dict() for c in self._companies]
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
                    # FIXME: Is this the best place to finalize base_year_production, ghg_s1s2, and ghg_s3 data?
                    # Something tells me these parameters should be removed in favor of querying historical data directly
                    company_data[ColumnsConfig.BASE_YEAR_PRODUCTION] = (
                        df_historic_data.loc[company_id, 'Productions', 'production'][base_year])
                    try:
                        company_data[ColumnsConfig.GHG_SCOPE12] = (
                            df_historic_data.loc[company_id, 'Emissions', 'S1S2'][base_year])
                    except KeyError:
                        if (company_id, 'Emissions', 'S2') not in df_historic_data.index:
                            logger.warning(f"Scope 2 data missing from company with ID {company_id}; treating as zero")
                            company_data[ColumnsConfig.GHG_SCOPE12] = df_historic_data.loc[
                                company_id, 'Emissions', 'S1'][base_year]
                        else:
                            # FIXME: This should not reach here because we should have calculated
                            # S1S2 as an emissions total upstream from S1+S2.
                            assert False
                            company_data[ColumnsConfig.GHG_SCOPE12] = (
                                df_historic_data.loc[company_id, 'Emissions', 'S1'][base_year]
                                + df_historic_data.loc[company_id, 'Emissions', 'S2'][base_year])
                    try:
                        company_data[ColumnsConfig.GHG_SCOPE3] = (
                            df_historic_data.loc[company_id, 'Emissions', 'S3'][base_year])
                    except KeyError:
                        # If there was no relevant historic data, don't try to use it
                        pass
                    company_data[ColumnsConfig.HISTORIC_DATA] = self._convert_historic_data(
                        df_historic_data.loc[[company_id]].reset_index()).dict()
                else:
                    company_data[ColumnsConfig.HISTORIC_DATA] = None

                if company_id in df_target_data.index:
                    company_data[ColumnsConfig.TARGET_DATA] = [td.dict() for td in self._convert_target_data(
                        # don't let a single row of df_target_data become a pd.Series
                        df_target_data.loc[[company_id]].reset_index())]
                else:
                    company_data[ColumnsConfig.TARGET_DATA] = None

                # handling of missing market cap data is mainly done in _convert_from_template_company_data()
                if company_data[ColumnsConfig.COMPANY_MARKET_CAP] is pd.NA:
                    company_data[ColumnsConfig.COMPANY_MARKET_CAP] = np.nan

                model_companies.append(ICompanyData.parse_obj(company_data))
            except ValidationError:
                logger.error(f"(One of) the input(s) of company with ID {company_id} is invalid")
                breakpoint()
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
        try:
            historic_t = asPintDataFrame(historic.set_index(['variable', 'company_id', 'scope']).T)
        except pint.errors.DimensionalityError as err:
            logger.error(f"Dimensionality error {err} in DataFrame\n{historic}")
            breakpoint()
        # Historic data may well have ragged left and right columns
        # all_na = historic.apply(lambda x: all([ITR.isnan(y.m) for y in x]), axis=1)
        # historic = historic[~all_na]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pint units don't like being twisted from columns to rows, but it's ok
            productions = historic_t[VariablesConfig.PRODUCTIONS].T
            emissions = historic_t[VariablesConfig.EMISSIONS].T
            emissions_intensities = historic_t[VariablesConfig.EMISSIONS_INTENSITIES].T

        hd = IHistoricData(productions=self._convert_to_historic_productions(productions),
                           emissions=self._convert_to_historic_emissions(emissions.reset_index('scope')),
                           emissions_intensities=self._convert_to_historic_ei(emissions_intensities.reset_index('scope')))
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

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company (company_id is a column)
        FIXME: Callers want non-fundamental data here: base_year_production, ghg_s1s2, ghg_s3
        """
        excluded_cols = ['projected_targets', 'projected_intensities', 'historic_data', 'target_data']
        df = pd.DataFrame.from_records(
            [ICompanyData.parse_obj({k:v for k, v in c.dict().items() if k not in excluded_cols}).dict()
             for c in self.get_company_data(company_ids)]).set_index(self.column_config.COMPANY_ID)
        # company_ids_idx = pd.Index(company_ids)
        # df = self.df_fundamentals.loc[company_ids_idx]
        return df

