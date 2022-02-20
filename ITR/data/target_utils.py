from typing import List

import pandas as pd
import numpy as np

# Timeline definition:

# Historical Data has a first year and a last year (typically 2015-2020)  
# Target Data has a base year (could be 2000, 2005, 2015) and a target year (typically 2030, 2050)

# In order to have valid data, the base year must be between the start and end years.

#Step 0: Overall function
from ITR.interfaces import ICompanyEIProjectionsScopes, ICompanyEIProjections, ITargetData


def compute_CAGR(first, last, period):
    """Input:
    @first: first value
    @last: last value
    @period: number of periods in the CAGR"""
    
    if period == 0:
        res = 1
    else:
        res = (last/first)**(1/period)-1
    return res

#Step 1: function for tagret trajectory

# data_emissions includes columns for: 
# ISIN, Date, Region, Scope 1, Scope 2

# data_production includes columns for:
# ISIN, Date, Financial Data (Revenue, Market Cap, Debt, etc), Steel Production, Eletricity Production, Other production (Oil & Gas, EVs, etc)

# data_benchmark includes columns for:
# Sector, Date, Region, Unit_intensity, Unit_Production (always %), Intensity, Production (Annual Growth)

# Returns a dataframe of a single ISIN, Region, Sector, Data for years 2020-2050:
# Also Emission, Production, intensity, CAGR, CAGR_emission, CAGR_production
# Also forecast_target, forecast_emission, forecast_production, forecast_intensity
def project_targets(targets: List[ITargetData], isin=None, data_emissions: pd.DataFrame=None, data_prod=None) -> ICompanyEIProjectionsScopes:
    """Input:
    @isin: isin of the company for which to compute the projection
    @data_emission: database with emission with emissions, intensity, sector and region columns
    @data_prod: database with production evolution from benchmark 
    
    If the company has no target or the target can't be processed, then the output the emission database, unprocessed
    """
    # global data_benchmark

    # TODO: expand function to handle multiple targets
    target = targets[0]

    #Get the intensity data
    df_isin = data_emissions.loc[lambda row:row["company_id"]==isin,:]

    # Get first and last year
    first_year = df_isin.loc[lambda row:row["intensity"].notnull(),"year"].min()
    last_year = df_isin.loc[lambda row:row["intensity"].notnull(),"year"].max()
    value_first_year = df_isin.loc[lambda row:row["year"]==first_year,"intensity"].values[0]
    value_last_year = df_isin.loc[lambda row:row["year"]==last_year,"intensity"].values[0]
    
    #Add the years until 2050
    temp = pd.DataFrame(range(2020, 2051), columns=['year'])
    df_isin = pd.merge(df_isin,
                       temp,
                       how='outer',
                       on='year')

    df_isin.loc[:, ['company_id','Region','Sector']] = df_isin.loc[:,['company_id','Region','Sector']].fillna(method='ffill')
    df_isin = df_isin.sort_values("year")

    #Solve for intensity and absolute
    if target.target_type == "intensity":
        #Simple case: the target is in intensity
        base_year = target.base_year
        if (base_year<last_year)&(base_year>first_year):
            target_year = target.end_year
            #Correction here for percentage
            target_value = df_isin.loc[lambda row:row["year"]==base_year,"intensity"].values[0] * (1 - target.target_reduction_pct / 100)
            df_isin.loc[lambda row:row["year"]==target_year,"intensity"] = target_value
            value_last_year_emission = df_isin.loc[lambda row:row["year"]==last_year,"Emission"].values[0]
            CAGR = compute_CAGR(value_last_year,target_value,(target_year - last_year))

            #Add CAGR and forecast
            df_isin['CAGR'] = 0
            df_isin['forecast_target'] = df_isin['intensity']

            #Input CAGR
            df_isin.loc[lambda row: row['year'].between(last_year + 1,
                                                       2050), 'CAGR'] = CAGR
            # Cumulative prod
            df_isin['CAGR'] = (1 + df_isin['CAGR']).cumprod()
            # Compute forecast
            df_isin.loc[lambda row: row['year'] > last_year, "forecast_target"] = \
            df_isin.loc[lambda row: row['year'] == last_year, 'forecast_target'].values[0] * df_isin['CAGR']
        else:#test is we have base data in sample
            CAGR = np.nan
        
    elif target.target_type == "absolute":
        #Complicated case, the target must be switched from absolute value to intensity. 
        #We use the benchmark production data
        #Compute Emission CAGR
        base_year = target.base_year
        if (base_year<last_year)&(base_year<first_year):

                target_year = target.end_year
                #Correction here for percentage
                target_value = df_isin.loc[lambda row:row["year"]==base_year,"Emission"].values[0] * (1 - target.target_reduction_pct / 100)
                df_isin.loc[lambda row:row["year"]==target_year,"Emission"] = target_value
                df_isin["Production"] = df_isin["Emission"] /df_isin["intensity"] 

                #Correction here for geometric evolution for production
                #Production is recalculated using intensity and emissions (maybe this should change accoridng to data QC)
                
                #First step: we compute the evolution for emissions (ie: the aboslute value)
                value_last_year_emission = df_isin.loc[lambda row:row["year"]==last_year,"Emission"].values[0]
                CAGR_abs  = compute_CAGR(value_last_year_emission,target_value,(target_year - last_year))

                #Add CAGR and forecast
                df_isin['CAGR_emission'] = 0
                df_isin['forecast_emission'] = df_isin['Emission']

                #Input CAGR
                df_isin.loc[lambda row: row['year'].between(last_year + 1,
                                                           2050), 'CAGR_emission'] = CAGR_abs
                # Cumulative prod
                df_isin['CAGR_emission'] = (1 + df_isin['CAGR_emission']).cumprod()
                # Compute forecast
                df_isin.loc[lambda row: row['year'] > last_year, "forecast_emission"] = \
                df_isin.loc[lambda row: row['year'] == last_year, 'forecast_emission'].values[0] * df_isin['CAGR_emission']

                #Second step: we compute the evolution for production, based on the benchmark production evolution

                #Compute benchmark CAGR (mean yearly evolution)
                sector=df_isin["Sector"].values[0]
                region=df_isin["Region"].values[0]
                data_benchmark = data_prod.loc[lambda row:(row["Sector"]==sector)&(row["Region"]==region),:]
                CAGR_prod = data_benchmark.loc[lambda row:(row["Date"]<=target_year)&(row["Date"]>=last_year),"Production"].mean()

                #Add CAGR and forecast
                df_isin['CAGR_production'] = 0
                df_isin['forecast_production'] = df_isin['Production']

                #Input CAGR
                df_isin.loc[lambda row: row['year'].between(last_year + 1,
                                                           2050), 'CAGR_production'] = CAGR_prod
                # Cumulative prod
                df_isin['CAGR_production'] = (1 + df_isin['CAGR_production']).cumprod()
                # Compute forecast
                df_isin.loc[lambda row: row['year'] > last_year, "forecast_production"] = \
                df_isin.loc[lambda row: row['year'] == last_year, 'forecast_production'].values[0] * df_isin['CAGR_production']


                #Final step: we divid and get the intensity evolution
                df_isin["forecast_intensity"] = df_isin["forecast_emission"] /df_isin["forecast_production"]


                #Approximation: here we say that the intensity evolution is that of emissions minus production 
                #If absolute decreases by 5% per year and production grows by 5% a year, intensity must decrease by 10% a year

        else:
                CAGR = np.nan
    
    else:
        #No target
        #Maybe modification needed here, depends on the output needed for the case where there is no target
        CAGR=np.nan

    return ICompanyEIProjectionsScopes(
        S1S2=ICompanyEIProjections,
        S3=None,
        S1S2S3=None
    )