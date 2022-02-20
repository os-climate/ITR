from typing import List

import pandas as pd
import numpy as np
import math

# Timeline definition:

# Historical Data has a first year and a last year (typically 2015-2020)  
# Target Data has a base year (could be 2000, 2005, 2015) and a target year (typically 2030, 2050)

# In order to have valid data, the base year must be between the start and end years.

# Step 0: Overall function
from pandas._libs.missing import NAType

from ITR.interfaces import ICompanyEIProjectionsScopes, ICompanyEIProjections, ITargetData, IHistoricData, \
    ICompanyEIProjection, pint_ify


def compute_CAGR(first, last, period):
    """Input:
    @first: first value
    @last: last value
    @period: number of periods in the CAGR"""

    if period == 0:
        res = 1
    else:
        # TODO: Replace ugly fix => pint unit error in below expression
        # CAGR doesn't work well with 100% reduction, so set it to small
        if last == 0:
            last = 0.000001
        res = (last / first).magnitude ** (1 / period) - 1
    return res


# Step 1: function for tagret trajectory

# data_emissions includes columns for: 
# ISIN, Date, Region, Scope 1, Scope 2

# data_production includes columns for:
# ISIN, Date, Financial Data (Revenue, Market Cap, Debt, etc), Steel Production, Eletricity Production, Other production (Oil & Gas, EVs, etc)

# data_benchmark includes columns for:
# Sector, Date, Region, Unit_intensity, Unit_Production (always %), Intensity, Production (Annual Growth)

# Returns a dataframe of a single ISIN, Region, Sector, Data for years 2020-2050:
# Also Emission, Production, intensity, CAGR, CAGR_emission, CAGR_production
# Also forecast_target, forecast_emission, forecast_production, forecast_intensity
def project_targets(targets: List[ITargetData], historic_data: IHistoricData, isin=None,
                    data_emissions: pd.DataFrame = None, data_prod=None) -> ICompanyEIProjectionsScopes:
    """Input:
    @isin: isin of the company for which to compute the projection
    @data_emission: database with emission with emissions, intensity, sector and region columns
    @data_prod: database with production evolution from benchmark 
    
    If the company has no target or the target can't be processed, then the output the emission database, unprocessed
    """
    # global data_benchmark

    # TODO: expand function to handle multiple targets / loop over scopes
    target = targets[0]
    scope = target.target_scope

    # Get the intensity data
    intensity_data = historic_data.emission_intensities.__getattribute__(scope.name)

    # Get first year data and last year data with non-null values
    first_year, value_first_year = intensity_data[0].year, intensity_data[0].value
    last_year_data = next((i for i in reversed(intensity_data) if type(i.value.magnitude) != NAType), None)
    last_year, value_last_year = last_year_data.year, last_year_data.value

    # Solve for intensity and absolute
    if target.target_type == "intensity":
        # Simple case: the target is in intensity
        base_year = target.base_year
        if (base_year < last_year):  # Removed condition base year > first_year. Do we care as long as base_year_qty is known?
            target_year = target.end_year
            # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
            target_value = pint_ify(target.target_base_qty * (1 - target.target_reduction_pct), target.target_base_unit)
            CAGR = compute_CAGR(value_last_year, target_value, (target_year - last_year))

            # projections = []
            # for y, year in enumerate(range(1 + last_year, 1 + target_year)):
            #     projection = ICompanyEIProjection(year=year, value=value_last_year * (1 + CAGR)**(y + 1))
            #     projections.append(projection)
            target_ei_projections = ICompanyEIProjections(projections=
                  [ICompanyEIProjection(year=year, value=value_last_year * (1 + CAGR) ** (y + 1))
                                        for y, year in enumerate(range(1 + last_year, 1 + target_year))]
            )

        else:  # test is we have base data in sample
            target_ei_projections = None

    elif target.target_type == "absolute":
        # Complicated case, the target must be switched from absolute value to intensity.
        # We use the benchmark production data
        # Compute Emission CAGR
        base_year = target.base_year
        if (base_year < last_year) & (base_year < first_year):

            target_year = target.end_year
            # Correction here for percentage
            target_value = df_isin.loc[lambda row: row["year"] == base_year, "Emission"].values[0] * (
                        1 - target.target_reduction_pct / 100)
            df_isin.loc[lambda row: row["year"] == target_year, "Emission"] = target_value
            df_isin["Production"] = df_isin["Emission"] / df_isin["intensity"]

            # Correction here for geometric evolution for production
            # Production is recalculated using intensity and emissions (maybe this should change accoridng to data QC)

            # First step: we compute the evolution for emissions (ie: the aboslute value)
            value_last_year_emission = df_isin.loc[lambda row: row["year"] == last_year, "Emission"].values[0]
            CAGR_abs = compute_CAGR(value_last_year_emission, target_value, (target_year - last_year))

            # Add CAGR and forecast
            df_isin['CAGR_emission'] = 0
            df_isin['forecast_emission'] = df_isin['Emission']

            # Input CAGR
            df_isin.loc[lambda row: row['year'].between(last_year + 1,
                                                        2050), 'CAGR_emission'] = CAGR_abs
            # Cumulative prod
            df_isin['CAGR_emission'] = (1 + df_isin['CAGR_emission']).cumprod()
            # Compute forecast
            df_isin.loc[lambda row: row['year'] > last_year, "forecast_emission"] = \
                df_isin.loc[lambda row: row['year'] == last_year, 'forecast_emission'].values[0] * df_isin[
                    'CAGR_emission']

            # Second step: we compute the evolution for production, based on the benchmark production evolution

            # Compute benchmark CAGR (mean yearly evolution)
            sector = df_isin["Sector"].values[0]
            region = df_isin["Region"].values[0]
            data_benchmark = data_prod.loc[lambda row: (row["Sector"] == sector) & (row["Region"] == region), :]
            CAGR_prod = data_benchmark.loc[
                lambda row: (row["Date"] <= target_year) & (row["Date"] >= last_year), "Production"].mean()

            # Add CAGR and forecast
            df_isin['CAGR_production'] = 0
            df_isin['forecast_production'] = df_isin['Production']

            # Input CAGR
            df_isin.loc[lambda row: row['year'].between(last_year + 1,
                                                        2050), 'CAGR_production'] = CAGR_prod
            # Cumulative prod
            df_isin['CAGR_production'] = (1 + df_isin['CAGR_production']).cumprod()
            # Compute forecast
            df_isin.loc[lambda row: row['year'] > last_year, "forecast_production"] = \
                df_isin.loc[lambda row: row['year'] == last_year, 'forecast_production'].values[0] * df_isin[
                    'CAGR_production']

            # Final step: we divid and get the intensity evolution
            df_isin["forecast_intensity"] = df_isin["forecast_emission"] / df_isin["forecast_production"]

            # Approximation: here we say that the intensity evolution is that of emissions minus production
            # If absolute decreases by 5% per year and production grows by 5% a year, intensity must decrease by 10% a year

        else:
            CAGR = np.nan

    else:
        # No target
        # Maybe modification needed here, depends on the output needed for the case where there is no target
        CAGR = np.nan

    return ICompanyEIProjectionsScopes(
        S1S2=target_ei_projections,
        S3=None,
        S1S2S3=None
    )
