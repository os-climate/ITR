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

from ITR.data.base_providers import BaseProviderProductionBenchmark
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
            last = first/201.0
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

# Remember that pd.Series are always well-behaved with pint[] quantities.  pd.DataFrame columns are well-behaved,
# but data across columns is not always well-behaved.  We therefore make this function assume we are projecting targets
# for a specific company, in a specific sector.  If we want to project targets for multiple sectors, we have to call it multiple times.
# This function doesn't need to know what sector it's computing for...only tha there is only one such, for however many scopes.
def project_targets(targets: List[ITargetData], historic_data: IHistoricData, production_bm: pd.Series = None,
                    data_prod=None) -> ICompanyEIProjectionsScopes:
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

    base_year = target.base_year

    # Solve for intensity and absolute
    if target.target_type == "intensity":
        # Simple case: the target is in intensity
        # Get the intensity data
        intensity_data = historic_data.emission_intensities.__getattribute__(scope.name)

        # Get last year data with non-null value
        last_year_data = next((i for i in reversed(intensity_data) if type(i.value.magnitude) != NAType), None)
        last_year, value_last_year = last_year_data.year, last_year_data.value

        if last_year is None or base_year >= last_year:
            target_ei_projections = None
        else:  # Removed condition base year > first_year. Do we care as long as base_year_qty is known?
            target_year = target.end_year
            # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
            target_value = pint_ify(target.target_base_qty * (1 - target.target_reduction_pct), target.target_base_unit)
            CAGR = compute_CAGR(value_last_year, target_value, (target_year - last_year))
            target_ei_projections = ICompanyEIProjections(projections=
                  [ICompanyEIProjection(year=year, value=value_last_year * (1 + CAGR) ** (y + 1))
                                        for y, year in enumerate(range(1 + last_year, 1 + target_year))]
            )

    elif target.target_type == "absolute":
        # Complicated case, the target must be switched from absolute value to intensity.
        # We use the benchmark production data
        # Compute Emission CAGR
        emission_data = historic_data.emissions.__getattribute__(scope.name)

        # Get last year data with non-null value
        last_year_data = next((e for e in reversed(emission_data) if type(e.value.magnitude) != NAType), None)
        last_year, value_last_year = last_year_data.year, last_year_data.value

        if last_year is None or base_year >= last_year:
            target_ei_projections = None
        else: # Removed condition base year > first_year. Do we care as long as base_year_qty is known?
            target_year = target.end_year
            # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
            target_value = pint_ify(target.target_base_qty * (1 - target.target_reduction_pct), target.target_base_unit)
            CAGR = compute_CAGR(value_last_year, target_value, (target_year - last_year))
            emission_projections = [value_last_year * (1 + CAGR) ** (y + 1)
                                    for y, year in enumerate(range(1 + last_year, 1 + target_year))]
            emission_projections = pd.Series(emission_projections, index=range(last_year + 1, target_year + 1),
                                             dtype=f'pint[{target.target_base_unit}]')
            production_projections = production_bm.loc[last_year + 1: target_year]
            ei_projections = emission_projections / production_projections

            target_ei_projections = ICompanyEIProjections(projections=
                                                          [ICompanyEIProjection(year=year, value=ei_projections[year])
                                                           for year in range(last_year + 1, target_year + 1)]
                                                          )

    ei_projection_scopes = {"S1S2": None, "S3": None, "S1S2S3": None}
    for scope in ei_projection_scopes.keys():
        scope_targets = [target for target in targets if target.target_scope.name == scope]
        scope_targets.sort(key=lambda target: (target.target_scope, target.end_year))
        while scope_targets:
            target = scope_targets.pop(0)
            base_year = target.base_year

            # Solve for intensity and absolute
            if target.target_type == "intensity":
                # Simple case: the target is in intensity
                # Get the intensity data
                intensity_data = historic_data.emission_intensities.__getattribute__(scope)

                # Get last year data with non-null value
                if ei_projection_scopes[scope] is not None:
                    last_year_data = ei_projection_scopes[scope].projections[-1]
                else:
                    last_year_data = next((i for i in reversed(intensity_data) if type(i.value.magnitude) != NAType),
                                          None)

                if last_year_data is None or base_year >= last_year_data.year:
                    ei_projection_scopes[scope] = None
                else:  # Removed condition base year > first_year. Do we care as long as base_year_qty is known?
                    last_year, value_last_year = last_year_data.year, last_year_data.value
                    target_year = target.end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_value = pint_ify(target.target_base_qty * (1 - target.target_reduction_pct),
                                            target.target_base_unit)
                    CAGR = compute_CAGR(value_last_year, target_value, (target_year - last_year))
                    if not scope_targets:  # Check if there are no more targets for this scope
                        target_year = 2050  # Value should come from somewhere else
                    ei_projections = [ICompanyEIProjection(year=year, value=value_last_year * (1 + CAGR) ** (y + 1))
                                      for y, year in enumerate(range(1 + last_year, 1 + target_year))]
                    if ei_projection_scopes[scope] is not None:
                        ei_projection_scopes[scope].projections.extend(ei_projections)
                    else:
                        ei_projection_scopes[scope] = ICompanyEIProjections(projections=ei_projections)

            elif target.target_type == "absolute":
                # Complicated case, the target must be switched from absolute value to intensity.
                # We use the benchmark production data
                # Compute Emission CAGR
                emission_data = historic_data.emissions.__getattribute__(scope)

                # Get last year data with non-null value
                if ei_projection_scopes[scope] is not None:
                    last_year = ei_projection_scopes[scope].projections[-1].year
                    last_year_data = next((e for e in emission_data if e.year == last_year), None)
                else:
                    last_year_data = next((e for e in reversed(emission_data) if type(e.value.magnitude) != NAType),
                                          None)

                if last_year_data is None or base_year >= last_year_data.year:
                    ei_projection_scopes[scope] = None
                else:  # Removed condition base year > first_year. Do we care as long as base_year_qty is known?
                    last_year, value_last_year = last_year_data.year, last_year_data.value
                    target_year = target.end_year
                    # Attribute target_reduction_pct of ITargetData is currently a fraction, not a percentage.
                    target_value = pint_ify(target.target_base_qty * (1 - target.target_reduction_pct),
                                            target.target_base_unit)
                    CAGR = compute_CAGR(value_last_year, target_value, (target_year - last_year))

                    if not scope_targets:  # Check if there are no more targets for this scope
                        target_year = 2050  # Value should come from somewhere else
                    emission_projections = [value_last_year * (1 + CAGR) ** (y + 1)
                                            for y, year in enumerate(range(last_year + 1, target_year + 1))]
                    emission_projections = pd.DataFrame([emission_projections],
                                                        columns=range(last_year + 1, target_year + 1))
                    production_projections = production_bm.loc[:, last_year + 1: target_year]
                    ei_projections = emission_projections / production_projections

                    ei_projections = [ICompanyEIProjection(year=year, value=ei_projections[year].values.quantity)
                                      for year in range(last_year + 1, target_year + 1)]
                    if ei_projection_scopes[scope] is not None:
                        ei_projection_scopes[scope].projections.extend(ei_projections)
                    else:
                        ei_projection_scopes[scope] = ICompanyEIProjections(projections=ei_projections)

            else:
                # No target (type) specified
                ei_projection_scopes[scope] = None

    return ICompanyEIProjectionsScopes(**ei_projection_scopes)
