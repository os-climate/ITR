import unittest
import pandas as pd
import json
import random

import ITR
from pint import Quantity
from ITR.data.osc_units import EI_Metric, EI_Quantity, asPintSeries

from ITR.interfaces import EScope
from ITR.interfaces import ICompanyData, ICompanyEIProjectionsScopes, ICompanyEIProjections, ICompanyEIProjection

class ITR_Encoder(json.JSONEncoder):
    def default(self, q):
        if isinstance(q, Quantity):
            if ITR.isna(q.m):
                return f"nan {q.u}"
            return f"{q:.5f}"
        elif isinstance(q, EScope):
            return q.value
        elif isinstance(q, pd.Series):
            # Inside the map function NA values become float64 nans and lose their units
            ser = q.map(lambda x: f"nan {q.pint.u}" if ITR.isna(x) else f"{x:.5f}")
            res = pd.DataFrame(data={'year': ser.index, 'value': ser.values}).to_dict('records')
            return res
        else:
            super().default(q)

class DequantifyQuantity(json.JSONEncoder):
    def default(self, q):
        if isinstance(q, Quantity):
            return q.m
        else:
            super().default(q)

def assert_pint_series_equal(case: unittest.case, left: pd.Series, right: pd.Series, places=7, msg=None, delta=None):
    # Helper function to avoid bug in pd.testing.assert_series_equal concerning pint series

    left_values = left.tolist()
    right_values = right.tolist()
    for i, value in enumerate(left_values):
        case.assertAlmostEqual(ITR.nominal_values(value.m),
                               ITR.nominal_values(right_values[i].to(value.u).m),
                               places, msg, delta)

    for i, value in enumerate(right_values):
        case.assertAlmostEqual(ITR.nominal_values(value.m),
                               ITR.nominal_values(left_values[i].to(value.u).m),
                               places, msg, delta)


def assert_pint_frame_equal(case: unittest.case, left: pd.DataFrame, right: pd.DataFrame, places=7, msg=None, delta=None):
    # Helper function to avoid bug in pd.testing.assert_frame_equal concerning pint series
    left_flat = left.values.flatten()
    right_flat = right.values.flatten()

    errors = []
    for d, data in enumerate(left_flat):
        try:
            case.assertAlmostEqual(ITR.nominal_values(data.m),
                                   ITR.nominal_values(right_flat[d].to(data.u).m),
                                   places, msg, delta)
        except AssertionError as e:
            errors.append(e.args[0])
    if errors:
        raise AssertionError('\n'.join(errors))

    for d, data in enumerate(right_flat):
        try:
            case.assertAlmostEqual(ITR.nominal_values(data.m),
                                   ITR.nominal_values(left_flat[d].to(data.u).m),
                                   places, msg, delta)
        except AssertionError as e:
            errors.append((e.args[0]))
    if errors:
        raise AssertionError('\n'.join(errors))

# General way to generate copmany data using Netzero Year (Slope)

# This method of value interpolation differs from the one in base_providers.py
# It estimates a little on the high side because it is linear, not CAGR-based
# But because benchmarks give data every 5 years, the cumulative error vs. the
# benchmarks is small.
def interpolate_value_at_year(y, bm_ei, ei_nz_year, ei_max_negative):
    i = bm_ei.index.searchsorted(y)
    if i==0:
        i = 1
    elif i==len(bm_ei):
        i = i-1
    first_y = bm_ei.index[i-1]
    last_y = bm_ei.index[i]
    if y < first_y or y > last_y:
        raise ValueError
    nz_interpolation = max(bm_ei.iloc[0]*(ei_nz_year-y)/(ei_nz_year-bm_ei.index[0]), ei_max_negative)
    bm_interpolation = (bm_ei[first_y]*(last_y-y) + bm_ei[last_y]*(y-first_y))/(last_y-first_y)
    return min(nz_interpolation, bm_interpolation)


def gen_company_data(company_name, company_id, region, sector, production,
                     bm_ei_scopes, ei_nz_year=2051, ei_max_negative=None) -> ICompanyData:
    ei_metric = str(bm_ei_scopes.iloc[0].dtype)[5:-1]
    if ei_max_negative is None:
        ei_max_negative = Quantity(0, ei_metric)
    company_dict = {
        'company_name': company_name,
        'company_id': company_id,
        'region': region,
        'sector': sector,
        'base_year_production': production,
    }
    scopes = bm_ei_scopes.droplevel([0,1]).index.tolist()
    scope_projections = {}
    for scope in scopes:
        try:
            bm_ei = asPintSeries(bm_ei_scopes.loc[sector, region, scope])
        except KeyError:
            bm_ei = asPintSeries(bm_ei_scopes.loc[sector, 'Global', scope])
        if scope == EScope.S1S2S3:
            if EScope.S1S2 in scopes and EScope.S3 in scopes:
                # Handled below
                pass
            elif EScope.S1S2 in scopes: # and EScope.S3 not in scopes
                # Compute S3 from S1S2S3 - S1S2
                company_dict['ghg_s3'] = production * (bm_ei[2019] - bm_ei_scopes.loc[(sector, slice(None), EScope.S1S2), 2019].iloc[0])
            elif EScope.S3 in scopes: # and EScope.S1S2 not in scopes
                # Compute S1S2 from S1S2S3 - S3
                company_dict['ghg_s1s2'] = production * (bm_ei[2019] - bm_ei_scopes.loc[(sector, slice(None), EScope.S3), 2019].iloc[0])
            else:
                s1s2_s3_split = random.uniform(0.5,0.9)
                company_dict['ghg_s1s2'] = (production * bm_ei[2019] * (1-s1s2_s3_split))
                company_dict['ghg_s3'] = (production * bm_ei[2019] * s1s2_s3_split)
                scope_projections[EScope.S1S2.name] = {
                    'ei_metric': EI_Metric(ei_metric),
                    'projections': [
                        ICompanyEIProjection.parse_obj({
                            'year':y,
                            'value': EI_Quantity(interpolate_value_at_year(y, bm_ei * (1-s1s2_s3_split), ei_nz_year, ei_max_negative)),
                        }) for y in range(2019, 2051)
                    ]
                }
                scope_projections[EScope.S3.name] = {
                    'ei_metric': EI_Metric(ei_metric),
                    'projections': [
                        ICompanyEIProjection.parse_obj({
                            'year':y,
                            'value': EI_Quantity(interpolate_value_at_year(y, bm_ei * s1s2_s3_split, ei_nz_year, ei_max_negative)),
                        }) for y in range(2019, 2051)
                    ]
                }
                continue

        if scope == EScope.S1S2 or (scope == EScope.S1 and EScope.S1S2 not in scopes):
            company_dict['ghg_s1s2'] = (production * bm_ei[2019])
        elif scope == EScope.S3:
            company_dict['ghg_s3'] = (production * bm_ei[2019])
        elif scope == EScope.S1S2S3:
            pass

        scope_projections[scope.name] = {
            'ei_metric': EI_Metric(ei_metric),
            'projections': [
                ICompanyEIProjection.parse_obj({
                    'year':y,
                    'value': EI_Quantity(interpolate_value_at_year(y, bm_ei, ei_nz_year, ei_max_negative)),
                }) for y in range(2019, 2051)
            ]
        }

    company_dict['projected_targets'] = ICompanyEIProjectionsScopes(**scope_projections)
    
    company_data = ICompanyData.parse_obj(company_dict)
    return company_data

