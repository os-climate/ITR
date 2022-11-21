import unittest
import pandas as pd
import json
import random

import ITR
from pint import Quantity
from ITR.utils import asPintSeries

from ITR.interfaces import EI_Metric, EI_Quantity, EScope
from ITR.interfaces import ICompanyData, ICompanyEIProjectionsScopes, ICompanyEIProjections, ICompanyEIProjection

class QuantityEncoder(json.JSONEncoder):
    def default(self, q):
        if isinstance(q, Quantity):
            return str(q)
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
    for d, data in enumerate(left):
        case.assertAlmostEqual(data, right[d], places, msg, delta)

    for d, data in enumerate(right):
        case.assertAlmostEqual(data, left[d], places, msg, delta)


def assert_pint_frame_equal(case: unittest.case, left: pd.DataFrame, right: pd.DataFrame, places=7, msg=None, delta=None):
    # Helper function to avoid bug in pd.testing.assert_frame_equal concerning pint series
    left_flat = left.values.flatten()
    right_flat = right.values.flatten()

    errors = []
    for d, data in enumerate(left_flat):
        try:
            case.assertAlmostEqual(data, right_flat[d], places, msg, delta)
        except AssertionError as e:
            errors.append(e.args[0])
    if errors:
        raise AssertionError('\n'.join(errors))

    for d, data in enumerate(right_flat):
        try:
            case.assertAlmostEqual(data, left_flat[d], places, msg, delta)
        except AssertionError as e:
            errors.append((e.args[0]))
    if errors:
        raise AssertionError('\n'.join(errors))

# General way to generate copmany data using Netzero Year (Slope)

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
        bm_ei = asPintSeries(bm_ei_scopes.loc[:, :, scope].reset_index().iloc[0, 2:])
        if scope == EScope.S1S2S3:
            if (EScope.S1S2 not in scopes
                or not bm_ei_scopes.loc[:,:,EScope.S1S2S3].equals(bm_ei_scopes.loc[:,:,EScope.S1S2])):
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
        else:
            if scope in [EScope.S1, EScope.S1S2]:
                company_dict['ghg_s1s2'] = (production * bm_ei[2019])
            elif scope == EScope.S3:
                company_dict['ghg_s3'] = (production * bm_ei[2019])
            else:
                raise ValueError
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

