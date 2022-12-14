import json
import unittest
import os
import datetime
from typing import List
import pandas as pd

import ITR
from utils import ITR_Encoder, assert_pint_series_equal
from ITR.data.osc_units import PA_
from ITR.configs import ColumnsConfig, VariablesConfig
from ITR.data.base_providers import BaseProviderProductionBenchmark, EITrajectoryProjector, EITargetProjector
from ITR.interfaces import EScope, ICompanyData, ProjectionControls, IProductionBenchmarkScopes, ITargetData


def is_pint_dict_equal(result: List[dict], reference: List[dict]) -> bool:
    is_equal = True
    for i in range(len(result)):
        if json.dumps(result[i], cls=ITR_Encoder) != json.dumps(reference[i], cls=ITR_Encoder):
            print(f"Differences in projections_dict[{i}]: company_name = {result[i]['company_name']}; company_id = {result[i]['company_id']}")
            for k, v in result[i].items():
                if k == 'ghg_s1s2' and not reference[i].get(k):
                    print("ghg_s1s2")
                    continue
                if k in ['projected_intensities', 'projected_targets']:
                    if not result[i][k]:
                        continue
                    for scope in result[i][k]:
                        if reference[i][k].get(scope):
                            vref = reference[i][k]
                            if not v.get(scope):
                                print(f"projection has no scope {scope} for {k}")
                                is_equal = False
                            elif json.dumps(v[scope]['projections'], cls=ITR_Encoder) != json.dumps(vref[scope]['projections'], cls=ITR_Encoder):
                                print(f"{k} differ for scope {scope}")
                                print(f"computed {k}:\n{json.dumps(v[scope]['projections'], cls=ITR_Encoder)}\n\nreference {k}:\n{json.dumps(vref[scope]['projections'], cls=ITR_Encoder)}\n\n")
                                is_equal = False
                        elif v.get(scope):
                            print(f"reference has no scope {scope} for projection_intensities")
                            # ???? is_equal = False
                    continue
                try:
                    vref = reference[i][k]
                    if json.dumps(v, cls=ITR_Encoder) != json.dumps(vref, cls=ITR_Encoder):
                        print(f"computed {k}:\n{json.dumps(v, cls=ITR_Encoder)}\n\nreference {k}:\n{json.dumps(vref, cls=ITR_Encoder)}\n\n")
                        is_equal = False
                except KeyError as e:
                    print(f"missing in reference: {k}: {json.dumps(v)}\n\n")
                    is_equal = False

    return is_equal


class TestProjector(unittest.TestCase):
    """
    Test the projector that converts historic data into emission intensity projections
    """

    def setUp(self) -> None:
        self.root: str = os.path.dirname(os.path.abspath(__file__))
        self.source_path: str = os.path.join(self.root, "inputs", "json", "test_project_companies.json")
        self.json_reference_path: str = os.path.join(self.root, "inputs", "json", "test_project_reference.json")
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        with open(self.source_path, 'r') as file:
            company_dicts = json.load(file)
        for company_dict in company_dicts:
            company_dict['report_date'] = datetime.date(2021, 12, 31)
        self.companies = [ICompanyData(**company_dict) for company_dict in company_dicts]
        self.projector = EITrajectoryProjector()

    def test_project(self):
        projections = self.projector.project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]
        test_successful = is_pint_dict_equal(projections_dict, reference_projections)

        self.assertEqual(test_successful, True)

    def test_targets(self):
        # Test that both absolute targets and intensity targets produce sane results
        # Our ICompanyData need only have: company_id, company_name, base_year_production, ghg_scope12, sector, region, target_data
        # We need bm production data in case we need to convert supplied emissions data to calculate intensity targets or
        # supplied intensity data to compute absolute targets
        company_data = self.companies
        company_dict = {
            field : [ getattr(c, field) for c in company_data ]
            for field in [ ColumnsConfig.BASE_YEAR_PRODUCTION, ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR, ColumnsConfig.REGION ]
        }
        company_dict[ColumnsConfig.SCOPE] = [ EScope.S1S2 ] * len(company_data)
        company_index = [ c.company_id for c in company_data ]
        company_sector_region_info = pd.DataFrame(company_dict, pd.Index(company_index, name='company_id'))
        bm_production_data = self.base_production_bm.get_company_projected_production(company_sector_region_info)
        expected_0 = pd.Series(PA_([0.131, # 2019
                                    0.123, 0.116, 0.108, 0.102, 0.096, 0.09, 0.084, 0.079, 0.074, 0.07,  # 2020-2029
                                    0.066, 0.058, 0.051, 0.045, 0.039, 0.034, 0.03, 0.026, 0.023, 0.02,   # 2030-2039
                                    0.017, 0.014, 0.012, 0.01, 0.008, 0.006, 0.005, 0.003, 0.002, 0.001, # 2040-2049
                                    0.0], # 2050
                                   dtype='pint[t CO2/GJ]'),
                               index=range(2019,2051),
                               name='expected_0')
        target_0 = ITargetData(**{
            'netzero_year': 2050,
            'target_type': 'intensity',
            'target_scope': EScope.S1S2,
            'target_start_year': 2020,
            'target_base_year': 2019,
            'target_end_year': 2030,
            'target_base_year_qty': 0.131037611,
            'target_base_year_unit': 't CO2/GJ',
            'target_reduction_pct': 0.5,
        })
        expected = [ expected_0 ]
        company_data[0].target_data = [ target_0 ]
        for i, c in enumerate(company_data):
            if c.target_data:
                projected_targets = EITargetProjector(self.projector.projection_controls).project_ei_targets(c, bm_production_data.loc[(c.company_id, EScope.S1S2)])
                assert_pint_series_equal(self,
                                         pd.Series(PA_([x.value.m for x in projected_targets.S1S2.projections],
                                                       dtype=projected_targets.S1S2.ei_metric),
                                                   index=range(2019,2051)),
                                         expected[i], places=3)
            else:
                assert c.projected_targets is None
        
    def test_extrapolate(self):
        with open(os.path.join(self.root, "inputs", "json", "test_fillna_companies.json"), 'r') as file:
            company_dicts = json.load(file)
        for company_dict in company_dicts:
            company_dict['report_date'] = datetime.date(2021, 12, 31)
        fillna_data = [ICompanyData(**company_dict) for company_dict in company_dicts]

        ei_projector = EITrajectoryProjector(ProjectionControls(UPPER_PERCENTILE=0.9, LOWER_PERCENTILE=0.1))
        historic_data = ei_projector._extract_historic_data(fillna_data)
        ei_projector._compute_missing_historic_ei(fillna_data, historic_data)

        historic_years = [column for column in historic_data.columns if type(column) == int]
        projection_years = range(max(historic_years), ei_projector.projection_controls.TARGET_YEAR)
        historic_intensities = historic_data[historic_years].query(
            f"variable=='{VariablesConfig.EMISSIONS_INTENSITIES}'")
        standardized_intensities = ei_projector._standardize(historic_intensities)
        intensity_trends = ei_projector._get_trends(standardized_intensities)
        extrapolated = ei_projector._extrapolate(intensity_trends, projection_years, historic_data)
        ei_projector._add_projections_to_companies(fillna_data, extrapolated)
        # Figure out what to test here--just making it through with funny company data is step 1!

    # Need test data in order to test mean
    def test_median(self):
        projections = EITrajectoryProjector(ProjectionControls(TREND_CALC_METHOD=pd.DataFrame.median)).project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]

        test_successful = is_pint_dict_equal(projections_dict, reference_projections)
        self.assertEqual(test_successful, True)

    def test_upper_lower(self):
        projections = EITrajectoryProjector(ProjectionControls(UPPER_PERCENTILE=0.9, LOWER_PERCENTILE=0.1)).project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]
        test_successful = is_pint_dict_equal(projections_dict, reference_projections)
        self.assertEqual(test_successful, True)


if __name__ == "__main__":
    s = pd.Series([1, None, 3], dtype=float)
    s.interpolate(method="linear")
    test = TestProjector()
    test.setUp()
    test.test_project()
