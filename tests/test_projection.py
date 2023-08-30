import json
import unittest
import os
import datetime
from typing import List
import pandas as pd

from utils import QuantityEncoder
from ITR.data.base_providers import EITrajectoryProjector
from ITR.interfaces import ICompanyData, ProjectionControls


def is_pint_dict_equal(result: List[dict], reference: List[dict]) -> bool:
    is_equal = True
    for i in range(len(result)):
        if json.dumps(result[i], cls=QuantityEncoder) != json.dumps(reference[i], cls=QuantityEncoder):
            print(f"Differences in projections_dict[{i}]: company_name = {result[i]['company_name']}; company_id = {result[i]['company_id']}")
            for k, v in result[i].items():
                if k == 'ghg_s1s2' and not reference[i].get(k):
                    print("ghg_s1s2")
                    continue
                if k == 'projected_intensities':
                    for scope in result[i][k]:
                        if reference[i][k].get(scope):
                            vref = reference[i][k]
                            if not v.get(scope):
                                print(f"projection has no scope {scope} for projection_intensities")
                                is_equal = False
                            elif json.dumps(v[scope]['projections'], cls=QuantityEncoder) != json.dumps(vref[scope]['projections'], cls=QuantityEncoder):
                                print(f"projected_intensities differ for scope {scope}")
                                print(f"computed {k}:\n{json.dumps(v[scope]['projections'], cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref[scope]['projections'], cls=QuantityEncoder)}\n\n")
                                is_equal = False
                        elif v.get(scope):
                            print(f"reference has no scope {scope} for projection_intensities")
                            # ???? is_equal = False
                    continue
                try:
                    vref = reference[i][k]
                    if json.dumps(v, cls=QuantityEncoder) != json.dumps(vref, cls=QuantityEncoder):
                        print(f"computed {k}:\n{json.dumps(v, cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref, cls=QuantityEncoder)}\n\n")
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
