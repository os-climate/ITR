import json
import unittest
import os
import datetime
import pandas as pd

from utils import QuantityEncoder
from ITR.data.base_providers import EITrajectoryProjector
from ITR.interfaces import ICompanyData, ProjectionControls


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
        test_failed = False
        projections = self.projector.project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]

        for i in range(len(projections_dict)):
            if json.dumps(projections_dict[i], cls=QuantityEncoder) != json.dumps(reference_projections[i], cls=QuantityEncoder):
                print(f"Differences in projections_dict[{i}]: company_name = {projections_dict[i]['company_name']}; company_id = {projections_dict[i]['company_id']}")
                for k, v in projections_dict[i].items():
                    if k == 'ghg_s1s2' and not reference_projections[i].get(k):
                        print("ghg_s1s2")
                        continue
                    if k == 'projected_intensities':
                        for scope in projections_dict[i][k]:
                            if reference_projections[i][k].get(scope):
                                vref = reference_projections[i][k]
                                if not v.get(scope):
                                    print(f"projection has no scope {scope} for projection_intensities")
                                    test_failed = True
                                elif json.dumps(v[scope]['projections'], cls=QuantityEncoder) != json.dumps(vref[scope]['projections'], cls=QuantityEncoder):
                                    print(f"projected_intensities differ for scope {scope}")
                                    print(f"computed {k}:\n{json.dumps(v[scope]['projections'], cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref[scope]['projections'], cls=QuantityEncoder)}\n\n")
                                    test_failed = True
                            elif v.get(scope):
                                print(f"reference has no scope {scope} for projection_intensities")
                                # ???? test_failed = True
                        continue
                    try:
                        vref = reference_projections[i][k]
                        if json.dumps(v, cls=QuantityEncoder) != json.dumps(vref, cls=QuantityEncoder):
                            print(f"computed {k}:\n{json.dumps(v, cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref, cls=QuantityEncoder)}\n\n")
                            test_failed = True
                    except KeyError as e:
                        print(f"missing in reference: {k}: {json.dumps(v)}\n\n")
                        test_failed = True

        self.assertEqual(test_failed, False)


    # Need test data in order to test mean
    def test_median(self):
        test_failed = False
        projections = EITrajectoryProjector(ProjectionControls(TREND_CALC_METHOD=pd.DataFrame.median)).project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]

        for i in range(len(projections_dict)):
            if json.dumps(projections_dict[i], cls=QuantityEncoder) != json.dumps(reference_projections[i], cls=QuantityEncoder):
                print(f"Differences in projections_dict[{i}]: company_name = {projections_dict[i]['company_name']}; company_id = {projections_dict[i]['company_id']}")
                for k, v in projections_dict[i].items():
                    if k == 'ghg_s1s2' and not reference_projections[i].get(k):
                        print("ghg_s1s2")
                        continue
                    if k == 'projected_intensities':
                        for scope in projections_dict[i][k]:
                            if reference_projections[i][k].get(scope):
                                vref = reference_projections[i][k]
                                if not v.get(scope):
                                    print(f"projection has no scope {scope} for projection_intensities")
                                    test_failed = True
                                elif json.dumps(v[scope]['projections'], cls=QuantityEncoder) != json.dumps(vref[scope]['projections'], cls=QuantityEncoder):
                                    print(f"projected_intensities differ for scope {scope}")
                                    print(f"computed {k}:\n{json.dumps(v[scope]['projections'], cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref[scope]['projections'], cls=QuantityEncoder)}\n\n")
                                    test_failed = True
                            elif v.get(scope):
                                print(f"reference has no scope {scope} for projection_intensities")
                                # ???? test_failed = True
                        continue
                    try:
                        vref = reference_projections[i][k]
                        if json.dumps(v, cls=QuantityEncoder) != json.dumps(vref, cls=QuantityEncoder):
                            print(f"computed {k}:\n{json.dumps(v, cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref, cls=QuantityEncoder)}\n\n")
                            test_failed = True
                    except KeyError as e:
                        print(f"missing in reference: {k}: {json.dumps(v)}\n\n")
                        test_failed = True

        self.assertEqual(test_failed, False)


    def test_upper_lower(self):
        test_failed = False
        projections = EITrajectoryProjector(ProjectionControls(UPPER_PERCENTILE=0.9, LOWER_PERCENTILE=0.1)).project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]

        for i in range(len(projections_dict)):
            if json.dumps(projections_dict[i], cls=QuantityEncoder) != json.dumps(reference_projections[i], cls=QuantityEncoder):
                print(f"Differences in projections_dict[{i}]: company_name = {projections_dict[i]['company_name']}; company_id = {projections_dict[i]['company_id']}")
                for k, v in projections_dict[i].items():
                    if k == 'ghg_s1s2' and not reference_projections[i].get(k):
                        print("ghg_s1s2")
                        continue
                    if k == 'projected_intensities':
                        for scope in projections_dict[i][k]:
                            if reference_projections[i][k].get(scope):
                                vref = reference_projections[i][k]
                                if not v.get(scope):
                                    print(f"projection has no scope {scope} for projection_intensities")
                                    test_failed = True
                                elif json.dumps(v[scope]['projections'], cls=QuantityEncoder) != json.dumps(vref[scope]['projections'], cls=QuantityEncoder):
                                    print(f"projected_intensities differ for scope {scope}")
                                    print(f"computed {k}:\n{json.dumps(v[scope]['projections'], cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref[scope]['projections'], cls=QuantityEncoder)}\n\n")
                                    test_failed = True
                            elif v.get(scope):
                                print(f"reference has no scope {scope} for projection_intensities")
                                # ???? test_failed = True
                        continue
                    try:
                        vref = reference_projections[i][k]
                        if json.dumps(v, cls=QuantityEncoder) != json.dumps(vref, cls=QuantityEncoder):
                            print(f"computed {k}:\n{json.dumps(v, cls=QuantityEncoder)}\n\nreference {k}:\n{json.dumps(vref, cls=QuantityEncoder)}\n\n")
                            test_failed = True
                    except KeyError as e:
                        print(f"missing in reference: {k}: {json.dumps(v)}\n\n")
                        test_failed = True

        self.assertEqual(test_failed, False)


if __name__ == "__main__":
    s = pd.Series([1, None, 3], dtype=float)
    s.interpolate(method="linear")
    test = TestProjector()
    test.setUp()
    test.test_project()
