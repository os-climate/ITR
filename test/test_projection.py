import json
import unittest
import os

from ITR.data.base_providers import EITrajectoryProjector
from ITR.interfaces import ICompanyData


def mystr(s):
    t = str(s).replace('CO2 * metric_ton', 't CO2').replace('gigajoule','GJ').replace(' / ', '/')
    if t.startswith('nan'):
        return json.loads('NaN')
    return t


def refstr(s):
    if s!=s:
        return json.loads('NaN')
    return str(s)


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
        self.companies = [ICompanyData(**company_dict) for company_dict in company_dicts]
        self.projector = EITrajectoryProjector()

    def test_project(self):
        projections = self.projector.project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]
        
        # json.dumps(projections_dict[0],default=mystr)
        # json.dumps(reference_projections[0],default=refstr)
        for i in range(len(projections_dict)):
            del(projections_dict[i]['production_metric'])
            if json.dumps(projections_dict[i],default=mystr)!=json.dumps(reference_projections[i],default=refstr):
                print(f"Differences in projections_dict[{i}]: company_name = {projections_dict[i]['company_name']}; company_id = {projections_dict[i]['company_id']}")
                for k, v in projections_dict[i].items():
                    vref = reference_projections[i][k]
                    if json.dumps(v,default=mystr)!=json.dumps(vref,default=refstr):
                        print(f"{k}:\n{json.dumps(v,default=mystr)}\n\n{json.dumps(vref,default=refstr)}\n\n")

        self.assertEqual(json.dumps(projections_dict,default=mystr), json.dumps(reference_projections,default=refstr))

import pandas as pd

if __name__ == "__main__":
    s = pd.Series([1, None, 3], dtype=float)
    s.interpolate(method="linear")
    test = TestProjector()
    test.setUp()
    test.test_project()
