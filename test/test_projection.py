import json
import unittest
import os
import datetime

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
        for company_dict in company_dicts:
            # TODO: fix json input and reference files!
            company_dict['report_date'] = datetime.date(2021, 12, 31)
        self.companies = [ICompanyData(**company_dict) for company_dict in company_dicts]
        self.projector = EITrajectoryProjector()

    def test_project(self):
        test_failed = False
        projections = self.projector.project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]
        
        # json.dumps(projections_dict[0],default=mystr)
        # json.dumps(reference_projections[0],default=refstr)
        for i in range(len(projections_dict)):
            del(projections_dict[i]['target_data']) # We are testing trajectory projections, not target projections
            del(projections_dict[i]['base_year_production']) # Use for target, not trajectory projections
            del(projections_dict[i]['company_ev_plus_cash']) # Not computed by trajectory code
            del(projections_dict[i]['emissions_metric']) # Not used by trajectory code
            del(projections_dict[i]['production_metric']) # Not used by trajectory code
            if json.dumps(projections_dict[i],default=mystr)!=json.dumps(reference_projections[i],default=refstr):
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
                                elif json.dumps(v[scope]['projections'],default=mystr)!=json.dumps(vref[scope]['projections'],default=refstr):
                                    print(f"projected_intensities differ for scope {scope}")
                                    print(f"computed {k}:\n{json.dumps(v[scope]['projections'],default=mystr)}\n\nreference {k}:\n{json.dumps(vref[scope]['projections'],default=refstr)}\n\n")
                                    test_failed = True
                            elif v.get(scope):
                                print(f"reference has no scope {scope} for projection_intensities")
                                # ???? test_failed = True
                        continue
                    try:
                        vref = reference_projections[i][k]
                        if json.dumps(v,default=mystr)!=json.dumps(vref,default=refstr):
                            print(f"computed {k}:\n{json.dumps(v,default=mystr)}\n\nreference {k}:\n{json.dumps(vref,default=refstr)}\n\n")
                            test_failed = True
                    except KeyError as e:
                        print(f"missing in reference: {k}: {json.dumps(v,default=mystr)}\n\n")
                        test_failed = True

        self.assertEqual(test_failed, False)

import pandas as pd

if __name__ == "__main__":
    s = pd.Series([1, None, 3], dtype=float)
    s.interpolate(method="linear")
    test = TestProjector()
    test.setUp()
    test.test_project()
