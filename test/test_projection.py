import json
import unittest
import os

from ITR.data.base_providers import EmissionIntensityProjector
from ITR.interfaces import ICompanyData


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
        self.projector = EmissionIntensityProjector()

    def test_project(self):
        projections = self.projector.project_intensities(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        projections_dict = [projection.dict() for projection in projections]
        self.assertEqual(json.dumps(projections_dict), json.dumps(reference_projections))
