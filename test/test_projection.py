import json
import unittest
import pandas as pd
import os
from ITR.data.data_providers import EmissionIntensityProjector
from ITR.interfaces import ICompanyData


class TestProjector(unittest.TestCase):
    """
    Test the projector that converts historic data into emission intensity projections
    """

    def setUp(self) -> None:
        self.root: str = os.path.dirname(os.path.abspath(__file__))
        self.source_path: str = os.path.join(self.root, "inputs", "json", "test_project_companies.json")
        self.reference_path: str = os.path.join(self.root, "inputs", "test_projection_reference.csv")

        with open(self.source_path, 'r') as file:
            company_dicts = json.load(file)
        companies = [ICompanyData(**company_dict) for company_dict in company_dicts]
        self.projector = EmissionIntensityProjector(companies)

    def test_project(self):
        projections = self.projector.project(as_dataframe=True)

        # Column names from read_csv are read as strings
        projections.columns = [str(col) for col in projections.columns]
        reference = pd.read_csv(self.reference_path)
        pd.testing.assert_frame_equal(projections, reference)
