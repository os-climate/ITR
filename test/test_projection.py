import json
import unittest
import os
import datetime
import pandas as pd

from utils import QuantityEncoder
from ITR.data.base_providers import EITrajectoryProjector
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
        for company_dict in company_dicts:
            company_dict['report_date'] = datetime.date(2021, 12, 31)
        self.companies = [ICompanyData(**company_dict) for company_dict in company_dicts]
        self.projector = EITrajectoryProjector()

    def test_project(self):
        test_failed = False
        projections = self.projector.project_ei_trajectories(self.companies)
        with open(self.json_reference_path, 'r') as file:
            reference_projections = json.load(file)

        # Column names from read_csv are read as strings
        projections.columns = [str(col) for col in projections.columns]
        reference = pd.read_csv(self.reference_path)
        pd.testing.assert_frame_equal(projections, reference)
