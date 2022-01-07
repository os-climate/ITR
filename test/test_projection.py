import unittest
import pandas as pd
import os
from ITR.data.data_providers import EmissionIntensityProjector


class TestProjector(unittest.TestCase):
    """
    Test the projector that converts historic data into emission intensity projections
    """

    def setUp(self) -> None:
        self.root: str = os.path.dirname(os.path.abspath(__file__))
        self.source_path: str = os.path.join(self.root, "inputs", "test_data_company.xlsx")
        self.reference_path: str = os.path.join(self.root, "inputs", "test_projection_reference.csv")

        intensities_historic: pd.DataFrame = pd.read_excel(self.source_path, 'historic_data')
        self.projector = EmissionIntensityProjector(intensities_historic)

    def test_project(self):
        projections = self.projector.project()

        # Column names from read_csv are read as strings
        projections.columns = [str(col) for col in projections.columns]
        reference = pd.read_csv(self.reference_path)
        pd.testing.assert_frame_equal(projections, reference)