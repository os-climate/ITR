import os
import unittest

import pandas as pd

from ITR.interfaces import EScope, IDataProviderTarget


class TestInterfaces(unittest.TestCase):
    """
    Test the interfaces.
    """

    def setUp(self) -> None:
        """
        """
        pass

    def test_Escope(self):
        self.assertEqual(EScope.get_result_scopes(), [EScope.S1S2, EScope.S3, EScope.S1S2S3])

    def test_IDataProviderTarget(self):
        self.assertEqual(IDataProviderTarget.validate_e("test"), "test")
        self.assertIsNone(IDataProviderTarget.validate_e(""))
        self.assertIsNone(IDataProviderTarget.validate_e("nan"))


