import unittest
import pandas as pd
import numpy as np
from pandas._testing import *

from pint import set_application_registry
from pint_pandas import PintArray, PintType
from openscm_units import unit_registry
PintType.ureg = unit_registry
ureg = unit_registry
set_application_registry(ureg)
Q_ = ureg.Quantity

ureg.define("CO2e = CO2 = CO2eq = CO2_eq")

pd.show_versions()

def pandas_mult_acc(a, b):
    df = a.multiply(b)
    return df.sum(axis=1)

def pint_mult_acc(a, b):
    df = a.multiply(b)
    return df.sum(axis=1).astype('pint[g CO2]')

class TestBaseProvider(unittest.TestCase):
    """
    Test the Base provider
    """

    def setUp(self) -> None:
        pass

    # PASS: series are equal
    def test_pandas_series_equality_1(self):
        projected_ei = pd.DataFrame([[1.0, 2.0], [4.0, 2.0]])
        projected_production = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]])
        expected_data = pd.Series([5.0, 8.0], index=[0, 1])
        result_data = pandas_mult_acc(projected_ei,projected_production)
        pd.testing.assert_series_equal(expected_data, result_data)

    # FAIL: series differ
    def test_pandas_series_equality_2(self):
        projected_ei = pd.DataFrame([[1.0, 2.0], [4.0, 2.0]])
        projected_production = pd.DataFrame([[1.0, 2.0], [1.0, 3.0]])
        expected_data = pd.Series([5.0, 8.0], index=[0, 1])
        result_data = pandas_mult_acc(projected_ei,projected_production)
        pd.testing.assert_series_equal(expected_data, result_data)

    # PASS: series are equal
    def test_pint_series_equality_1(self):
        projected_ei = pd.DataFrame([[Q_(1.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')], [Q_(4.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')]], dtype='pint[g CO2/Wh]')
        projected_production = pd.DataFrame([[Q_(1.0, 'Wh'), Q_(2.0, 'Wh')], [Q_(1.0, 'Wh'), Q_(2.0, 'Wh')]], dtype='pint[Wh]')
        expected_data = pd.Series([5.0, 8.0], index=[0, 1], dtype='pint[g CO2]')
        result_data = pint_mult_acc(projected_ei,projected_production)
        pd.testing.assert_series_equal(expected_data, result_data)

    # PASS: extension arrays are equal
    def test_pint_series_equality_2(self):
        projected_ei = pd.DataFrame([[Q_(1.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')], [Q_(4.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')]], dtype='pint[g CO2/Wh]')
        projected_production = pd.DataFrame([[Q_(1.0, 'Wh'), Q_(2.0, 'Wh')], [Q_(1.0, 'Wh'), Q_(2.0, 'Wh')]], dtype='pint[Wh]')
        expected_data = pd.Series([5.0, 8.0], index=[0, 1], dtype='pint[g CO2]')
        result_data = pint_mult_acc(projected_ei,projected_production)
        pd.testing.assert_extension_array_equal(expected_data.values, result_data.values)

    # Should FAIL, but ERROR instead
    def test_pint_series_equality_3(self):
        projected_ei = pd.DataFrame([[Q_(1.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')], [Q_(4.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')]], dtype='pint[g CO2/Wh]')
        projected_production = pd.DataFrame([[Q_(1.0, 'Wh'), Q_(2.0, 'Wh')], [Q_(1.0, 'Wh'), Q_(3.0, 'Wh')]], dtype='pint[Wh]')
        expected_data = pd.Series([5.0, 8.0], index=[0, 1], dtype='pint[g CO2]')
        result_data = pint_mult_acc(projected_ei,projected_production)
        # Expected to fail because expected data and result data differ,
        pd._testing.assert_series_equal(expected_data, result_data)

    # Should FAIL, but ERROR instead
    def test_pint_series_equality_4(self):
        projected_ei = pd.DataFrame([[Q_(1.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')], [Q_(4.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')]], dtype='pint[g CO2/Wh]')
        projected_production = pd.DataFrame([[Q_(1.0, 'Wh'), Q_(2.0, 'Wh')], [Q_(1.0, 'Wh'), Q_(3.0, 'Wh')]], dtype='pint[Wh]')
        expected_data = pd.Series([5.0, 8.0], index=[0, 1], dtype='pint[g CO2]')
        result_data = pint_mult_acc(projected_ei,projected_production)
        # Expected to fail because expected data and result data differ
        pd._testing.assert_extension_array_equal(expected_data.values, result_data.values)

    # FAIL: numpy arrays differ differ
    def test_pint_series_equality_5(self):
        projected_ei = pd.DataFrame([[Q_(1.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')], [Q_(4.0, 'g CO2/Wh'), Q_(2.0, 'g CO2/Wh')]], dtype='pint[g CO2/Wh]')
        projected_production = pd.DataFrame([[Q_(1.0, 'Wh'), Q_(2.0, 'Wh')], [Q_(1.0, 'Wh'), Q_(3.0, 'Wh')]], dtype='pint[Wh]')
        expected_data = pd.Series([5.0, 8.0], index=[0, 1], dtype='pint[g CO2]')
        result_data = pint_mult_acc(projected_ei,projected_production)
        # Expected to fail because expected data and result data differ
        pd._testing.assert_numpy_array_equal(np.asarray(expected_data), np.asarray(result_data))
