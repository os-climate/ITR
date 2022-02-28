import unittest
import pandas as pd


def assert_pint_series_equal(case: unittest.case, left: pd.Series, right: pd.Series):
    # Helper function to avoid bug in pd.testing.assert_series_equal concerning pint series
    for d, data in enumerate(left):
        case.assertAlmostEqual(data, right[d])

    for d, data in enumerate(right):
        case.assertAlmostEqual(data, left[d])


def assert_pint_frame_equal(case: unittest.case, left: pd.DataFrame, right: pd.DataFrame):
    # Helper function to avoid bug in pd.testing.assert_frame_equal concerning pint series
    left_flat = left.values.flatten()
    right_flat = right.values.flatten()

    errors = []
    for d, data in enumerate(left_flat):
        try:
            case.assertAlmostEqual(data, right_flat[d])
        except AssertionError as e:
            errors.append(e.args[0])
    if errors:
        raise AssertionError('\n'.join(errors))

    for d, data in enumerate(right_flat):
        case.assertAlmostEqual(data, left_flat[d])