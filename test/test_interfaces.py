import os
import unittest

import pandas as pd
from pint import Quantity

from ITR.data.osc_units import ureg, Q_, PA_

from ITR.interfaces import EScope, PowerGeneration, IntensityMetric, IProjection, IBenchmark, ICompanyData, \
    ICompanyEIProjectionsScopes, ICompanyEIProjections, ITargetData


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

    def test_PowerGeneration(self):
        x = PowerGeneration(units='MWh')
        print(f"\n PowerGeneration: x.units = {x.units}\n\n")

    def test_IProjection(self):
        row = pd.Series([0.9, 0.8, 0.7],
                       index=[2019, 2020, 2021],
                       name='ei_bm')

        bm = IBenchmark(region='North America', sector='Steel', benchmark_metric={'units':'dimensionless'},
                        projections=[IProjection(year=int(k), value=Q_(v, ureg('dimensionless'))) for k, v in row.items()])

    def test_ICompanyProjectionScopes(self):
        row = pd.Series([0.9, 0.8, 0.7],
                       index=[2019, 2020, 2021],
                       name='nl_steel')
        p = [IProjection(year=int(k), value=Q_(v, ureg('Fe_ton'))) for k, v in row.items()]
        S1S2=ICompanyEIProjections(projections=p, ei_metric=IntensityMetric.parse_obj({'units':'t CO2/Fe_ton'}))
        x = ICompanyEIProjectionsScopes(S1S2=S1S2)

    def test_ICompanyData(self):
        company_data = ICompanyData(
            company_name="Company AV",
            company_id="US6293775085",
            region="Europe",
            sector="Steel",
            emissions_metric={"units": "t CO2"},
            production_metric={"units": "Fe_ton"},
            target_probability=0.123,
            projected_targets = None,
            projected_intensities = None,
            country='US6293775085',
            ghg_s1s2=89800001.4,
            ghg_s3=89800001.4,
            base_year_production=71500001.3960884,
            company_revenue=7370536918
        )

    def test_ICompanyData_S1S2(self):
        exp_s1 = 1234
        exp_s2 = 5678
        exp_s1s2 = exp_s1 + exp_s2

        # Test saving S1S2 from args
        cd = ICompanyData(
            company_name = "cd1",
            company_id = 1,
            region = "Somewhere",
            sector = "Steel",
            ghg_s1s2=exp_s1s2,
        )
        self.assertEqual(cd.ghg_s1, None)
        self.assertEqual(cd.ghg_s2, None)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)

        # Test saving S1, S2 from args
        cd = ICompanyData(
            company_name = "cd2",
            company_id = 2,
            region = "Somewhere",
            sector = "Steel",
            ghg_s1=exp_s1,
            ghg_s2=exp_s2,
        )
        self.assertEqual(cd.ghg_s1.magnitude, exp_s1)
        self.assertEqual(cd.ghg_s2.magnitude, exp_s2)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)

        # Test saving S1S2 from history
        hd = {'emissions' : {
            'S1' : [ ],
            'S2' : [ ],
            'S1S2' : [{'year' : 2022, 'value' : Quantity(value = exp_s1s2)}],
            'S3' : [ ],
            'S1S2S3': [ ],
            }}
        cd = ICompanyData(
            company_name = "cd3",
            company_id = 3,
            region = "Somewhere",
            sector = "Steel",
            historic_data = hd,
        )
        self.assertEqual(cd.ghg_s1, None)
        self.assertEqual(cd.ghg_s2, None)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)

        # Test saving S1, S2 from history
        hd = {'emissions' : {
            'S1' : [{'year' : 2022, 'value' : Quantity(value = exp_s1)}],
            'S2' : [{'year' : 2022, 'value' : Quantity(value = exp_s2)}],
            'S1S2' : [ ],
            'S3' : [ ],
            'S1S2S3': [ ],
            }}
        cd = ICompanyData(
            company_name = "cd4",
            company_id = 4,
            region = "Somewhere",
            sector = "Steel",
            historic_data = hd,
        )
        self.assertEqual(cd.ghg_s1.magnitude, exp_s1)
        self.assertEqual(cd.ghg_s2.magnitude, exp_s2)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)

        # Test priority of S1, S2 over S1S2, in args
        cd = ICompanyData(
            company_name = "cd5",
            company_id = 5,
            region = "Somewhere",
            sector = "Steel",
            ghg_s1=exp_s1,
            ghg_s2=exp_s2,
            ghg_s1s2=8888,
        )
        self.assertEqual(cd.ghg_s1.magnitude, exp_s1)
        self.assertEqual(cd.ghg_s2.magnitude, exp_s2)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)

        # Test priority of S1, S2 over S1S2, in history
        hd = {'emissions' : {
            'S1' : [{'year' : 2022, 'value' : Quantity(value = exp_s1)}],
            'S2' : [{'year' : 2022, 'value' : Quantity(value = exp_s2)}],
            'S1S2' : [{'year' : 2022, 'value' : Quantity(value = 8888)}],
            'S3' : [ ],
            'S1S2S3': [ ],
            }}
        cd = ICompanyData(
            company_name = "cd6",
            company_id = 6,
            region = "Somewhere",
            sector = "Steel",
            historic_data = hd,
        )
        self.assertEqual(cd.ghg_s1.magnitude, exp_s1)
        self.assertEqual(cd.ghg_s2.magnitude, exp_s2)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)

        # Test priority of args over history
        hd = {'emissions' : {
            'S1' : [ ],
            'S2' : [ ],
            'S1S2' : [{'year' : 2022, 'value' : Quantity(value = 8888)}],
            'S3' : [ ],
            'S1S2S3': [ ],
            }}
        cd = ICompanyData(
            company_name = "cd7",
            company_id = 7,
            region = "Somewhere",
            sector = "Steel",
            historic_data = hd,
            ghg_s1s2=exp_s1s2,
        )
        self.assertEqual(cd.ghg_s1, None)
        self.assertEqual(cd.ghg_s2, None)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)

    def test_ICompanyData_S3(self):
        exp_s1s2 = 1234
        exp_s3 = 5678

        # Test saving S3 from args
        cd = ICompanyData(
            company_name = "cd1",
            company_id = 1,
            region = "Somewhere",
            sector = "Steel",
            ghg_s1s2=exp_s1s2,
            ghg_s3=exp_s3,
        )
        self.assertEqual(cd.ghg_s1, None)
        self.assertEqual(cd.ghg_s2, None)
        self.assertEqual(cd.ghg_s1s2.magnitude, exp_s1s2)
        self.assertEqual(cd.ghg_s3.magnitude, exp_s3)

        # Test saving S3 from history
        hd = {'emissions' : {
            'S1' : [ ],
            'S2' : [ ],
            'S1S2' : [ ],
            'S3' : [ {'year' : 2022, 'value' : Quantity(value = exp_s3)} ],
            'S1S2S3': [ ],
            }}
        cd = ICompanyData(
            company_name = "cd3",
            company_id = 3,
            region = "Somewhere",
            sector = "Steel",
            ghg_s1s2=exp_s1s2,
            historic_data = hd,
        )
        self.assertEqual(cd.ghg_s1, None)
        self.assertEqual(cd.ghg_s2, None)
        self.assertEqual(cd.ghg_s3.magnitude, exp_s3)

    def test_ITargetData(self):
        target_data = ITargetData(
            netzero_year=2022,
            target_type='Absolute',
            target_scope=EScope.S1S2,
            target_start_year=2020,
            target_base_year=2018,
            target_end_year=2040,
            target_base_year_qty=2.0,
            target_base_year_unit='t CO2',
            target_reduction_pct=0.2
        )

    def test_fail_ITargetData(self):
        with self.assertRaises(ValueError):
            target_data = ITargetData(
                netzero_year=2022,
                target_type='absolute',
                target_scope=EScope.S1S2,
                target_start_year=2020,
                target_base_year=2018,
                target_end_year=2020,  # This value should be larger than 2022
                target_base_year_qty=2.0,
                target_base_year_unit='t CO2',
                target_reduction_pct=0.2
            )
