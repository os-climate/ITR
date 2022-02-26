import os
import unittest

import pandas as pd

from ITR.data.osc_units import ureg, Q_, PA_

from ITR.interfaces import EScope, PowerGenerationWh, IProjection, IBenchmark, ICompanyData, ICompanyEIProjectionsScopes, ICompanyEIProjections


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

    def test_PowerGenerationWh(self):
        x = PowerGenerationWh(units='MWh')
        print(f"\n PowerGenerationWh: x.units = {x.units}\n\n")

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
        S1S2=ICompanyEIProjections(projections=p)
        x = ICompanyEIProjectionsScopes(S1S2=S1S2)

    def test_ICompanyData(self):
        company_data = ICompanyData(
            company_name="Company AV",
            company_id="US6293775085",
            region="Europe",
            sector="Steel",
            emissions_metric={ "units": "t CO2"},
            production_metric={ "units": "Fe_ton"},
            target_probability=0.123,
            projected_targets = None,
            projected_intensities = None,
            country='US6293775085',
            ghg_s1s2=89800001.4,
            ghg_s3=89800001.4,
            company_revenue=7370536918
        )
