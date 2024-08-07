import unittest

import pandas as pd

import ITR  # noqa F401
from ITR.configs import TemperatureScoreConfig
from ITR.data.osc_units import (
    Q_,
    BenchmarkMetric,
    EI_Metric,
    EI_Quantity,
    ProductionMetric,
)
from ITR.interfaces import (
    EScope,
    IBenchmark,
    ICompanyData,
    ICompanyEIProjection,
    ICompanyEIProjections,
    ICompanyEIProjectionsScopes,
    ITargetData,
    UProjection,
)


class TestInterfaces(unittest.TestCase):
    """Test the interfaces."""

    def setUp(self) -> None:
        """ """
        pass

    def test_tcre(self):
        tsc = TemperatureScoreConfig()
        assert tsc.CONTROLS_CONFIG.tcre == Q_(2.2, "delta_degC")
        self.assertEqual(
            tsc.CONTROLS_CONFIG.tcre_multiplier,
            Q_(0.0006004366812227075, "delta_degC/(Gt CO2)"),
        )
        print(
            f"tcre_multiplier: {tsc.CONTROLS_CONFIG.tcre_multiplier} == {Q_(0.0006004366812227075, 'delta_degC/(Gt CO2)')}"
        )

    def test_Escope(self):
        self.assertEqual(
            EScope.get_result_scopes(),
            [EScope.S1, EScope.S1S2, EScope.S3, EScope.S1S2S3],
        )

    def test_ProductionMetric(self):
        x = ProductionMetric("MWh")
        print(f"\n ProductionMetric('MWh'): {x}\n\n")

    def test_IProjection(self):
        row = pd.Series([0.9, 0.8, 0.7], index=[2019, 2020, 2021], name="ei_bm")

        bm = IBenchmark(  # noqa: F841
            region="North America",
            sector="Steel",
            benchmark_metric=BenchmarkMetric("dimensionless"),
            projections_nounits=[
                UProjection(year=int(k), value=v) for k, v in row.items()
            ],
        )

    def test_ICompanyProjectionScopes(self):
        row = pd.Series([0.9, 0.8, 0.7], index=[2019, 2020, 2021], name="nl_steel")
        p = [
            ICompanyEIProjection(
                year=int(k), value=EI_Quantity(Q_(v, "t CO2/(t Steel)"))
            )
            for k, v in row.items()
        ]
        S1S2 = ICompanyEIProjections(
            projections=p, ei_metric=EI_Metric("t CO2/(t Steel)")
        )
        x = ICompanyEIProjectionsScopes(S1S2=S1S2)  # noqa: F841

    def test_ICompanyData(self):
        company_data = ICompanyData(  # noqa: F841
            company_name="Company AV",
            company_id="US6293775085",
            region="Europe",
            sector="Steel",
            emissions_metric="t CO2",
            production_metric="t Steel",
            target_probability=0.123,
            projected_targets=None,
            projected_intensities=None,
            country="US6293775085",
            ghg_s1s2=Q_(89800001.4, "t CO2"),
            ghg_s3="89800001.4 t CO2",
            base_year_production="71500001.3960884 t Steel",
            company_revenue=Q_(7370536918, "USD"),
        )

    def test_ITargetData(self):
        target_data = ITargetData(  # noqa: F841
            netzero_year=2022,
            target_type="Absolute",
            target_scope=EScope.S1S2,
            target_start_year=2020,
            target_base_year=2018,
            target_end_year=2040,
            target_base_year_qty=2.0,
            target_base_year_unit="t CO2",
            target_reduction_pct=0.2,
        )

    def test_fail_ITargetData(self):
        with self.assertRaises(ValueError):
            target_data = ITargetData(  # noqa: F841
                netzero_year=2022,
                target_type="absolute",
                target_scope=EScope.S1S2,
                target_start_year=2020,
                target_base_year=2018,
                target_end_year=2020,  # This value should be larger than start year
                target_base_year_qty=2.0,
                target_base_year_unit="t CO2",
                target_reduction_pct=0.2,
            )
