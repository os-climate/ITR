import datetime
import json
import os
import unittest
import warnings
from typing import List

import pandas as pd
from utils import ITR_Encoder, assert_pint_series_equal

import ITR  # noqa F401
from ITR import data_dir
from ITR.configs import ColumnsConfig, VariablesConfig
from ITR.data.base_providers import (
    BaseProviderProductionBenchmark,
    EITargetProjector,
    EITrajectoryProjector,
)
from ITR.data.osc_units import PA_, Q_, asPintDataFrame
from ITR.interfaces import (
    EScope,
    ICompanyData,
    IProductionBenchmarkScopes,
    ITargetData,
    ProjectionControls,
)


def is_pint_dict_equal(result: List[dict], reference: List[dict]) -> bool:
    is_equal = True
    for i in range(len(result)):
        if json.dumps(result[i], cls=ITR_Encoder) != json.dumps(reference[i], cls=ITR_Encoder):
            print(
                f"Differences in projections_dict[{i}]: company_name = {result[i]['company_name']}; company_id = {result[i]['company_id']}"
            )
            for k, v in result[i].items():
                if k == "ghg_s1s2" and not reference[i].get(k):
                    print("ghg_s1s2")
                    continue
                if k in ["projected_intensities", "projected_targets"]:
                    if not result[i][k]:
                        continue
                    for scope in result[i][k]:
                        if reference[i][k].get(scope):
                            vref = reference[i][k]
                            if not v.get(scope):
                                print(f"projection has no scope {scope} for {k}")
                                is_equal = False
                                continue
                            vproj = v[scope]["projections"]
                            if isinstance(vproj, pd.Series):
                                vproj = [
                                    {
                                        "year": k,
                                        "value": " ".join(["{:.5f}".format(v.m), str(v.u)]),
                                    }
                                    for k, v in vproj.to_dict().items()
                                ]
                            if json.dumps(vproj, cls=ITR_Encoder) != json.dumps(
                                vref[scope]["projections"], cls=ITR_Encoder
                            ):
                                print(f"{k} differ for scope {scope}")
                                print(
                                    #
                                    f"computed {k}:\n{json.dumps(vproj, cls=ITR_Encoder)}\n\nreference {k}:\n{json.dumps(vref[scope]['projections'], cls=ITR_Encoder)}\n\n"  # noqa: E501
                                )
                                is_equal = False
                        elif v.get(scope):
                            print(f"reference has no scope {scope} for projection_intensities")
                            # ???? is_equal = False
                    continue
                try:
                    vref = reference[i][k]
                    if json.dumps(v, cls=ITR_Encoder) != json.dumps(vref, cls=ITR_Encoder):
                        print(
                            f"computed {k}:\n{json.dumps(v, cls=ITR_Encoder)}\n\nreference {k}:\n{json.dumps(vref, cls=ITR_Encoder)}\n\n"
                        )
                        is_equal = False
                except KeyError as exc:  # noqa: F841
                    print(f"missing in reference: {k}: {json.dumps(v)}\n\n")
                    is_equal = False

    return is_equal


class TestProjector(unittest.TestCase):
    """
    Test the projector that converts historic data into emission intensity projections
    """

    def setUp(self) -> None:
        self.root: str = os.path.dirname(os.path.abspath(__file__))
        self.source_path: str = os.path.join(self.root, "inputs", "json", "test_project_companies.json")
        self.json_reference_path: str = os.path.join(self.root, "inputs", "json", "test_project_reference.json")
        self.benchmark_prod_json = os.path.join(data_dir, "benchmark_production_OECM.json")
        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.model_validate(parsed_json)
        self.base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

        with open(self.source_path, "r") as file:
            company_dicts = json.load(file)
        for company_dict in company_dicts:
            company_dict["report_date"] = datetime.date(2021, 12, 31)
        self.companies = [ICompanyData(**company_dict) for company_dict in company_dicts]
        self.projector = EITrajectoryProjector()
        with open(self.json_reference_path, "r") as file:
            itr_encoder = ITR_Encoder()
            self.reference_projections = json.load(file)

            def fixup_json_list_values(json_list):
                if json_list is None:
                    return
                for item in json_list:
                    if not item["value"].startswith("nan"):
                        item["value"] = itr_encoder.encode(Q_(item["value"])).strip('"')

            for refp in self.reference_projections:
                for k, vref in refp.items():
                    if not vref:
                        continue
                    if k in ["base_year_production", "ghg_s1s2"]:
                        refp[k] = itr_encoder.encode(Q_(vref)).strip('"')
                        continue
                    if k == "historic_data":
                        for hd_key in vref:
                            if hd_key == "productions":
                                fixup_json_list_values(vref[hd_key])
                            else:
                                if vref[hd_key]:
                                    for scope in vref[hd_key]:
                                        if vref[hd_key][scope]:
                                            fixup_json_list_values(vref[hd_key][scope])

                    elif k in ["projected_intensities", "projected_targets"]:
                        for scope in vref:
                            if vref[scope]:
                                fixup_json_list_values(vref[scope]["projections"])

    def test_trajectories(self):
        projections = self.projector.project_ei_trajectories(self.companies)

        projections_dict = [projection.model_dump() for projection in projections]
        test_successful = is_pint_dict_equal(projections_dict, self.reference_projections)
        self.assertEqual(test_successful, True)

    def test_targets(self):
        # Test that both absolute targets and intensity targets produce sane results
        # Our ICompanyData need only have: company_id, company_name, base_year_production, ghg_scope12, sector, region, target_data
        # We need bm production data in case we need to convert supplied emissions data to calculate intensity targets or
        # supplied intensity data to compute absolute targets
        company_data = self.companies
        company_dict = {
            field: [getattr(c, field) for c in company_data]
            for field in [
                ColumnsConfig.BASE_YEAR_PRODUCTION,
                ColumnsConfig.GHG_SCOPE12,
                ColumnsConfig.SECTOR,
                ColumnsConfig.REGION,
            ]
        }
        company_dict[ColumnsConfig.SCOPE] = [EScope.S1S2] * len(company_data)
        company_index = [c.company_id for c in company_data]
        company_sector_region_info = pd.DataFrame(company_dict, pd.Index(company_index, name="company_id"))
        bm_production_data = self.base_production_bm.get_company_projected_production(company_sector_region_info)
        expected_0 = pd.Series(
            PA_(
                [
                    0.131,  # 2019
                    0.123,
                    0.116,
                    0.108,
                    0.102,
                    0.096,
                    0.09,
                    0.084,
                    0.079,
                    0.074,
                    0.07,  # 2020-2029
                    0.066,
                    0.058,
                    0.051,
                    0.045,
                    0.039,
                    0.034,
                    0.03,
                    0.026,
                    0.023,
                    0.02,  # 2030-2039
                    0.017,
                    0.014,
                    0.012,
                    0.01,
                    0.008,
                    0.006,
                    0.005,
                    0.003,
                    0.002,
                    0.001,  # 2040-2049
                    0.0,
                ],  # 2050
                dtype="pint[t CO2/GJ]",
            ),
            index=range(2019, 2051),
            name="expected_0",
        )
        target_0 = ITargetData(
            **{
                "netzero_year": 2050,
                "target_type": "intensity",
                "target_scope": EScope.S1S2,
                "target_start_year": 2020,
                "target_base_year": 2019,
                "target_end_year": 2030,
                "target_base_year_qty": 0.131037611,
                "target_base_year_unit": "t CO2/GJ",
                "target_reduction_pct": 0.5,
            }
        )
        expected = [expected_0]
        company_data[0].target_data = [target_0]
        for i, c in enumerate(company_data):
            if c.target_data:
                projected_targets = EITargetProjector(self.projector.projection_controls).project_ei_targets(
                    c,
                    bm_production_data.loc[(c.company_id, EScope.S1S2)],
                    ei_df_t=pd.DataFrame(),
                )
                test_projection = projected_targets.S1S2
                if isinstance(test_projection.projections, pd.Series):
                    test_projection = test_projection.projections
                else:
                    test_projection = pd.Series(
                        PA_(
                            [x.value.m_as(test_projection.ei_metric) for x in test_projection.projections],
                            dtype=test_projection.ei_metric,
                        ),
                        index=range(2019, 2051),
                    )
                assert_pint_series_equal(self, test_projection, expected[i], places=3)
            else:
                assert c.projected_targets.empty

    def test_extrapolate(self):
        with open(os.path.join(self.root, "inputs", "json", "test_fillna_companies.json"), "r") as file:
            company_dicts = json.load(file)
        for company_dict in company_dicts:
            company_dict["report_date"] = datetime.date(2021, 12, 31)
        fillna_data = [ICompanyData(**company_dict) for company_dict in company_dicts]

        ei_projector = EITrajectoryProjector(ProjectionControls(UPPER_PERCENTILE=0.9, LOWER_PERCENTILE=0.1))
        historic_df = ei_projector._extract_historic_df(fillna_data)
        ei_projector._align_and_compute_missing_historic_ei(fillna_data, historic_df)

        historic_years = [column for column in historic_df.columns if isinstance(column, int)]
        projection_years = range(max(historic_years), ei_projector.projection_controls.TARGET_YEAR + 1)
        with warnings.catch_warnings():
            # Don't worry about warning that we are intentionally dropping units as we transpose
            warnings.simplefilter("ignore")
            historic_intensities_t = asPintDataFrame(
                historic_df[historic_years].query(f"variable=='{VariablesConfig.EMISSIONS_INTENSITIES}'").T
            ).pint.dequantify()
        standardized_intensities_t = ei_projector._standardize(historic_intensities_t)
        intensity_trends_t = ei_projector._get_trends(standardized_intensities_t)
        extrapolated_t = ei_projector._extrapolate(intensity_trends_t, projection_years, historic_intensities_t)
        # Restore row-wise shape of DataFrame
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pint units don't like being twisted from columns to rows, but it's ok
            ei_projector._add_projections_to_companies(fillna_data, extrapolated_t.pint.quantify())
        # Figure out what to test here--just making it through with funny company data is step 1!

    # Need test data in order to test mean
    def test_median(self):
        projections = EITrajectoryProjector(
            ProjectionControls(TREND_CALC_METHOD=pd.DataFrame.median)
        ).project_ei_trajectories(self.companies)
        projections_dict = [projection.model_dump() for projection in projections]

        test_successful = is_pint_dict_equal(projections_dict, self.reference_projections)
        self.assertEqual(test_successful, True)

    def test_upper_lower(self):
        projections = EITrajectoryProjector(
            ProjectionControls(UPPER_PERCENTILE=0.9, LOWER_PERCENTILE=0.1)
        ).project_ei_trajectories(self.companies)
        projections_dict = [projection.model_dump() for projection in projections]
        test_successful = is_pint_dict_equal(projections_dict, self.reference_projections)
        self.assertEqual(test_successful, True)


if __name__ == "__main__":
    s = pd.Series([1, None, 3], dtype=float)
    s.interpolate(method="linear")
    test = TestProjector()
    test.setUp()
    test.test_trajectories()
    test.test_targets()
