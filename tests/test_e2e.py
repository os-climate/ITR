import unittest
from ITR.interfaces import (
    EScope,
    ETimeFrames,
    PortfolioCompany,
)
from ITR.data.osc_units import ureg, Q_, PA_

from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod
import copy
import ITR
from ITR.data.data_warehouse import DataWarehouse
from typing import List
from ITR.interfaces import ICompanyAggregates, ICompanyEIProjectionsScopes, IProjection


class TestDataWareHouse(DataWarehouse):
    def __init__(
            self, companies: List[ICompanyAggregates]
    ):
        self.companies = companies

    def get_preprocessed_company_data(self, company_ids: List[str]) -> List[ICompanyAggregates]:
        return self.companies


class EndToEndTest(unittest.TestCase):
    """
    This class is containing a set of end to end tests:
    - basic flow from creating companies/targets up to calculating aggregated values
    - edge cases for grouping
    - high load tests (>1000 targets)
    - testing of all different input values and running thru the whole process (tbd)
    """

    def setUp(self):
        # base_year is 2019
        company_id = "BaseCompany"
        self.BASE_COMP_SCORE = Q_(3.85, ureg.delta_degC)
        self.company_base = ICompanyAggregates(
            company_name=company_id,
            company_id=company_id,
            base_year_production=IProjection.parse_obj({"year": 2019, "value":Q_(1000000.0, ureg('Fe_ton'))}).value,
            ghg_s1s2=IProjection.parse_obj({"year": 2019, "value":Q_(1698247.4347547039, ureg('t CO2'))}).value,
            ghg_s3=IProjection.parse_obj({"year": 2019, "value":Q_(0.0, ureg('t CO2'))}).value,
            emissions_metric={'units':'t CO2'},
            production_metric={'units':'Fe_ton'},
            company_revenue=100,
            company_market_cap=100,
            company_enterprise_value=100,
            company_total_assets=100,
            company_cash_equivalents=100,
            cumulative_budget="345325664.840567 t CO2",
            cumulative_trajectory="3745094638.52858 t CO2",
            cumulative_target="3769096510.09909 t CO2",
            target_probability=0.428571428571428,
            isic='A12',
            sector='Steel',
            region='Europe',
            benchmark_global_budget="396 Gt CO2",
            benchmark_temperature="1.5 delta_degC",
            projected_intensities=ICompanyEIProjectionsScopes.parse_obj({
                "S1S2": {
                    "ei_metric": {'units': "t CO2/Fe_ton"},
                    "projections": [
                        {
                            "year": "2019",
                            "value": "1.6982474347547039 t CO2/Fe_ton"
                        },
                        {
                            "year": "2020",
                            "value": "1.6982474347547039 t CO2/Fe_ton"
                        },
                        {
                            "year": "2021",
                            "value": "1.5908285727976157 t CO2/Fe_ton"
                        }
                    ]
                }
            }),
            projected_targets=ICompanyEIProjectionsScopes.parse_obj({
                "S1S2": {
                    "ei_metric": {'units': "t CO2/Fe_ton"},
                    "projections": [
                        {
                            "year": "2019",
                            "value": "1.6982474347547039 t CO2/Fe_ton"
                        },
                        {
                            "year": "2020",
                            "value": "1.6982474347547039 t CO2/Fe_ton"
                        },
                        {
                            "year": "2021",
                            "value": "1.5577542305393455 t CO2/Fe_ton"
                        }
                    ]
                }
            })
        )

        # pf
        self.pf_base = PortfolioCompany(
            company_name=company_id,
            company_id=company_id,
            investment_value=100,
            company_isin=company_id,
        )

    def test_basic(self):
        """
        This test is just a very basic workflow going thru all calculations up to temp score
        """

        # Setup test provider
        company = copy.deepcopy(self.company_base)
        data_provider = TestDataWareHouse([company])

        # Calculate Temp Scores
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        # portfolio data
        pf_company = copy.deepcopy(self.pf_base)
        portfolio_data = ITR.utils.get_data(data_provider, [pf_company])

        # Verify data
        scores = temp_score.calculate(portfolio_data)
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores.index), 1)

    def test_chaos(self):
        # TODO: go thru lots of different parameters on company & target level and try to break it
        pass

    def test_basic_flow(self):
        """
        This test is going all the way to the aggregated calculations
        """

        companies, pf_companies = self.create_base_companies(["A", "B"])

        data_provider = TestDataWareHouse(companies=companies)

        # Calculate scores & Aggregated values
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data(data_provider, pf_companies)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # verify that results exist
        self.assertEqual(agg_scores.long.S1S2.all.score, self.BASE_COMP_SCORE)

    # Run some regression tests
    # @unittest.skip("only run for longer test runs")
    def test_regression_companies(self):

        nr_companies = 1000

        # test 10000 companies
        companies: List[ICompanyAggregates] = []
        pf_companies: List[PortfolioCompany] = []

        for i in range(nr_companies):
            company_id = f"Company {str(i)}"
            # company
            company = copy.deepcopy(self.company_base)
            company.company_id = company_id
            companies.append(company)

            # pf company
            pf_company = PortfolioCompany(
                company_name=company_id,
                company_id=company_id,
                investment_value=100,
                company_isin=company_id,
            )
            pf_companies.append(pf_company)

        data_provider = TestDataWareHouse(companies=companies)

        # Calculate scores & Aggregated values
        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
        )

        portfolio_data = ITR.utils.get_data(data_provider, pf_companies)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        self.assertAlmostEqual(agg_scores.long.S1S2.all.score, self.BASE_COMP_SCORE)

    def test_grouping(self):
        """
        Testing the grouping feature with two different industry levels and making sure the results are present
        """
        # make 2+ companies and group them together
        industry_levels = ["Manufacturer", "Energy"]
        company_ids = ["A", "B"]
        companies_all: List[ICompanyAggregates] = []
        pf_companies_all: List[PortfolioCompany] = []

        for ind_level in industry_levels:

            company_ids_with_level = [f"{ind_level}_{company_id}" for company_id in company_ids]

            companies, pf_companies = self.create_base_companies(company_ids_with_level)
            for company in companies:
                company.industry_level_1 = ind_level

            companies_all.extend(companies)
            pf_companies_all.extend(pf_companies)

        data_provider = TestDataWareHouse(companies=companies_all)

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS,
            grouping=["industry_level_1"]
        )

        portfolio_data = ITR.utils.get_data(data_provider, pf_companies_all)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        for ind_level in industry_levels:
            self.assertAlmostEqual(agg_scores.long.S1S2.grouped[ind_level].score, self.BASE_COMP_SCORE)

    def test_score_cap(self):

        companies, pf_companies = self.create_base_companies(["A"])
        data_provider = TestDataWareHouse(companies=companies)

        temp_score = TemperatureScore(
            time_frames=[ETimeFrames.LONG],
            scopes=[EScope.S1S2],
            aggregation_method=PortfolioAggregationMethod.WATS
        )

        portfolio_data = ITR.utils.get_data(data_provider, pf_companies)
        scores = temp_score.calculate(portfolio_data)
        agg_scores = temp_score.aggregate_scores(scores)

        # add verification

    def create_base_companies(self, company_ids: List[str]):
        """
        This is a helper method to create base companies that can be used for the test cases
        """
        companies: List[ICompanyAggregates] = []
        pf_companies: List[PortfolioCompany] = []
        for company_id in company_ids:
            # company
            company = copy.deepcopy(self.company_base)
            company.company_id = company_id
            companies.append(company)

            # pf company
            pf_company = PortfolioCompany(
                company_name=company_id,
                company_id=company_id,
                investment_value=100,
                company_isin=company_id,
                region='Europe',
                sector='Steel'
            )
            pf_companies.append(pf_company)

        return companies, pf_companies


if __name__ == "__main__":
    test = EndToEndTest()
    test.setUp()
    test.test_basic()
    test.test_basic_flow()
    test.test_regression_companies()
    test.test_score_cap()
    test.test_grouping()
