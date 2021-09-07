"""
This file defines the constants used throughout the different classes. In order to redefine these settings whilst using
the module, extend the respective config class and pass it to the class as the "constants" parameter.
"""
from .interfaces import TemperatureScoreControls

class TabsConfig:
    FUNDAMENTAL = "fundamental_data"
    TARGET = "target_data"
    PROJECTED_EI = "projected_ei_in_Wh"
    PROJECTED_PRODUCTION = "projected_production"
    PROJECTED_TARGET = "projected_target"

class ColumnsConfig:
    # Define a constant for each column used in the
    COMPANY_ID = "company_id"
    COMPANY_ISIN = "company_isin"
    COMPANY_ISIC = "isic"
    REGRESSION_PARAM = "param"
    REGRESSION_INTERCEPT = "intercept"
    MARKET_CAP = "company_market_cap"
    INVESTMENT_VALUE = "investment_value"
    COMPANY_ENTERPRISE_VALUE = "company_enterprise_value"
    COMPANY_EV_PLUS_CASH = "company_ev_plus_cash"
    COMPANY_TOTAL_ASSETS = "company_total_assets"
    TARGET_REFERENCE_NUMBER = "target_type"
    SCOPE = "scope"
    REDUCTION_FROM_BASE_YEAR = "reduction_from_base_year"
    START_YEAR = "start_year"
    VARIABLE = "variable"
    SLOPE = "slope"
    TIME_FRAME = "time_frame"
    MODEL = "model"
    ANNUAL_REDUCTION_RATE = "annual_reduction_rate"
    EMISSIONS_IN_SCOPE = "emissions_in_scope"
    TEMPERATURE_SCORE = "temperature_score"
    COMPANY_NAME = "company_name"
    OWNED_EMISSIONS = "owned_emissions"
    COUNTRY = 'country'
    SECTOR = 'sector'
    GHG_SCOPE12 = 'ghg_s1s2'
    GHG_SCOPE3 = 'ghg_s3'
    COMPANY_REVENUE = 'company_revenue'
    CASH_EQUIVALENTS = 'company_cash_equivalents'
    TARGET_CLASSIFICATION = 'target_classification'
    REDUCTION_AMBITION = 'reduction_ambition'
    BASE_YEAR = 'base_year'
    END_YEAR = 'end_year'
    SBTI_VALIDATED = 'sbti_validated'
    ACHIEVED_EMISSIONS = "achieved_reduction"
    ISIC = 'isic'
    INDUSTRY_LVL1 = "industry_level_1"
    INDUSTRY_LVL2 = "industry_level_2"
    INDUSTRY_LVL3 = "industry_level_3"
    INDUSTRY_LVL4 = "industry_level_4"
    COVERAGE_S1 = 'coverage_s1'
    COVERAGE_S2 = 'coverage_s2'
    COVERAGE_S3 = 'coverage_s3'
    INTENSITY_METRIC = 'intensity_metric'
    INTENSITY_METRIC_SR15 = 'intensity_metric'
    TARGET_TYPE_SR15 = "target_type"
    SR15_VARIABLE = "sr15_variable"
    REGRESSION_MODEL = 'Regression_model'
    BASEYEAR_GHG_S1 = 'base_year_ghg_s1'
    BASEYEAR_GHG_S2 = 'base_year_ghg_s2'
    BASEYEAR_GHG_S3 = 'base_year_ghg_s3'
    REGION = 'region'
    ENGAGEMENT_TARGET = 'engagement_target'
    CUMULATIVE_BUDGET = 'cumulative_budget'
    CUMULATIVE_TRAJECTORY = 'cumulative_trajectory'
    CUMULATIVE_TARGET = 'cumulative_target'
    TARGET_PROBABILITY = 'target_probability'

    # Output columns
    WEIGHTED_TEMPERATURE_SCORE = "weighted_temperature_score"
    CONTRIBUTION_RELATIVE = "contribution_relative"
    CONTRIBUTION = "contribution"

class SectorsConfig:
    STEEL = "Steel"
    ELECTRICITY = "Electricity Utilities"
    INFORMATION_TECHNOLOGY = "Information Technology"
    INDUSTRIALS = "Industrials"
    FINANCIALS = "Financials"
    HEALTH_CARE = "Health Care"

    CORRECTION_SECTORS = [ELECTRICITY]

class PortfolioAggregationConfig:
    COLS = ColumnsConfig


class TemperatureScoreConfig(PortfolioAggregationConfig):

    """
    """

    TEMPERATURE_RESULTS = 'temperature_results'
    INVESTMENT_VALUE = "investment_value"
    CONTROLS_CONFIG = TemperatureScoreControls(
        base_year = 2019,
        target_end_year = 2050,
        projection_start_year = 2010,
        projection_end_year = 2019,
        tcre = 2.2,
        carbon_conversion = 3664.0,
        scenario_target_temperature = 1.5,
        global_budget = 396,
        current_temperature = 1.5,
        energy_unit_conversion_factor = 3.6
    )



