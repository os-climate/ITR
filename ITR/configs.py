"""
This file defines the constants used throughout the different classes. In order to redefine these settings whilst using
the module, extend the respective config class and pass it to the class as the "constants" parameter.
"""
from .interfaces import TemperatureScoreControls


class ColumnsConfig:
    # Define a constant for each column used in the
    COMPANY_ID = "company_id"
    COMPANY_ISIN = "company_isin"
    COMPANY_ISIC = "isic"
    MARKET_CAP = "company_market_cap"
    INVESTMENT_VALUE = "investment_value"
    COMPANY_ENTERPRISE_VALUE = "company_enterprise_value"
    COMPANY_EV_PLUS_CASH = "company_ev_plus_cash"
    COMPANY_TOTAL_ASSETS = "company_total_assets"
    SCOPE = "scope"
    START_YEAR = "start_year"
    VARIABLE = "variable"
    SLOPE = "slope"
    TIME_FRAME = "time_frame"
    TEMPERATURE_SCORE = "temperature_score"
    COMPANY_NAME = "company_name"
    OWNED_EMISSIONS = "owned_emissions"
    COUNTRY = 'country'
    SECTOR = 'sector'
    GHG_SCOPE12 = 'ghg_s1s2'
    GHG_SCOPE3 = 'ghg_s3'
    COMPANY_REVENUE = 'company_revenue'
    CASH_EQUIVALENTS = 'company_cash_equivalents'
    BASE_YEAR = 'base_year'
    END_YEAR = 'end_year'
    ISIC = 'isic'
    INDUSTRY_LVL1 = "industry_level_1"
    INDUSTRY_LVL2 = "industry_level_2"
    INDUSTRY_LVL3 = "industry_level_3"
    INDUSTRY_LVL4 = "industry_level_4"
    REGION = 'region'
    CUMULATIVE_BUDGET = 'cumulative_budget'
    CUMULATIVE_TRAJECTORY = 'cumulative_trajectory'
    CUMULATIVE_TARGET = 'cumulative_target'
    TARGET_PROBABILITY = 'target_probability'
    BENCHMARK_TEMP = 'benchmark_temperature'
    BENCHMARK_GLOBAL_BUDGET = 'benchmark_global_budget'
    BASE_EI = 'emission_intensity_at_base_year'
    PROJECTED_EI = 'projected_intensities'
    PROJECTED_TARGETS = 'projected_targets'
    HISTORIC_PRODUCTIONS = 'historic_productions'
    HISTORIC_EMISSIONS = 'historic_emissions'
    HISTORIC_EI = 'historic_emission_intensities'
    TRAJECTORY_SCORE = 'trajectory_score'
    TRAJECTORY_OVERSHOOT = 'trajectory_overshoot_ratio'
    TARGET_SCORE = 'target_score'
    TARGET_OVERSHOOT = 'target_overshoot_ratio'

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


class VariablesConfig:
    EMISSIONS = "Emissions"
    PRODUCTIONS = "Productions"
    EMISSION_INTENSITIES = "Emission Intensities"


class TabsConfig:
    FUNDAMENTAL = "fundamental_data"
    PROJECTED_EI = "projected_ei_in_Wh"
    PROJECTED_PRODUCTION = "projected_production"
    PROJECTED_TARGET = "projected_target"
    HISTORIC_DATA = "historic_data"


class PortfolioAggregationConfig:
    COLS = ColumnsConfig


class TemperatureScoreConfig(PortfolioAggregationConfig):
    TEMPERATURE_RESULTS = 'temperature_results'
    CONTROLS_CONFIG = TemperatureScoreControls(
        base_year=2019,
        target_end_year=2050,
        projection_start_year=2010,
        projection_end_year=2019,
        tcre=2.2,
        carbon_conversion=3664.0,
        scenario_target_temperature=1.5
    )


class ProjectionConfig:
    LOWER_PERCENTILE: float = 0.1
    UPPER_PERCENTILE: float = 0.9

    LOWER_DELTA: float = -0.10
    UPPER_DELTA: float = +0.03

    TARGET_YEAR: int = 2050
